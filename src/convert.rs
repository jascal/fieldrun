//! `fieldrun convert` — turn a Hugging Face model into a fieldrun bundle, in pure Rust, no torch/Python.
//!
//! Reads the model's `safetensors` (mmapped, via HF's own Rust crate — single-file or sharded via the index.json) +
//! `config.json`, transposes/quantises each tensor, and streams it straight into the bundle blob — so RAM ≈ one tensor
//! at a time, not the whole model. The build-side counterpart of the runtime: the whole pipeline (convert + run) is now
//! framework-free. Bit-identical to the torch export (int8 uses round-ties-even to match numpy).

use std::collections::HashMap;
use std::io::Write;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// A model's weights, possibly sharded. mmaps each file (address space, not RAM); tensors are read on demand.
struct Model {
    mmaps: Vec<Mmap>,
    idx: HashMap<String, usize>, // tensor name -> mmap index
}

impl Model {
    fn open(dir: &str) -> Model {
        let index = format!("{dir}/model.safetensors.index.json");
        let (mmaps, idx) = if std::path::Path::new(&index).exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
            let wm = v["weight_map"].as_object().expect("weight_map");
            let mut files: Vec<String> = wm.values().filter_map(|f| f.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            let file_idx: HashMap<&String, usize> = files.iter().enumerate().map(|(i, f)| (f, i)).collect();
            let mmaps: Vec<Mmap> = files.iter().map(|f| mmap(&format!("{dir}/{f}"))).collect();
            let idx = wm.iter().map(|(k, f)| (k.clone(), file_idx[&f.as_str().unwrap().to_string()])).collect();
            (mmaps, idx)
        } else {
            let mm = mmap(&format!("{dir}/model.safetensors"));
            let names: Vec<String> = SafeTensors::deserialize(&mm).unwrap().names().into_iter().map(String::from).collect();
            (vec![mm], names.into_iter().map(|n| (n, 0)).collect())
        };
        Model { mmaps, idx }
    }

    fn has(&self, name: &str) -> bool {
        self.idx.contains_key(name)
    }

    fn read(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let i = *self.idx.get(name).unwrap_or_else(|| panic!("convert: missing tensor {name}"));
        let st = SafeTensors::deserialize(&self.mmaps[i]).unwrap();
        let t = st.tensor(name).unwrap();
        let b = t.data();
        let v: Vec<f32> = match t.dtype() {
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => panic!("convert: unsupported dtype {d:?} for {name}"),
        };
        (t.shape().to_vec(), v)
    }
}

fn mmap(path: &str) -> Mmap {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("convert: open {path}: {e}"));
    unsafe { Mmap::map(&f).unwrap() }
}

struct BundleWriter {
    bin: std::io::BufWriter<std::fs::File>,
    arrays: Vec<serde_json::Value>,
    offset: usize,
}

impl BundleWriter {
    fn new(stem: &str) -> std::io::Result<BundleWriter> {
        Ok(BundleWriter { bin: std::io::BufWriter::new(std::fs::File::create(format!("{stem}.fieldrun.bin"))?), arrays: Vec::new(), offset: 0 })
    }

    fn entry(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: usize) {
        self.arrays.push(serde_json::json!({ "name": name, "dtype": dtype, "shape": shape, "offset": self.offset, "bytes": bytes }));
        self.offset += bytes;
    }

    fn put_f16(&mut self, name: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(data.len() * 2);
        for &v in data {
            buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        self.bin.write_all(&buf)?;
        self.entry(name, "f16", shape, buf.len());
        Ok(())
    }

    /// int8 from a (rows, cols) f32 source, scale per output column `j`. `transpose`: source is (out, in) (nn.Linear) →
    /// store (in, out); else source is already (in, out) (GPT-2 Conv1D) → store as-is.
    fn put_i8(&mut self, name: &str, data: &[f32], rows: usize, cols: usize, transpose: bool) -> std::io::Result<()> {
        let (out, inp) = if transpose { (rows, cols) } else { (cols, rows) };
        let at = |i: usize, j: usize| if transpose { data[j * inp + i] } else { data[i * out + j] };
        let mut scale = vec![0f32; out];
        for (j, sc) in scale.iter_mut().enumerate() {
            *sc = ((0..inp).fold(0f32, |m, i| m.max(at(i, j).abs())) / 127.0).max(1e-8);
        }
        let mut wt = vec![0u8; inp * out];
        for i in 0..inp {
            for (j, &s) in scale.iter().enumerate() {
                wt[i * out + j] = ((at(i, j) / s).round_ties_even().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        self.bin.write_all(&wt)?;
        self.entry(name, "i8", &[inp, out], wt.len());
        self.put_f16(&format!("{name}__scale"), &scale, &[out])
    }

    fn finish(self, stem: &str, mut manifest: serde_json::Value) -> std::io::Result<()> {
        manifest["arrays"] = serde_json::Value::Array(self.arrays);
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest)?)
    }
}

fn geti(c: &serde_json::Value, k: &str) -> Option<usize> {
    c.get(k).and_then(|v| v.as_u64()).map(|v| v as usize)
}
fn getf(c: &serde_json::Value, k: &str) -> Option<f64> {
    c.get(k).and_then(|v| v.as_f64())
}

pub fn convert(model_dir: &str, arch: &str, dtype: &str, out_stem: &str) -> std::io::Result<()> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{model_dir}/config.json"))?)?;
    let m = Model::open(model_dir);
    let shards = m.mmaps.len();
    let n = match arch {
        "gpt2" => convert_gpt2(&cfg, &m, dtype, out_stem)?,
        "rope" => convert_rope(&cfg, &m, dtype, out_stem)?,
        "gemma" => convert_gemma(&cfg, &m, dtype, out_stem)?,
        other => panic!("convert: arch {other:?} not supported (gpt2, rope, gemma)"),
    };
    println!("[convert] {n} arrays -> {out_stem}.fieldrun.json/.bin (arch={arch}, dtype={dtype}, {shards} shard(s), no torch)");
    Ok(())
}

fn convert_gpt2(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    let (nl, nh, d) = (geti(c, "n_layer").unwrap(), geti(c, "n_head").unwrap(), geti(c, "n_embd").unwrap());
    let (npos, vocab) = (geti(c, "n_positions").unwrap(), geti(c, "vocab_size").unwrap());
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "gpt2",
        "config": [nl, nh, d, npos, vocab] });
    let mut w = BundleWriter::new(stem)?;
    let i8 = dtype == "int8";
    let pre = if m.has("transformer.wte.weight") { "transformer." } else { "" }; // GPT2LMHeadModel vs bare state dict
    let f16 = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_f16(name, &dt, &s)
    };
    // wte/wpe/ln_f kept f16; Conv1D weights (already (in,out)) int8 without transpose
    f16(&mut w, "wte", &format!("{pre}wte.weight"))?;
    f16(&mut w, "wpe", &format!("{pre}wpe.weight"))?;
    f16(&mut w, "ln_f.weight", &format!("{pre}ln_f.weight"))?;
    f16(&mut w, "ln_f.bias", &format!("{pre}ln_f.bias"))?;
    for l in 0..nl {
        let p = format!("{pre}h.{l}.");
        f16(&mut w, &format!("h{l}.ln_1.weight"), &format!("{p}ln_1.weight"))?;
        f16(&mut w, &format!("h{l}.ln_1.bias"), &format!("{p}ln_1.bias"))?;
        f16(&mut w, &format!("h{l}.ln_2.weight"), &format!("{p}ln_2.weight"))?;
        f16(&mut w, &format!("h{l}.ln_2.bias"), &format!("{p}ln_2.bias"))?;
        for (fr, hf) in [("attn.c_attn", "attn.c_attn"), ("attn.c_proj", "attn.c_proj"), ("mlp.c_fc", "mlp.c_fc"), ("mlp.c_proj", "mlp.c_proj")] {
            let (s, dt) = m.read(&format!("{p}{hf}.weight"));
            if i8 { w.put_i8(&format!("h{l}.{fr}.weight"), &dt, s[0], s[1], false)?; } else { w.put_f16(&format!("h{l}.{fr}.weight"), &dt, &s)?; }
            f16(&mut w, &format!("h{l}.{fr}.bias"), &format!("{p}{hf}.bias"))?;
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

fn rope_theta_eps(c: &serde_json::Value) -> (f64, f64) {
    let theta = getf(c, "rope_theta")
        .or_else(|| c.get("rope_parameters").and_then(|v| v.get("rope_theta")).and_then(|t| t.as_f64()))
        .unwrap_or(10000.0);
    (theta, getf(c, "rms_norm_eps").unwrap_or(1e-6))
}

// shared: write the linear/embed/bias arrays for a Llama/Qwen/Gemma layer stack. `norm_offset` adds 1.0 to norm
// weights (Gemma's x·(1+w)). `norms` lists the (fieldrun, hf) RMSNorm names per layer.
fn convert_rope(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let (theta, eps) = rope_theta_eps(c);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "rope",
        "config": [nl, nh, nkv, hd, d, ffn, vocab, tie as usize], "config_f": [theta, eps] });
    let mut w = BundleWriter::new(stem)?;
    write_layers(&mut w, c, m, dtype, nl, tie, &[("in_ln", "input_layernorm"), ("post_ln", "post_attention_layernorm")], false)?;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

fn convert_gemma(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let (theta, eps) = rope_theta_eps(c);
    let attn_cap = getf(c, "attn_logit_softcapping").unwrap_or(0.0);
    let final_cap = getf(c, "final_logit_softcapping").unwrap_or(0.0);
    let qscalar = getf(c, "query_pre_attn_scalar").unwrap_or(hd as f64);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true);
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "gemma",
        "config": [nl, nh, nkv, hd, d, ffn, vocab, tie as usize],
        "config_f": [theta, eps, attn_cap, final_cap, qscalar, (d as f64).sqrt()] });
    let mut w = BundleWriter::new(stem)?;
    let norms = [("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm"),
                 ("pre_feedforward_layernorm", "pre_feedforward_layernorm"), ("post_feedforward_layernorm", "post_feedforward_layernorm")];
    write_layers(&mut w, c, m, dtype, nl, tie, &norms, true)?;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// Shared Llama/Qwen/Gemma writer: embed (f16) + final norm + per-layer norms (with optional +1 bake) + the
/// q/k/v/o/gate/up/down Linears (transposed, int8 or f16) + optional q/k/v bias.
fn write_layers(w: &mut BundleWriter, c: &serde_json::Value, m: &Model, dtype: &str, nl: usize, tie: bool,
                norms: &[(&str, &str)], bake1: bool) -> std::io::Result<()> {
    let i8 = dtype == "int8";
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, mut dt) = m.read(hf);
        if bake1 { for v in dt.iter_mut() { *v += 1.0; } } // Gemma RMSNorm: x·(1+w)
        w.put_f16(name, &dt, &s)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); // (out, in)
        if i8 { w.put_i8(name, &dt, s[0], s[1], true) } else {
            let (out, inp) = (s[0], s[1]);
            let mut t = vec![0f32; inp * out];
            for o in 0..out { for i in 0..inp { t[i * out + o] = dt[o * inp + i]; } }
            w.put_f16(name, &t, &[inp, out])
        }
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_f16("embed", &ed, &es)?;
    norm(w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); // (vocab,d) -> (d,vocab) f16
        let mut t = vec![0f32; s[0] * s[1]];
        for o in 0..s[0] { for i in 0..s[1] { t[i * s[0] + o] = dt[o * s[1] + i]; } }
        w.put_f16("lm_head", &t, &[s[1], s[0]])?;
    }
    let _ = c;
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        // norms: (fieldrun name, HF name) — rope renames to in_ln/post_ln; gemma keeps the 4 HF norm names
        for (frn, hfn) in norms {
            norm(w, &format!("l{l}.{frn}"), &format!("{p}{hfn}.weight"))?;
        }
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            lin(w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
            if m.has(&format!("{p}{proj}.bias")) {
                let (s, dt) = m.read(&format!("{p}{proj}.bias"));
                w.put_f16(&format!("l{l}.{proj}.bias"), &dt, &s)?;
            }
        }
    }
    Ok(())
}
