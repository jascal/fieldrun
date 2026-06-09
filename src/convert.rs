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

    fn put_f32(&mut self, name: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.bin.write_all(&buf)?;
        self.entry(name, "f32", shape, buf.len());
        Ok(())
    }

    /// A "small" 1D/2D array (embed, norms, biases, lm_head): f32 when `dtype=="f32"`, else f16. Big linears keep their
    /// own int8/f16/f32 path (`put_i8`/`put_lin`); the f32 dtype gives a bit-exact bundle for the faithfulness gate.
    fn put_small(&mut self, name: &str, data: &[f32], shape: &[usize], dtype: &str) -> std::io::Result<()> {
        if dtype == "f32" { self.put_f32(name, data, shape) } else { self.put_f16(name, data, shape) }
    }

    /// A weight linear from an (out, in) source: int8 (transposed, per-column), or f32/f16 transposed to (in, out).
    fn put_lin(&mut self, name: &str, data: &[f32], out: usize, inp: usize, dtype: &str) -> std::io::Result<()> {
        if dtype == "int8" { return self.put_i8(name, data, out, inp, true); }
        let mut t = vec![0f32; inp * out];
        for o in 0..out { for i in 0..inp { t[i * out + o] = data[o * inp + i]; } }
        self.put_small(name, &t, &[inp, out], dtype)
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
        "gemma3" => convert_gemma3(&cfg, &m, dtype, out_stem)?,
        "gemma4" => convert_gemma4(&cfg, &m, dtype, out_stem)?,
        "qwen3moe" => convert_qwen3moe(&cfg, &m, dtype, out_stem)?,
        "mla" => convert_mla(&cfg, &m, dtype, out_stem)?,
        other => panic!("convert: arch {other:?} not supported (gpt2, rope, gemma, gemma3, gemma4, qwen3moe, mla)"),
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
    let sml = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    // wte/wpe/ln_f small (f16 or f32); Conv1D weights (already (in,out)) int8 without transpose, else small as-is
    sml(&mut w, "wte", &format!("{pre}wte.weight"))?;
    sml(&mut w, "wpe", &format!("{pre}wpe.weight"))?;
    sml(&mut w, "ln_f.weight", &format!("{pre}ln_f.weight"))?;
    sml(&mut w, "ln_f.bias", &format!("{pre}ln_f.bias"))?;
    for l in 0..nl {
        let p = format!("{pre}h.{l}.");
        sml(&mut w, &format!("h{l}.ln_1.weight"), &format!("{p}ln_1.weight"))?;
        sml(&mut w, &format!("h{l}.ln_1.bias"), &format!("{p}ln_1.bias"))?;
        sml(&mut w, &format!("h{l}.ln_2.weight"), &format!("{p}ln_2.weight"))?;
        sml(&mut w, &format!("h{l}.ln_2.bias"), &format!("{p}ln_2.bias"))?;
        for (fr, hf) in [("attn.c_attn", "attn.c_attn"), ("attn.c_proj", "attn.c_proj"), ("mlp.c_fc", "mlp.c_fc"), ("mlp.c_proj", "mlp.c_proj")] {
            let (s, dt) = m.read(&format!("{p}{hf}.weight"));
            if i8 { w.put_i8(&format!("h{l}.{fr}.weight"), &dt, s[0], s[1], false)?; } else { w.put_small(&format!("h{l}.{fr}.weight"), &dt, &s, dtype)?; }
            sml(&mut w, &format!("h{l}.{fr}.bias"), &format!("{p}{hf}.bias"))?;
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

/// Gemma 3: the Gemma-2 stack plus QK-norm (per-head RMSNorm on q/k), dual-base RoPE (local θ for sliding layers,
/// global θ for full layers), a 5:1 sliding:full layer pattern, and NO logit soft-capping. head_dim is shared across
/// layer types (unlike Gemma 4). Per-layer sliding flags (from `layer_types`) are packed into `config` so the kernel
/// needn't re-derive the pattern. `config_f` carries both RoPE bases.
fn convert_gemma3(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let eps = getf(c, "rms_norm_eps").unwrap_or(1e-6);
    let qscalar = getf(c, "query_pre_attn_scalar").unwrap_or(hd as f64);
    let window = geti(c, "sliding_window").unwrap_or(4096);
    let pattern = geti(c, "sliding_window_pattern").unwrap_or(6);
    let (theta_local, theta_global) = gemma3_thetas(c);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true);
    // per-layer sliding flag: prefer the serialized `layer_types` list, else derive ((i+1)%pattern != 0), last forced full
    let lt = c.get("layer_types").and_then(|v| v.as_array());
    let mut config: Vec<usize> = vec![nl, nh, nkv, hd, d, ffn, vocab, tie as usize, window];
    for l in 0..nl {
        // Gemma 3 (unlike Gemma 4) does NOT force the last layer to full — the pattern stands as-is.
        let full = lt.and_then(|a| a.get(l)).and_then(|s| s.as_str())
            .map(|s| s == "full_attention")
            .unwrap_or((l + 1) % pattern == 0);
        config.push(if full { 0 } else { 1 });
    }
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "gemma3",
        "config": config, "config_f": [theta_local, theta_global, eps, qscalar, (d as f64).sqrt()] });
    let mut w = BundleWriter::new(stem)?;
    let norms = [("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm"),
                 ("pre_feedforward_layernorm", "pre_feedforward_layernorm"), ("post_feedforward_layernorm", "post_feedforward_layernorm"),
                 ("self_attn.q_norm", "self_attn.q_norm"), ("self_attn.k_norm", "self_attn.k_norm")];
    write_layers(&mut w, c, m, dtype, nl, tie, &norms, true)?;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

// Gemma 3 dual RoPE bases: sliding (local) layers vs full (global) layers. Accept both the new `rope_parameters`
// nesting and the legacy flat `rope_local_base_freq` / `rope_theta`.
fn gemma3_thetas(c: &serde_json::Value) -> (f64, f64) {
    let nested = |kind: &str| c.get("rope_parameters").and_then(|v| v.get(kind)).and_then(|t| t.get("rope_theta")).and_then(|t| t.as_f64());
    let local = nested("sliding_attention").or_else(|| getf(c, "rope_local_base_freq")).unwrap_or(10_000.0);
    let global = nested("full_attention").or_else(|| getf(c, "rope_theta")).unwrap_or(1_000_000.0);
    (local, global)
}

/// Gemma 4 (dense text path: PLE on, MoE off). Adds to the Gemma-3 backbone: RMSNorm uses the weight *directly*
/// (NOT (1+w) — Gemma 4 inits norm weights to 1.0), value-norm (RMS, no weight → no array), attention scaling = 1.0,
/// a *different* head_dim on global layers (so q/k/v/o shapes differ per layer type), partial-rotary "proportional"
/// RoPE on global layers (handled in the kernel by zero-padding inv_freq), and the Per-Layer-Embedding gated-residual
/// block. MoE / attention_k_eq_v / KV-sharing are separate follow-on increments (this asserts they're off).
fn convert_gemma4(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    assert!(!c.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false), "gemma4: attention_k_eq_v not yet supported");
    assert_eq!(geti(c, "num_kv_shared_layers").unwrap_or(0), 0, "gemma4: KV-sharing not yet supported");
    let moe = c.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false);
    let n_exp = geti(c, "num_experts").unwrap_or(0);
    let topk = geti(c, "top_k_experts").unwrap_or(0);
    let moe_inter = geti(c, "moe_intermediate_size").unwrap_or(0);
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let nkv_g = geti(c, "num_global_key_value_heads").unwrap_or(nkv);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let hd_g = geti(c, "global_head_dim").unwrap_or(hd);
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let ple = geti(c, "hidden_size_per_layer_input").unwrap_or(256);
    let eps = getf(c, "rms_norm_eps").unwrap_or(1e-6);
    let window = geti(c, "sliding_window").unwrap_or(512);
    let pattern = geti(c, "sliding_window_pattern").unwrap_or(6);
    let (theta_local, theta_global) = gemma3_thetas(c);
    let prf = c.get("rope_parameters").and_then(|v| v.get("full_attention")).and_then(|t| t.get("partial_rotary_factor"))
        .and_then(|t| t.as_f64()).unwrap_or(0.25);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true);
    let lt = c.get("layer_types").and_then(|v| v.as_array());
    let full_of = |l: usize| lt.and_then(|a| a.get(l)).and_then(|s| s.as_str())
        .map(|s| s == "full_attention").unwrap_or((l + 1) % pattern == 0);
    // Gemma 4 forces the last layer to full_attention.
    let is_full = |l: usize| full_of(l) || l + 1 == nl;
    let mut config: Vec<usize> = vec![nl, nh, nkv, nkv_g, hd, hd_g, d, ffn, vocab, tie as usize, window, ple,
                                      moe as usize, n_exp, topk, moe_inter];
    for l in 0..nl { config.push(if is_full(l) { 0 } else { 1 }); } // sliding flags start at config[16]
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "gemma4",
        "config": config, "config_f": [theta_local, theta_global, eps, prf] });

    let mut w = BundleWriter::new(stem)?;
    let i8 = dtype == "int8";
    // RMSNorm: Gemma 4 uses the weight directly (no +1 bake).
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); // (out, in)
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    // main + PLE embeddings (both f16/f32, never int8 — embed stays low-precision)
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_small("embed", &ed, &es, dtype)?;
    let (es2, ed2) = m.read("model.embed_tokens_per_layer.weight"); // (vocab_per_layer, nl*ple)
    w.put_small("embed_per_layer", &ed2, &es2, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    norm(&mut w, "per_layer_projection_norm", "model.per_layer_projection_norm.weight")?;
    // per_layer_model_projection: Linear(d -> nl*ple); the int8 W8A8 path needs the weight, so keep it f16/f32 like a norm
    {
        let (s, dt) = m.read("model.per_layer_model_projection.weight"); // (nl*ple, d)
        let (out, inp) = (s[0], s[1]);
        let mut t = vec![0f32; inp * out];
        for o in 0..out { for i in 0..inp { t[i * out + o] = dt[o * inp + i]; } }
        w.put_small("per_layer_model_projection", &t, &[inp, out], dtype)?;
    }
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); // (vocab, d) — raw for rowdot_f32, low-precision
        w.put_small("lm_head", &dt, &s, dtype)?;
    }
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        for nm in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
                   "self_attn.q_norm", "self_attn.k_norm", "post_per_layer_input_norm"] {
            norm(&mut w, &format!("l{l}.{nm}"), &format!("{p}{nm}.weight"))?;
        }
        // v_norm has with_scale=False (no weight) → nothing to write.
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "per_layer_input_gate", "per_layer_projection"] {
            lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        if moe {
            // extra MoE norms (RMSNorm, no bake); router.norm is with_scale=False (no weight).
            for nm in ["post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"] {
                norm(&mut w, &format!("l{l}.{nm}"), &format!("{p}{nm}.weight"))?;
            }
            // router: proj (Linear d->E), scale (d), per_expert_scale (E) — all small, f16/f32
            lin(&mut w, &format!("l{l}.router.proj"), &format!("{p}router.proj.weight"))?;
            let (ss, sd) = m.read(&format!("{p}router.scale"));
            w.put_small(&format!("l{l}.router.scale"), &sd, &ss, dtype)?;
            let (ps, pd) = m.read(&format!("{p}router.per_expert_scale"));
            w.put_small(&format!("l{l}.router.per_expert_scale"), &pd, &ps, dtype)?;
            // experts: gate_up_proj (E, 2*moe_inter, d), down_proj (E, d, moe_inter) — write EACH expert as its own
            // int8 array so a single expert can be paged in independently (the mmap-offload contract).
            let (gus, gud) = m.read(&format!("{p}experts.gate_up_proj")); // (E, 2*mi, d)
            let (dns, dnd) = m.read(&format!("{p}experts.down_proj"));     // (E, d, mi)
            let (gu_out, gu_in) = (gus[1], gus[2]); // (2*mi, d)
            let (dn_out, dn_in) = (dns[1], dns[2]); // (d, mi)
            for e in 0..n_exp {
                let gu = &gud[e * gu_out * gu_in..(e + 1) * gu_out * gu_in];
                w.put_lin(&format!("l{l}.experts.{e}.gate_up"), gu, gu_out, gu_in, dtype)?;
                let dn = &dnd[e * dn_out * dn_in..(e + 1) * dn_out * dn_in];
                w.put_lin(&format!("l{l}.experts.{e}.down"), dn, dn_out, dn_in, dtype)?;
            }
        }
    }
    let _ = i8;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// Qwen3-MoE: the RoPE backbone (RMSNorm, single-base RoPE, GQA, SwiGLU) + QK-norm (per-head RMSNorm on q/k) + a
/// per-layer MoE-or-dense FFN. The MoE block is a plain-gate router (softmax → top-k → optional renorm, no scales) over
/// packed experts (same gate_up/down 3D layout as Gemma-4); the router+experts run on the post-attention-normed hidden.
/// Experts are written one int8 array each (offload), so this reaches Qwen3-MoE on a ≤24 GB box. No attention bias
/// (Qwen3 dropped it), no embed scale, no soft-capping. Sliding window (use_sliding_window) is a follow-on.
fn convert_qwen3moe(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    assert!(!c.get("use_sliding_window").and_then(|v| v.as_bool()).unwrap_or(false), "qwen3moe: sliding window not yet supported");
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let (theta, eps) = rope_theta_eps(c);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let n_exp = geti(c, "num_experts").or_else(|| geti(c, "num_local_experts")).unwrap_or(0); // Qwen serializes num_local_experts
    let topk = geti(c, "num_experts_per_tok").unwrap_or(0);
    let moe_inter = geti(c, "moe_intermediate_size").unwrap_or(0);
    let norm_topk = c.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(false);
    let sparse_step = geti(c, "decoder_sparse_step").unwrap_or(1);
    let mlp_only: Vec<usize> = c.get("mlp_only_layers").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect()).unwrap_or_default();
    let is_moe = |l: usize| !mlp_only.contains(&l) && n_exp > 0 && (l + 1) % sparse_step == 0;
    let mut config: Vec<usize> = vec![nl, nh, nkv, hd, d, ffn, vocab, tie as usize, n_exp, topk, moe_inter, norm_topk as usize];
    for l in 0..nl { config.push(if is_moe(l) { 1 } else { 0 }); } // MoE flags start at config[12]
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "qwen3moe",
        "config": config, "config_f": [theta, eps] });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); // standard RMSNorm, weight used directly (no bake)
        w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_small("embed", &ed, &es, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); // (vocab, d) — raw for rowdot_f32, low-precision
        w.put_small("lm_head", &dt, &s, dtype)?;
    }
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        norm(&mut w, &format!("l{l}.in_ln"), &format!("{p}input_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.post_ln"), &format!("{p}post_attention_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.q_norm"), &format!("{p}self_attn.q_norm.weight"))?;
        norm(&mut w, &format!("l{l}.k_norm"), &format!("{p}self_attn.k_norm.weight"))?;
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
            lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        if is_moe(l) {
            lin(&mut w, &format!("l{l}.gate"), &format!("{p}mlp.gate.weight"))?; // router (n_exp, d) -> (d, n_exp)
            // Qwen3-MoE checkpoints store each expert as separate gate_proj/up_proj/down_proj Linears (not packed).
            for e in 0..n_exp {
                for (fr, hf) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] {
                    lin(&mut w, &format!("l{l}.experts.{e}.{fr}"), &format!("{p}mlp.experts.{e}.{hf}.weight"))?;
                }
            }
        } else {
            for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
            }
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// DeepSeek-V3 / Kimi-K2 — MLA (multi-head latent attention) + DeepSeek MoE. MLA compresses q and kv through low-rank
/// down→up projections (q_a/q_b, kv_a/kv_b) with a 128-dim no-RoPE part and a 64-dim shared decoupled-RoPE part, and a
/// distinct v_head_dim. The MoE has a shared always-on expert plus group-limited sigmoid routing (with a learned bias
/// correction). The first `first_k_dense_replace` layers are dense. Experts written one int8 array each (offload).
fn convert_mla(c: &serde_json::Value, m: &Model, dtype: &str, stem: &str) -> std::io::Result<usize> {
    let nl = geti(c, "num_hidden_layers").unwrap();
    let nh = geti(c, "num_attention_heads").unwrap();
    let d = geti(c, "hidden_size").unwrap();
    let q_lora = geti(c, "q_lora_rank").unwrap_or(0);
    let kv_lora = geti(c, "kv_lora_rank").unwrap();
    let qk_nope = geti(c, "qk_nope_head_dim").unwrap();
    let qk_rope = geti(c, "qk_rope_head_dim").unwrap();
    let v_head = geti(c, "v_head_dim").unwrap();
    let vocab = geti(c, "vocab_size").unwrap();
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let n_routed = geti(c, "n_routed_experts").or_else(|| geti(c, "num_local_experts")).unwrap();
    let n_shared = geti(c, "n_shared_experts").unwrap_or(1);
    let topk = geti(c, "num_experts_per_tok").unwrap();
    let moe_inter = geti(c, "moe_intermediate_size").unwrap();
    let n_group = geti(c, "n_group").unwrap_or(1);
    let topk_group = geti(c, "topk_group").unwrap_or(1);
    let norm_topk = c.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true);
    let first_k = geti(c, "first_k_dense_replace").unwrap_or(0);
    let ffn_dense = geti(c, "intermediate_size").unwrap();
    let (theta, eps) = rope_theta_eps(c);
    let routed_scaling = getf(c, "routed_scaling_factor").unwrap_or(1.0);
    let config: Vec<usize> = vec![nl, nh, d, q_lora, kv_lora, qk_nope, qk_rope, v_head, vocab, tie as usize,
        n_routed, n_shared, topk, moe_inter, n_group, topk_group, norm_topk as usize, first_k, ffn_dense];
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "mla",
        "config": config, "config_f": [theta, eps, routed_scaling] });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_small(name, &dt, &s, dtype) // standard RMSNorm, weight used directly
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_small("embed", &ed, &es, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); w.put_small("lm_head", &dt, &s, dtype)?;
    }
    // experts ship either packed (experts.gate_up_proj/down_proj 3D) or per-expert Linears — write per-expert gate/up/down either way.
    let mut write_experts = |w: &mut BundleWriter, p: &str, l: usize| -> std::io::Result<()> {
        let packed = format!("{p}mlp.experts.gate_up_proj");
        if m.has(&packed) {
            let (gus, gud) = m.read(&packed);        // (E, 2*mi, d)
            let (dns, dnd) = m.read(&format!("{p}mlp.experts.down_proj")); // (E, d, mi)
            let (gu_out, gu_in) = (gus[1], gus[2]);
            let (dn_out, dn_in) = (dns[1], dns[2]);
            let mi = gu_out / 2;
            for e in 0..n_routed {
                let base = e * gu_out * gu_in;
                w.put_lin(&format!("l{l}.experts.{e}.gate"), &gud[base..base + mi * gu_in], mi, gu_in, dtype)?;
                w.put_lin(&format!("l{l}.experts.{e}.up"), &gud[base + mi * gu_in..base + gu_out * gu_in], mi, gu_in, dtype)?;
                let db = e * dn_out * dn_in;
                w.put_lin(&format!("l{l}.experts.{e}.down"), &dnd[db..db + dn_out * dn_in], dn_out, dn_in, dtype)?;
            }
        } else {
            for e in 0..n_routed {
                for (fr, hf) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] {
                    let (s, dt) = m.read(&format!("{p}mlp.experts.{e}.{hf}.weight"));
                    w.put_lin(&format!("l{l}.experts.{e}.{fr}"), &dt, s[0], s[1], dtype)?;
                }
            }
        }
        Ok(())
    };
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        norm(&mut w, &format!("l{l}.in_ln"), &format!("{p}input_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.post_ln"), &format!("{p}post_attention_layernorm.weight"))?;
        // MLA projections
        if q_lora > 0 {
            lin(&mut w, &format!("l{l}.q_a"), &format!("{p}self_attn.q_a_proj.weight"))?;
            norm(&mut w, &format!("l{l}.q_a_ln"), &format!("{p}self_attn.q_a_layernorm.weight"))?;
            lin(&mut w, &format!("l{l}.q_b"), &format!("{p}self_attn.q_b_proj.weight"))?;
        } else {
            lin(&mut w, &format!("l{l}.q"), &format!("{p}self_attn.q_proj.weight"))?;
        }
        lin(&mut w, &format!("l{l}.kv_a"), &format!("{p}self_attn.kv_a_proj_with_mqa.weight"))?;
        norm(&mut w, &format!("l{l}.kv_a_ln"), &format!("{p}self_attn.kv_a_layernorm.weight"))?;
        lin(&mut w, &format!("l{l}.kv_b"), &format!("{p}self_attn.kv_b_proj.weight"))?;
        lin(&mut w, &format!("l{l}.o_proj"), &format!("{p}self_attn.o_proj.weight"))?;
        // FFN: dense for the first k layers, else MoE (routed experts + shared expert)
        if l < first_k {
            for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
            }
        } else {
            lin(&mut w, &format!("l{l}.gate"), &format!("{p}mlp.gate.weight"))?; // router (n_routed, d) -> (d, n_routed)
            let (bs, bd) = m.read(&format!("{p}mlp.gate.e_score_correction_bias"));
            w.put_small(&format!("l{l}.gate_bias"), &bd, &bs, dtype)?;
            write_experts(&mut w, &p, l)?;
            for (fr, hf) in [("shared.gate", "gate_proj"), ("shared.up", "up_proj"), ("shared.down", "down_proj")] {
                lin(&mut w, &format!("l{l}.{fr}"), &format!("{p}mlp.shared_experts.{hf}.weight"))?;
            }
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// Shared Llama/Qwen/Gemma writer: embed (f16) + final norm + per-layer norms (with optional +1 bake) + the
/// q/k/v/o/gate/up/down Linears (transposed, int8 or f16) + optional q/k/v bias.
fn write_layers(w: &mut BundleWriter, c: &serde_json::Value, m: &Model, dtype: &str, nl: usize, tie: bool,
                norms: &[(&str, &str)], bake1: bool) -> std::io::Result<()> {
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, mut dt) = m.read(hf);
        if bake1 { for v in dt.iter_mut() { *v += 1.0; } } // Gemma RMSNorm: x·(1+w)
        w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); // (out, in)
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_small("embed", &ed, &es, dtype)?;
    norm(w, "norm", "model.norm.weight")?;
    if !tie {
        // unembed is read row-wise by rowdot_f32 as (vocab, d) → store raw (NOT transposed), low-precision like embed
        let (s, dt) = m.read("lm_head.weight"); // (vocab, d)
        w.put_small("lm_head", &dt, &s, dtype)?;
    }
    let _ = c;
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        // norms: (fieldrun name, HF name) — rope renames to in_ln/post_ln; gemma keeps the HF norm names; gemma3 also
        // carries per-head self_attn.q_norm / self_attn.k_norm (QK-norm), which fit the same (1+w)-baked path.
        for (frn, hfn) in norms {
            norm(w, &format!("l{l}.{frn}"), &format!("{p}{hfn}.weight"))?;
        }
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            lin(w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
            if m.has(&format!("{p}{proj}.bias")) {
                let (s, dt) = m.read(&format!("{p}{proj}.bias"));
                w.put_small(&format!("l{l}.{proj}.bias"), &dt, &s, dtype)?;
            }
        }
    }
    Ok(())
}
