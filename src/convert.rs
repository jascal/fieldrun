//! `fieldrun convert` — turn a Hugging Face model into a fieldrun bundle, in pure Rust, no torch/Python.
//!
//! Reads the model's `safetensors` (mmapped, via HF's own Rust crate) + `config.json`, transposes/quantises each
//! tensor, and streams it straight into the bundle blob — so RAM ≈ one tensor at a time, not the whole model (a 24 GB
//! machine can convert a model far larger than would fit in torch). This is the build-side counterpart of the runtime:
//! the whole pipeline (convert + run) is now framework-free. Validated by top-1 agreement vs the torch-exported bundle.

use std::io::Write;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

#[derive(Deserialize)]
struct HfConfig {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

/// Streams the bundle blob to disk and accumulates the manifest, so nothing larger than one tensor is held in RAM.
struct BundleWriter {
    bin: std::io::BufWriter<std::fs::File>,
    arrays: Vec<serde_json::Value>,
    offset: usize,
}

impl BundleWriter {
    fn new(stem: &str) -> std::io::Result<BundleWriter> {
        let bin = std::io::BufWriter::new(std::fs::File::create(format!("{stem}.fieldrun.bin"))?);
        Ok(BundleWriter { bin, arrays: Vec::new(), offset: 0 })
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

    fn put_i8(&mut self, name: &str, w_outin: &[f32], out: usize, inp: usize) -> std::io::Result<()> {
        // per-output-column int8 (matches the runtime): scale[j] = max_i |W[j,i]|/127; store transposed (in, out).
        let mut scale = vec![0f32; out];
        for (j, sc) in scale.iter_mut().enumerate() {
            let mx = (0..inp).fold(0f32, |m, i| m.max(w_outin[j * inp + i].abs()));
            *sc = (mx / 127.0).max(1e-8);
        }
        let mut wt = vec![0u8; inp * out];
        for i in 0..inp {
            for (j, &s) in scale.iter().enumerate() {
                // round-ties-to-even to match numpy's np.round (so the bundle is bit-identical to the torch export)
                wt[i * out + j] = ((w_outin[j * inp + i] / s).round_ties_even().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        self.bin.write_all(&wt)?;
        self.entry(name, "i8", &[inp, out], wt.len());
        self.put_f16(&format!("{name}__scale"), &scale, &[out])?;
        Ok(())
    }

    fn finish(self, stem: &str, manifest: serde_json::Value) -> std::io::Result<()> {
        let mut m = manifest;
        m["arrays"] = serde_json::Value::Array(self.arrays);
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&m)?)?;
        Ok(())
    }
}

fn read_f32(st: &SafeTensors, name: &str) -> (Vec<usize>, Vec<f32>) {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("convert: missing tensor {name}"));
    let b = t.data();
    let v: Vec<f32> = match t.dtype() {
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => panic!("convert: unsupported tensor dtype {d:?} for {name}"),
    };
    (t.shape().to_vec(), v)
}

pub fn convert(model_dir: &str, arch: &str, dtype: &str, out_stem: &str) -> std::io::Result<()> {
    let cfg: HfConfig = serde_json::from_str(&std::fs::read_to_string(format!("{model_dir}/config.json"))?)?;
    let file = std::fs::File::open(format!("{model_dir}/model.safetensors"))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap).expect("convert: parse safetensors");
    match arch {
        "rope" => convert_rope(&cfg, &st, dtype, out_stem),
        other => panic!("convert: arch {other:?} not yet supported (rope)"),
    }
}

fn convert_rope(c: &HfConfig, st: &SafeTensors, dtype: &str, stem: &str) -> std::io::Result<()> {
    let nkv = c.num_key_value_heads.unwrap_or(c.num_attention_heads);
    let hd = c.head_dim.unwrap_or(c.hidden_size / c.num_attention_heads);
    let theta = c.rope_theta
        .or_else(|| c.rope_parameters.as_ref().and_then(|v| v.get("rope_theta").and_then(|t| t.as_f64())))
        .unwrap_or(10000.0);
    let eps = c.rms_norm_eps.unwrap_or(1e-6);
    let manifest = serde_json::json!({
        "format": "fieldrun-bundle", "version": 1, "arch": "rope",
        "config": [c.num_hidden_layers, c.num_attention_heads, nkv, hd, c.hidden_size, c.intermediate_size, c.vocab_size, c.tie_word_embeddings as usize],
        "config_f": [theta, eps],
    });
    let mut w = BundleWriter::new(stem)?;
    let quant = dtype == "int8";
    let mut lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (shape, data) = read_f32(st, hf); // (out, in)
        let (out, inp) = (shape[0], shape[1]);
        if quant {
            w.put_i8(name, &data, out, inp)
        } else {
            let mut t = vec![0f32; inp * out]; // transpose (out,in) -> (in,out)
            for o in 0..out {
                for i in 0..inp {
                    t[i * out + o] = data[o * inp + i];
                }
            }
            w.put_f16(name, &t, &[inp, out])
        }
    };
    let (es, ed) = read_f32(st, "model.embed_tokens.weight"); // (vocab, d) — kept f16 (also the tied unembed)
    w.put_f16("embed", &ed, &es)?;
    let (ns, nd) = read_f32(st, "model.norm.weight");
    w.put_f16("norm", &nd, &ns)?;
    if !c.tie_word_embeddings {
        let (s, d) = read_f32(st, "lm_head.weight"); // (vocab, d) -> store (d, vocab) f16 for the unembed
        let mut t = vec![0f32; s[0] * s[1]];
        for o in 0..s[0] {
            for i in 0..s[1] {
                t[i * s[0] + o] = d[o * s[1] + i];
            }
        }
        w.put_f16("lm_head", &t, &[s[1], s[0]])?;
    }
    for l in 0..c.num_hidden_layers {
        let p = format!("model.layers.{l}.");
        let (s, d) = read_f32(st, &format!("{p}input_layernorm.weight"));
        w.put_f16(&format!("l{l}.in_ln"), &d, &s)?;
        let (s, d) = read_f32(st, &format!("{p}post_attention_layernorm.weight"));
        w.put_f16(&format!("l{l}.post_ln"), &d, &s)?;
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
            if st.tensor(&format!("{p}{proj}.bias")).is_ok() {
                let (s, d) = read_f32(st, &format!("{p}{proj}.bias"));
                w.put_f16(&format!("l{l}.{proj}.bias"), &d, &s)?;
            }
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    println!("[convert] {} arrays -> {stem}.fieldrun.json/.bin (arch=rope, dtype={dtype}, no torch)", n);
    Ok(())
}
