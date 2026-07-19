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

/// int4 group size (weights along the input dim share one fp16 scale per group). 32 is the GGUF/AWQ default.
const I4_GROUP: usize = 32;

/// Optional per-tensor dtype overrides (`--dtype-map alloc.json`): tensor name -> dtype, applied in `put_lin`
/// so a single bundle can mix precisions per linear (CERTIFIED_QUANT_PROPOSAL.md). The loader already
/// dispatches per-array dtype (bundle.rs:317), so this is the only convert-side change mixed precision needs.
static DTYPE_MAP: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();

/// Install the per-tensor dtype overrides (call once in `main`, before `convert`).
pub fn set_dtype_map(m: HashMap<String, String>) { let _ = DTYPE_MAP.set(m); }

/// The dtype for `name`: the override if present, else the global `dflt`.
fn dmap_dtype<'a>(name: &str, dflt: &'a str) -> std::borrow::Cow<'a, str> {
    match DTYPE_MAP.get().and_then(|m| m.get(name)) {
        Some(d) => std::borrow::Cow::Owned(d.clone()),
        None => std::borrow::Cow::Borrowed(dflt),
    }
}

/// q4a (affine group-int4) group size. 64 (vs int4's 32) keeps the bytes equal — int4 carries one fp16 scale/group
/// (4 + 16/32 = 4.5 bpw), q4a carries an fp16 scale AND min/group (4 + (16+16)/64 = 4.5 bpw) — so a bench A/B isolates
/// the *affine* (min-offset) quality win at the SAME bundle size. See the affine quant in `put_q4a`.
const Q4A_GROUP: usize = 32;

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

    /// Shape of `name` without materializing its data (just parses the safetensors header).
    fn shape(&self, name: &str) -> Vec<usize> {
        let i = *self.idx.get(name).unwrap_or_else(|| panic!("convert: missing tensor {name}"));
        let st = SafeTensors::deserialize(&self.mmaps[i]).unwrap();
        st.tensor(name).unwrap().shape().to_vec()
    }

    /// Read rows `[row0, row0+nrows)` of `name` viewed as 2D `[shape[0], prod(shape[1..])]`, as f32, WITHOUT
    /// materializing the whole tensor — slices the mmap'd bytes directly. Lets convert quantise giant embedding /
    /// per-expert tensors block-by-block so peak RAM is one block, not prod(shape)*4 bytes (the Gemma-4 PLE table is
    /// several GB; its f32 materialisation would OOM a small box).
    fn read_rows(&self, name: &str, row0: usize, nrows: usize) -> Vec<f32> {
        let i = *self.idx.get(name).unwrap_or_else(|| panic!("convert: missing tensor {name}"));
        let st = SafeTensors::deserialize(&self.mmaps[i]).unwrap();
        let t = st.tensor(name).unwrap();
        let shape = t.shape();
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        let (esz, dt) = match t.dtype() {
            Dtype::F32 => (4usize, Dtype::F32),
            Dtype::F16 => (2, Dtype::F16),
            Dtype::BF16 => (2, Dtype::BF16),
            d => panic!("convert: unsupported dtype {d:?} for {name}"),
        };
        let b = t.data();
        let start = row0 * inner * esz;
        let end = (row0 + nrows) * inner * esz;
        let slice = &b[start..end];
        match dt {
            Dtype::F32 => slice.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::F16 => slice.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::BF16 => slice.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => unreachable!(),
        }
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

    // Like `entry` but records the int4 group size (the on-disk bytes are packed, so they aren't prod(shape)*sizeof).
    fn entry_g(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: usize, group: usize) {
        self.arrays.push(serde_json::json!({ "name": name, "dtype": dtype, "shape": shape, "offset": self.offset,
                                             "bytes": bytes, "group": group }));
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
        let dtype = dmap_dtype(name, dtype);       // per-tensor override from --dtype-map, else the global dtype
        let dtype: &str = &dtype;
        if dtype == "int8" { return self.put_i8(name, data, out, inp, true); }
        if dtype == "int4" { return self.put_i4(name, data, out, inp, true); }
        if dtype == "q4a" { return self.put_q4a(name, data, out, inp, true); }
        let mut t = vec![0f32; inp * out];
        for o in 0..out { for i in 0..inp { t[i * out + o] = data[o * inp + i]; } }
        self.put_small(name, &t, &[inp, out], dtype)
    }

    /// Group-wise symmetric **int4** from an (out, in) source (nn.Linear, `transpose=true`) or an (in, out) source
    /// (GPT-2 Conv1D, `false`). Stored OUTPUT-COLUMN-MAJOR `(out, in)` with two 4-bit values per byte along `in`, and an
    /// fp16 scale per (output-column, group-of-`in`) — half the bytes of int8, dequantised to f32 on read. Symmetric
    /// `[-7, 7]`, two's-complement nibble. For MoE experts this halves the bytes paged in per token (the offload lever).
    fn put_i4(&mut self, name: &str, data: &[f32], rows: usize, cols: usize, transpose: bool) -> std::io::Result<()> {
        let (out, inp) = if transpose { (rows, cols) } else { (cols, rows) };
        let at = |i: usize, j: usize| if transpose { data[j * inp + i] } else { data[i * out + j] }; // logical W[in=i, out=j]
        let g = I4_GROUP.min(inp.max(1));
        let ng = inp.div_ceil(g);
        let mut scale = vec![0f32; out * ng];
        for j in 0..out {
            for gi in 0..ng {
                let (lo, hi) = (gi * g, ((gi + 1) * g).min(inp));
                let amax = (lo..hi).fold(0f32, |m, i| m.max(at(i, j).abs()));
                scale[j * ng + gi] = (amax / 7.0).max(1e-8);
            }
        }
        let row_bytes = inp.div_ceil(2);
        let mut packed = vec![0u8; out * row_bytes];
        for j in 0..out {
            for i in 0..inp {
                let s = scale[j * ng + i / g];
                let q = (at(i, j) / s).round_ties_even().clamp(-7.0, 7.0) as i8;
                let nib = (q as u8) & 0x0F;
                let b = &mut packed[j * row_bytes + i / 2];
                if i % 2 == 0 { *b |= nib; } else { *b |= nib << 4; }
            }
        }
        self.bin.write_all(&packed)?;
        self.entry_g(name, "i4", &[out, inp], packed.len(), g);
        self.put_f16(&format!("{name}__scale"), &scale, &[out, ng])
    }

    /// Affine (asymmetric) group **q4a** from an (out, in) source (`transpose=true`) or (in, out) (`false`). Unsigned
    /// 4-bit `q ∈ [0,15]` with a per-group fp16 `scale` AND `min`; dequant `x = scale*q + min`. The min offset (vs
    /// symmetric int4) fits a group whose values aren't zero-centred far better — the same idea as GGUF Q4_1/Q4_K. With
    /// group 64 the bytes match int4's 4.5 bpw, so the only variable vs int4 is the affine reconstruction. Stored
    /// output-column-major `(out, in)`, 2 nibbles/byte along `in`; siblings `__scale` and `__min`, each fp16 (out, ng).
    fn put_q4a(&mut self, name: &str, data: &[f32], rows: usize, cols: usize, transpose: bool) -> std::io::Result<()> {
        let (out, inp) = if transpose { (rows, cols) } else { (cols, rows) };
        let at = |i: usize, j: usize| if transpose { data[j * inp + i] } else { data[i * out + j] }; // logical W[in=i, out=j]
        let g = Q4A_GROUP.min(inp.max(1));
        let ng = inp.div_ceil(g);
        let (mut scale, mut mins) = (vec![0f32; out * ng], vec![0f32; out * ng]);
        for j in 0..out {
            for gi in 0..ng {
                let (lo, hi) = (gi * g, ((gi + 1) * g).min(inp));
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for i in lo..hi {
                    let v = at(i, j);
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                scale[j * ng + gi] = ((mx - mn) / 15.0).max(1e-8); // 15 levels for unsigned 4-bit
                mins[j * ng + gi] = mn;
            }
        }
        let row_bytes = inp.div_ceil(2);
        let mut packed = vec![0u8; out * row_bytes];
        for j in 0..out {
            for i in 0..inp {
                let gi = j * ng + i / g;
                let q = (((at(i, j) - mins[gi]) / scale[gi]).round_ties_even()).clamp(0.0, 15.0) as u8; // unsigned [0,15]
                let b = &mut packed[j * row_bytes + i / 2];
                if i % 2 == 0 {
                    *b |= q;
                } else {
                    *b |= q << 4;
                }
            }
        }
        self.bin.write_all(&packed)?;
        self.entry_g(name, "q4a", &[out, inp], packed.len(), g);
        self.put_f16(&format!("{name}__scale"), &scale, &[out, ng])?;
        self.put_f16(&format!("{name}__min"), &mins, &[out, ng])
    }

    /// ROW-MAJOR per-row int8 for the embed/unembed `(vocab, d)`: each vocab row gets one fp16 scale (`amax/127`),
    /// stored row-major so it matches the row-contiguous access in `rows_f32`/`rowdot_f32`. Halves the largest tensor
    /// (a big-vocab lm_head) vs f16. Sibling `__scale` (vocab,). Dtype tag `rowi8`.
    fn put_embed_i8(&mut self, name: &str, data: &[f32], vocab: usize, d: usize) -> std::io::Result<()> {
        let mut scale = vec![0f32; vocab];
        let mut q = vec![0u8; vocab * d];
        for r in 0..vocab {
            let amax = (0..d).fold(0f32, |m, c| m.max(data[r * d + c].abs()));
            let s = (amax / 127.0).max(1e-8);
            scale[r] = s;
            for c in 0..d {
                q[r * d + c] = ((data[r * d + c] / s).round_ties_even().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        self.bin.write_all(&q)?;
        self.entry(name, "rowi8", &[vocab, d], q.len());
        self.put_f16(&format!("{name}__scale"), &scale, &[vocab])
    }

    /// Write the embed/unembed under the per-tensor-role policy: `embed_dtype=="int8"` → row-major quant (the largest
    /// tensor for a big-vocab model — the highest-leverage memory lever); anything else falls back to the normal small
    /// path keyed on the linear `dtype` (f16/f32). Opt-in: default keeps the embed f16 so the f32/int8 gates are intact.
    fn put_embed(&mut self, name: &str, data: &[f32], shape: &[usize], embed_dtype: &str, dtype: &str) -> std::io::Result<()> {
        match embed_dtype {
            "int8" if shape.len() == 2 => self.put_embed_i8(name, data, shape[0], shape[1]),
            _ => self.put_small(name, data, shape, dtype),
        }
    }

    /// Stream a big 2D embedding-like tensor (`rows × inner`) straight from the source mmap to the bundle in row
    /// BLOCKS, so convert never holds the whole thing as f32 (the Gemma-4 PLE table is multi-GB). Output dtype mirrors
    /// the non-streamed `put_embed`: `embed_dtype=="int8"` → per-row int8 (`rowi8` + f16 row scale, byte-identical to
    /// `put_embed_i8`); else f16 (or f32 when `dtype=="f32"`, byte-identical to `put_small`). The runtime gathers these
    /// via `rows_f32`, which already dequantises rowi8/f16/f32, so the on-disk format is unchanged — only the peak RAM is.
    fn put_embed_streamed(&mut self, m: &Model, name: &str, hf: &str, embed_dtype: &str, dtype: &str) -> std::io::Result<()> {
        let shape = m.shape(hf);
        let rows = shape[0];
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        const BLK: usize = 8192; // rows per block — caps the transient f32 buffer at BLK*inner
        let rowi8 = embed_dtype == "int8" && shape.len() == 2;
        let f32_out = !rowi8 && dtype == "f32";
        let mut scale = if rowi8 { vec![0f32; rows] } else { Vec::new() };
        let mut total = 0usize;
        let mut r = 0;
        while r < rows {
            let nb = BLK.min(rows - r);
            let blk = m.read_rows(hf, r, nb); // nb*inner f32 (one block, bounded)
            if rowi8 {
                let mut q = vec![0u8; nb * inner];
                for rr in 0..nb {
                    let amax = (0..inner).fold(0f32, |mx, c| mx.max(blk[rr * inner + c].abs()));
                    let s = (amax / 127.0).max(1e-8);
                    scale[r + rr] = s;
                    for c in 0..inner {
                        q[rr * inner + c] = ((blk[rr * inner + c] / s).round_ties_even().clamp(-127.0, 127.0) as i8) as u8;
                    }
                }
                self.bin.write_all(&q)?;
                total += q.len();
            } else if f32_out {
                let mut buf = Vec::with_capacity(nb * inner * 4);
                for &v in &blk { buf.extend_from_slice(&v.to_le_bytes()); }
                self.bin.write_all(&buf)?;
                total += buf.len();
            } else {
                let mut buf = Vec::with_capacity(nb * inner * 2);
                for &v in &blk { buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()); }
                self.bin.write_all(&buf)?;
                total += buf.len();
            }
            r += nb;
        }
        let dt = if rowi8 { "rowi8" } else if f32_out { "f32" } else { "f16" };
        self.entry(name, dt, &shape, total);
        if rowi8 {
            self.put_f16(&format!("{name}__scale"), &scale, &[rows])?;
        }
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

fn eos_ids(c: &serde_json::Value) -> Vec<i64> {
    match c.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_i64().map(|x| vec![x]).unwrap_or_default(),
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_i64()).collect(),
        _ => vec![],
    }
}

pub fn convert(model_dir: &str, arch: &str, dtype: &str, embed_dtype: &str, out_stem: &str) -> std::io::Result<()> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{model_dir}/config.json"))?)?;
    // Pre-flight: catch the common "wrong --arch for this model" mistake with a clear message instead of a deep
    // missing-config-key panic. Map the HF config's model_type to the fieldrun arch family and only error on a
    // CONFIDENT mismatch (model_type recognized AND it maps to a different arch) — novel/unknown models pass through.
    let model_type = cfg["model_type"].as_str().unwrap_or("");
    if !model_type.is_empty() {
        // HF model_type → fieldrun arch. Match by longest token that prefixes the model_type, so e.g. "gemma3_text"
        // resolves to `gemma3` (not `gemma`) and "deepseek_v3" to `mla`. Longest-prefix-wins disambiguates the families.
        let table: &[(&str, &[&str])] = &[
            ("gpt2", &["gpt2"]),
            ("neox", &["gpt_neox"]),
            ("rope", &["llama", "qwen2", "mistral", "phi"]),
            ("gemma", &["gemma", "gemma2"]),
            ("gemma3", &["gemma3"]),
            ("gemma4", &["gemma4"]),
            ("qwen3moe", &["qwen3_moe", "qwen3moe"]),
            ("mla", &["deepseek", "deepseek_v3", "kimi"]),
            ("minimax", &["minimax"]),
            ("bert", &["bert"]),
            // DeepSeek-V4 is NOT "V3 plus deltas": it replaces MLA with a new attention class (shared-KV MQA + sink +
            // undo-RoPE + grouped o_proj, mHC residual, sqrtsoftplus MoE, and CSA/HCA compressors). Longest-prefix-wins
            // routes "deepseek_v4" here (11 chars) over `mla`'s "deepseek" (8). convert_dsv4 handles the sliding subset
            // and refuses checkpoints that actually use the CSA/HCA compressors (a later stage).
            ("dsv4", &["deepseek_v4"]),
        ];
        let best = table
            .iter()
            .filter_map(|(a, toks)| {
                toks.iter().filter(|t| model_type.starts_with(**t)).map(|t| t.len()).max().map(|l| (*a, l))
            })
            .max_by_key(|&(_, l)| l)
            .map(|(a, _)| a);
        if let Some(right) = best {
            if right != arch {
                panic!("convert: this checkpoint is model_type={model_type:?}, which fieldrun maps to --arch {right}, \
                        but you passed --arch {arch}. Re-run with `--arch {right}`.");
            }
        }
    }
    let m = Model::open(model_dir);
    let shards = m.mmaps.len();
    // the bundle stem may live in a subdirectory (the default groups bundles under bundles/<name>/) — create it.
    if let Some(p) = std::path::Path::new(out_stem).parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let n = match arch {
        "gpt2" => convert_gpt2(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "neox" => convert_neox(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "rope" => convert_rope(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "gemma" => convert_gemma(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "gemma3" => convert_gemma3(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "gemma4" => convert_gemma4(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "qwen3moe" => convert_qwen3moe(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "qwen35moe" => convert_qwen35moe(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "mla" => convert_mla(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "minimax" => convert_minimax(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "dsv4" => convert_dsv4(&cfg, &m, dtype, embed_dtype, out_stem)?,
        "bert" => convert_bert(&cfg, &m, dtype, embed_dtype, out_stem)?,
        other => panic!("convert: arch {other:?} not supported (gpt2, neox, rope, gemma, gemma3, gemma4, qwen3moe, mla, minimax, bert)"),
    };
    // record the source's EOS token id(s) in the manifest (used to stop API/chat generation) — single point for all archs.
    let eos = eos_ids(&cfg);
    if !eos.is_empty() {
        let mf = format!("{out_stem}.fieldrun.json");
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mf)?)?;
        v["eos"] = serde_json::json!(eos);
        std::fs::write(&mf, serde_json::to_string(&v)?)?;
    }
    // copy the tokenizer next to the bundle so `--serve` (OpenAI/Anthropic) and `--chat` can do text I/O.
    let tok_src = format!("{model_dir}/tokenizer.json");
    if std::path::Path::new(&tok_src).exists() {
        let _ = std::fs::copy(&tok_src, format!("{out_stem}.tokenizer.json"));
    }
    println!("[convert] {n} arrays -> {out_stem}.fieldrun.json/.bin (arch={arch}, dtype={dtype}, {shards} shard(s), eos={eos:?}, no torch)");
    Ok(())
}

/// Encoder-only BERT (deepset/gbert-base class): word/position/token-type embeddings + embeddings LayerNorm,
/// per layer fused-nothing q/k/v/attn-out/fc/out Linears (all with biases) and two post-LN LayerNorms. No LM
/// head is converted (the pooler and any MLM cls head are skipped — the product is hidden states).
/// config: [n_layer, n_head, d, d_ff, vocab, max_pos, type_vocab]; config_f: [layer_norm_eps].
fn convert_bert(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let (nl, nh, d) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "num_attention_heads").unwrap(), geti(c, "hidden_size").unwrap());
    let (ffn, vocab) = (geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let (npos, ntype) = (geti(c, "max_position_embeddings").unwrap(), geti(c, "type_vocab_size").unwrap_or(2));
    let eps = getf(c, "layer_norm_eps").unwrap_or(1e-12);
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "bert",
        "config": [nl, nh, d, ffn, vocab, npos, ntype], "config_f": [eps] });
    let mut w = BundleWriter::new(stem)?;
    let pre = if m.has("bert.embeddings.word_embeddings.weight") { "bert." } else { "" }; // BertForMaskedLM vs BertModel
    let sml = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    {
        let (s, dt) = m.read(&format!("{pre}embeddings.word_embeddings.weight"));
        w.put_embed("wte", &dt, &s, edt, dtype)?;
    }
    sml(&mut w, "wpe", &format!("{pre}embeddings.position_embeddings.weight"))?;
    sml(&mut w, "wtt", &format!("{pre}embeddings.token_type_embeddings.weight"))?;
    sml(&mut w, "emb_ln.weight", &format!("{pre}embeddings.LayerNorm.weight"))?;
    sml(&mut w, "emb_ln.bias", &format!("{pre}embeddings.LayerNorm.bias"))?;
    for l in 0..nl {
        let p = format!("{pre}encoder.layer.{l}.");
        for (fr, hf) in [("q", "attention.self.query"), ("k", "attention.self.key"), ("v", "attention.self.value"),
                         ("ao", "attention.output.dense"), ("fc", "intermediate.dense"), ("out", "output.dense")] {
            let (s, dt) = m.read(&format!("{p}{hf}.weight"));
            w.put_lin(&format!("l{l}.{fr}.weight"), &dt, s[0], s[1], dtype)?;
            sml(&mut w, &format!("l{l}.{fr}.bias"), &format!("{p}{hf}.bias"))?;
        }
        sml(&mut w, &format!("l{l}.ln1.weight"), &format!("{p}attention.output.LayerNorm.weight"))?;
        sml(&mut w, &format!("l{l}.ln1.bias"), &format!("{p}attention.output.LayerNorm.bias"))?;
        sml(&mut w, &format!("l{l}.ln2.weight"), &format!("{p}output.LayerNorm.weight"))?;
        sml(&mut w, &format!("l{l}.ln2.bias"), &format!("{p}output.LayerNorm.bias"))?;
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

fn convert_gpt2(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
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
    // wte (tied embed/unembed) honours the per-role embed policy; wpe/ln_f stay small (f16 or f32)
    {
        let (s, dt) = m.read(&format!("{pre}wte.weight"));
        w.put_embed("wte", &dt, &s, edt, dtype)?;
    }
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
            if i8 { w.put_i8(&format!("h{l}.{fr}.weight"), &dt, s[0], s[1], false)?; }
            else if dtype == "int4" { w.put_i4(&format!("h{l}.{fr}.weight"), &dt, s[0], s[1], false)?; }
            else { w.put_small(&format!("h{l}.{fr}.weight"), &dt, &s, dtype)?; }
            sml(&mut w, &format!("h{l}.{fr}.bias"), &format!("{p}{hf}.bias"))?;
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// GPT-NeoX / Pythia: LayerNorm(+bias) backbone, **parallel residual**, **partial rotary** (`rotary_pct` of each
/// head), GELU MLP, untied `embed_out`. The fused `attention.query_key_value` packs [q,k,v] PER HEAD (rows
/// `h·3·hd .. (h+1)·3·hd` are that head's q,k,v stacked), so it's de-interleaved into plain q/k/v linears here —
/// the kernel never sees the fusion. config: [nl, nh, hd, d, ffn, vocab, rot_ndims, parallel]; config_f: [theta, eps].
fn convert_neox(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let nh = geti(c, "num_attention_heads").unwrap();
    let d = geti(c, "hidden_size").unwrap();
    let hd = d / nh;
    let (nl, ffn, vocab) = (geti(c, "num_hidden_layers").unwrap(), geti(c, "intermediate_size").unwrap(), geti(c, "vocab_size").unwrap());
    let rot = (((hd as f64 * getf(c, "rotary_pct").unwrap_or(1.0)).round() as usize).max(2)) & !1; // even, ≥2
    let theta = getf(c, "rotary_emb_base").unwrap_or(10000.0);
    let eps = getf(c, "layer_norm_eps").unwrap_or(1e-5);
    let par = c.get("use_parallel_residual").and_then(|v| v.as_bool()).unwrap_or(true);
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "neox",
        "config": [nl, nh, hd, d, ffn, vocab, rot, par as usize], "config_f": [theta, eps] });
    let mut w = BundleWriter::new(stem)?;
    let sml = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    {
        let (s, dt) = m.read("gpt_neox.embed_in.weight");
        w.put_embed("embed", &dt, &s, edt, dtype)?;
    }
    {
        // untied unembed: read row-wise by rowdot_f32 as (vocab, d) → stored raw, not transposed
        let (s, dt) = m.read("embed_out.weight");
        w.put_embed("lm_head", &dt, &s, edt, dtype)?;
    }
    sml(&mut w, "ln_f.weight", "gpt_neox.final_layer_norm.weight")?;
    sml(&mut w, "ln_f.bias", "gpt_neox.final_layer_norm.bias")?;
    for l in 0..nl {
        let p = format!("gpt_neox.layers.{l}.");
        sml(&mut w, &format!("l{l}.ln1.weight"), &format!("{p}input_layernorm.weight"))?;
        sml(&mut w, &format!("l{l}.ln1.bias"), &format!("{p}input_layernorm.bias"))?;
        sml(&mut w, &format!("l{l}.ln2.weight"), &format!("{p}post_attention_layernorm.weight"))?;
        sml(&mut w, &format!("l{l}.ln2.bias"), &format!("{p}post_attention_layernorm.bias"))?;
        // de-interleave the fused qkv: per head h, source rows (h·3 + which)·hd .. +hd are q/k/v for which = 0/1/2
        let (qs, qd) = m.read(&format!("{p}attention.query_key_value.weight")); // (3d, d)
        assert_eq!(qs, vec![3 * d, d], "neox qkv weight shape");
        let (_, qb) = m.read(&format!("{p}attention.query_key_value.bias")); // (3d,)
        for (which, nm) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            let mut sub = vec![0f32; d * d];
            let mut sb = vec![0f32; d];
            for h in 0..nh {
                let src = (h * 3 + which) * hd;
                let dst = h * hd;
                sub[dst * d..(dst + hd) * d].copy_from_slice(&qd[src * d..(src + hd) * d]);
                sb[dst..dst + hd].copy_from_slice(&qb[src..src + hd]);
            }
            w.put_lin(&format!("l{l}.{nm}"), &sub, d, d, dtype)?;
            w.put_small(&format!("l{l}.{nm}.bias"), &sb, &[d], dtype)?;
        }
        for (fr, hf) in [("dense", "attention.dense"), ("fc_in", "mlp.dense_h_to_4h"), ("fc_out", "mlp.dense_4h_to_h")] {
            let (s, dt) = m.read(&format!("{p}{hf}.weight"));
            w.put_lin(&format!("l{l}.{fr}"), &dt, s[0], s[1], dtype)?;
            sml(&mut w, &format!("l{l}.{fr}.bias"), &format!("{p}{hf}.bias"))?;
        }
    }
    let _ = ffn;
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
fn convert_rope(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
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
    write_layers(&mut w, c, m, dtype, edt, nl, tie, &[("in_ln", "input_layernorm"), ("post_ln", "post_attention_layernorm")], false)?;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

fn convert_gemma(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
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
    write_layers(&mut w, c, m, dtype, edt, nl, tie, &norms, true)?;
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// Gemma 3: the Gemma-2 stack plus QK-norm (per-head RMSNorm on q/k), dual-base RoPE (local θ for sliding layers,
/// global θ for full layers), a 5:1 sliding:full layer pattern, and NO logit soft-capping. head_dim is shared across
/// layer types (unlike Gemma 4). Per-layer sliding flags (from `layer_types`) are packed into `config` so the kernel
/// needn't re-derive the pattern. `config_f` carries both RoPE bases.
fn convert_gemma3(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
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
    write_layers(&mut w, c, m, dtype, edt, nl, tie, &norms, true)?;
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
/// block. attention_k_eq_v (global layers drop v_proj) and KV-sharing (the last num_kv_shared_layers drop k/v/k_norm) are
/// both supported.
fn convert_gemma4(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    // The released checkpoints ship the multimodal composite `Gemma4Config` (model_type "gemma4") with all the text
    // params under `text_config`; a standalone `Gemma4ForCausalLM` saves the flat `Gemma4TextConfig` (no nesting). Read
    // text params from `text_config` when present, else the top level. `tie_word_embeddings` lives at the TOP level on
    // the composite, so it falls back to `c`.
    let tc = c.get("text_config").unwrap_or(c);
    // Tensor-name prefix: the multimodal composite checkpoints nest the text model under `model.language_model.*`
    // (alongside `vision_tower`/`audio_tower`, which we ignore); a standalone `Gemma4ForCausalLM` uses `model.*`.
    let lmp = if m.has("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" };
    // attention_k_eq_v: GLOBAL layers carry no v_proj — V is the k_proj output (value-normed) — and use
    // num_global_key_value_heads KV heads. We record the flag and skip the absent v_proj weights below.
    let k_eq_v = tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false);
    // KV-sharing: the last `n_kv_shared` layers carry no k_proj/v_proj/k_norm — they reuse an earlier same-type layer's
    // assembled K/V. We record the count and skip those absent weights below (the kernel resolves the source layer).
    let n_kv_shared = geti(tc, "num_kv_shared_layers").unwrap_or(0);
    let moe = tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false);
    let n_exp = geti(tc, "num_experts").unwrap_or(0);
    let topk = geti(tc, "top_k_experts").unwrap_or(0);
    let moe_inter = geti(tc, "moe_intermediate_size").unwrap_or(0);
    let nh = geti(tc, "num_attention_heads").unwrap();
    let nkv = geti(tc, "num_key_value_heads").unwrap_or(nh);
    let nkv_g = geti(tc, "num_global_key_value_heads").unwrap_or(nkv);
    let d = geti(tc, "hidden_size").unwrap();
    let hd = geti(tc, "head_dim").unwrap_or(d / nh);
    let hd_g = geti(tc, "global_head_dim").unwrap_or(hd);
    let (nl, ffn, vocab) = (geti(tc, "num_hidden_layers").unwrap(), geti(tc, "intermediate_size").unwrap(), geti(tc, "vocab_size").unwrap());
    let ple = geti(tc, "hidden_size_per_layer_input").unwrap_or(256);
    let eps = getf(tc, "rms_norm_eps").unwrap_or(1e-6);
    let window = geti(tc, "sliding_window").unwrap_or(512);
    let pattern = geti(tc, "sliding_window_pattern").unwrap_or(6);
    let (theta_local, theta_global) = gemma3_thetas(tc);
    let prf = tc.get("rope_parameters").and_then(|v| v.get("full_attention")).and_then(|t| t.get("partial_rotary_factor"))
        .and_then(|t| t.as_f64()).unwrap_or(0.25);
    let tie = tc.get("tie_word_embeddings").or_else(|| c.get("tie_word_embeddings")).and_then(|v| v.as_bool()).unwrap_or(true);
    let lt = tc.get("layer_types").and_then(|v| v.as_array());
    let full_of = |l: usize| lt.and_then(|a| a.get(l)).and_then(|s| s.as_str())
        .map(|s| s == "full_attention").unwrap_or((l + 1) % pattern == 0);
    // Gemma 4 forces the last layer to full_attention.
    let is_full = |l: usize| full_of(l) || l + 1 == nl;
    // Per-layer output scalar (`layer_scalar`, a persistent buffer applied as the LAST op of each decoder layer in
    // `Gemma4ForCausalLM`; default 1.0). Read it per layer so a checkpoint that ships a non-1.0 value runs faithfully
    // instead of silently diverging; appended to config_f after the 4 rope/eps scalars.
    let layer_scalar: Vec<f64> = (0..nl).map(|l| {
        let k = format!("{lmp}.layers.{l}.layer_scalar");
        if m.has(&k) { m.read(&k).1.first().copied().unwrap_or(1.0) as f64 } else { 1.0 }
    }).collect();
    let mut config: Vec<usize> = vec![nl, nh, nkv, nkv_g, hd, hd_g, d, ffn, vocab, tie as usize, window, ple,
                                      moe as usize, n_exp, topk, moe_inter];
    for l in 0..nl { config.push(if is_full(l) { 0 } else { 1 }); } // sliding flags start at config[16]
    config.push(k_eq_v as usize); // attention_k_eq_v flag at config[16+nl]
    config.push(n_kv_shared);     // num_kv_shared_layers at config[17+nl]
    let first_shared = nl - n_kv_shared;
    let is_shared = |l: usize| n_kv_shared > 0 && l >= first_shared; // shared layers drop k_proj/v_proj/k_norm
    let mut config_f: Vec<f64> = vec![theta_local, theta_global, eps, prf];
    config_f.extend(&layer_scalar); // config_f[4..4+nl] = per-layer layer_scalar
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "gemma4",
        "config": config, "config_f": config_f });

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
    // main + PLE embeddings — STREAMED (block-by-block) so the multi-GB Gemma-4 tables never materialise as f32 in
    // convert (the f32 peak of `embed_tokens_per_layer` alone is ~prod(shape)*4 = many GB and would OOM a small box).
    // Both follow `embed_dtype`: default f16 (byte-identical to the old put_embed/put_small), `--embed-dtype int8`
    // halves them to rowi8. The runtime gathers both via `rows_f32`, which dequantises rowi8/f16/f32 transparently.
    w.put_embed_streamed(&m, "embed", &format!("{lmp}.embed_tokens.weight"), edt, dtype)?;
    norm(&mut w, "norm", &format!("{lmp}.norm.weight"))?;
    // Per-Layer-Embedding tables — ONLY when PLE is enabled (`hidden_size_per_layer_input>0`). The 26B-A4B MoE ships
    // ple=0 (no PLE), so these tensors are absent in the checkpoint; reading them unconditionally would panic.
    if ple > 0 {
        w.put_embed_streamed(&m, "embed_per_layer", &format!("{lmp}.embed_tokens_per_layer.weight"), edt, dtype)?; // (vocab_per_layer, nl*ple)
        norm(&mut w, "per_layer_projection_norm", &format!("{lmp}.per_layer_projection_norm.weight"))?;
        // per_layer_model_projection: Linear(d -> nl*ple); the int8 W8A8 path needs the weight, so keep it f16/f32 like a norm
        let (s, dt) = m.read(&format!("{lmp}.per_layer_model_projection.weight")); // (nl*ple, d)
        let (out, inp) = (s[0], s[1]);
        let mut t = vec![0f32; inp * out];
        for o in 0..out { for i in 0..inp { t[i * out + o] = dt[o * inp + i]; } }
        w.put_small("per_layer_model_projection", &t, &[inp, out], dtype)?;
    }
    if !tie {
        w.put_embed_streamed(&m, "lm_head", "lm_head.weight", edt, dtype)?; // (vocab, d) — streamed like embed
    }
    for l in 0..nl {
        let p = format!("{lmp}.layers.{l}.");
        for nm in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
                   "self_attn.q_norm", "self_attn.k_norm", "post_per_layer_input_norm"] {
            // KV-shared layers reuse an earlier layer's K, so they have no k_norm.
            if nm == "self_attn.k_norm" && is_shared(l) {
                continue;
            }
            // PLE-off models (ple=0, e.g. 26B-A4B) have no post_per_layer_input_norm.
            if nm == "post_per_layer_input_norm" && ple == 0 {
                continue;
            }
            norm(&mut w, &format!("l{l}.{nm}"), &format!("{p}{nm}.weight"))?;
        }
        // v_norm has with_scale=False (no weight) → nothing to write.
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "per_layer_input_gate", "per_layer_projection"] {
            // attention_k_eq_v global layers have no v_proj (the kernel reuses k_proj as V) → skip the absent weight.
            if proj == "self_attn.v_proj" && k_eq_v && is_full(l) {
                continue;
            }
            // PLE-off models (ple=0) have no per-layer-input gate/projection.
            if (proj == "per_layer_input_gate" || proj == "per_layer_projection") && ple == 0 {
                continue;
            }
            // KV-shared layers have no k_proj/v_proj (they reuse an earlier same-type layer's assembled K/V).
            if (proj == "self_attn.k_proj" || proj == "self_attn.v_proj") && is_shared(l) {
                continue;
            }
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
            // int8/int4 array so a single expert can be paged in independently (the mmap-offload contract). Read each
            // expert's slice STREAMED (one expert at a time) rather than materialising all E experts as f32 — for a
            // many-expert MoE the whole 3D tensor is several GB of f32 and would OOM a small box.
            let gus = m.shape(&format!("{p}experts.gate_up_proj")); // (E, 2*mi, d)
            let dns = m.shape(&format!("{p}experts.down_proj"));    // (E, d, mi)
            let (gu_out, gu_in) = (gus[1], gus[2]); // (2*mi, d)
            let (dn_out, dn_in) = (dns[1], dns[2]); // (d, mi)
            for e in 0..n_exp {
                let gu = m.read_rows(&format!("{p}experts.gate_up_proj"), e, 1); // expert e, flat (2*mi*d)
                w.put_lin(&format!("l{l}.experts.{e}.gate_up"), &gu, gu_out, gu_in, dtype)?;
                let dn = m.read_rows(&format!("{p}experts.down_proj"), e, 1); // expert e, flat (d*mi)
                w.put_lin(&format!("l{l}.experts.{e}.down"), &dn, dn_out, dn_in, dtype)?;
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
/// (Qwen3 dropped it), no embed scale, no soft-capping. Sliding window (`use_sliding_window`) applies ONE window to
/// every layer (no per-layer pattern; the window is appended to `config` after the MoE flags, 0 = full attention).
fn convert_qwen3moe(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let swa = c.get("use_sliding_window").and_then(|v| v.as_bool()).unwrap_or(false);
    let window = if swa { geti(c, "sliding_window").unwrap_or(4096) } else { 0 };
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
    config.push(window); // sliding window, all layers (0 = full attention)
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
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); // (vocab, d) — raw for rowdot_f32, low-precision
        w.put_embed("lm_head", &dt, &s, edt, dtype)?;
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

/// Qwen3.6 (`qwen3_5_moe`) — hybrid: 3-of-4 layers are Gated DeltaNet *linear* attention, the rest full GQA
/// attention; every layer has a SparseMoeBlock (softmax top-k routed experts + a sigmoid-gated shared expert).
/// The text config is NESTED under `text_config` (the family is VL/omni-capable). Experts ship as PACKED 3D
/// tensors (`experts.gate_up_proj` [E, 2·moe_inter, d], `experts.down_proj` [E, d, moe_inter]); we unpack them
/// into the per-expert `l{l}.experts.{e}.{gate,up,down}` layout so the runtime reuses the qwen3moe MoE path.
/// config_i: [nl, nh, nkv, hd, d, vocab, tied, n_exp, topk, moe_inter, shared_inter, norm_topk,
///            num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_k, <nl layer_types: 0=full 1=linear>]
fn convert_qwen35moe(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let tc = c.get("text_config").unwrap_or(c); // text dims are nested in the composite config
    let nh = geti(tc, "num_attention_heads").unwrap();
    let nkv = geti(tc, "num_key_value_heads").unwrap_or(nh);
    let d = geti(tc, "hidden_size").unwrap();
    let hd = geti(tc, "head_dim").unwrap_or(d / nh);
    let (nl, vocab) = (geti(tc, "num_hidden_layers").unwrap(), geti(tc, "vocab_size").unwrap());
    let (theta, eps) = rope_theta_eps(tc);
    let tie = tc.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let n_exp = geti(tc, "num_experts").unwrap_or(0);
    let topk = geti(tc, "num_experts_per_tok").unwrap_or(0);
    let moe_inter = geti(tc, "moe_intermediate_size").unwrap_or(0);
    let shared_inter = geti(tc, "shared_expert_intermediate_size").unwrap_or(0);
    let norm_topk = tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(false);
    let (nvh, nkh) = (geti(tc, "linear_num_value_heads").unwrap(), geti(tc, "linear_num_key_heads").unwrap());
    let (hkd, hvd) = (geti(tc, "linear_key_head_dim").unwrap(), geti(tc, "linear_value_head_dim").unwrap());
    let conv_k = geti(tc, "linear_conv_kernel_dim").unwrap_or(4);
    // partial RoPE: only the first `rotary_dim` of each head_dim is rotated (full attention only)
    let prf = tc.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let rotary_dim = (hd as f64 * prf) as usize;
    let ltypes: Vec<String> = tc.get("layer_types").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    let is_linear = |l: usize| ltypes.get(l).map(|s| s == "linear_attention").unwrap_or(false);

    let mut config: Vec<usize> = vec![nl, nh, nkv, hd, d, vocab, tie as usize, n_exp, topk, moe_inter,
                                      shared_inter, norm_topk as usize, nvh, nkh, hkd, hvd, conv_k];
    for l in 0..nl { config.push(if is_linear(l) { 1 } else { 0 }); } // layer_types at config[17..17+nl]
    config.push(rotary_dim); // config[17+nl] = rotary_dim (partial RoPE width)
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "qwen35moe",
        "config": config, "config_f": [theta, eps] });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight");
        w.put_embed("lm_head", &dt, &s, edt, dtype)?;
    }
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        norm(&mut w, &format!("l{l}.in_ln"), &format!("{p}input_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.post_ln"), &format!("{p}post_attention_layernorm.weight"))?;
        if is_linear(l) {
            let q = format!("{p}linear_attn.");
            for proj in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] {
                lin(&mut w, &format!("l{l}.linear_attn.{proj}"), &format!("{q}{proj}.weight"))?;
            }
            let (cs, cd) = m.read(&format!("{q}conv1d.weight")); // [conv_dim, 1, k] -> store flat [conv_dim, k]
            w.put_small(&format!("l{l}.linear_attn.conv1d"), &cd, &[cs[0], cs[2]], dtype)?;
            // dt_bias / A_log are bare params; norm carries a `.weight` suffix
            for (nm, hf, sz) in [("dt_bias", "dt_bias", nvh), ("A_log", "A_log", nvh), ("norm", "norm.weight", hvd)] {
                let (_, td) = m.read(&format!("{q}{hf}"));
                w.put_small(&format!("l{l}.linear_attn.{nm}"), &td, &[sz], dtype)?;
            }
        } else {
            for nm in ["q_norm", "k_norm"] {
                norm(&mut w, &format!("l{l}.{nm}"), &format!("{p}self_attn.{nm}.weight"))?;
            }
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
                lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
            }
        }
        // MoE (every layer): router + experts (handle BOTH on-disk layouts) + sigmoid-gated shared expert
        lin(&mut w, &format!("l{l}.gate"), &format!("{p}mlp.gate.weight"))?;
        if m.has(&format!("{p}mlp.experts.gate_up_proj")) {
            // packed 3D: gate_up_proj [E, 2*moe_inter, d], down_proj [E, d, moe_inter] — slice per expert
            let (gs, gd) = m.read(&format!("{p}mlp.experts.gate_up_proj"));
            let (ds, dd) = m.read(&format!("{p}mlp.experts.down_proj"));
            assert_eq!(gs, [n_exp, 2 * moe_inter, d], "l{l} gate_up_proj shape (E, 2*moe_inter, d)");
            assert_eq!(ds, [n_exp, d, moe_inter], "l{l} down_proj shape (E, d, moe_inter)");
            let (gu_stride, dn_stride) = (gs[1] * gs[2], ds[1] * ds[2]);
            for e in 0..n_exp {
                let gu = &gd[e * gu_stride..(e + 1) * gu_stride]; // [2*moe_inter, d] row-major
                w.put_lin(&format!("l{l}.experts.{e}.gate"), &gu[..moe_inter * d], moe_inter, d, dtype)?;
                w.put_lin(&format!("l{l}.experts.{e}.up"), &gu[moe_inter * d..], moe_inter, d, dtype)?;
                let dn = &dd[e * dn_stride..(e + 1) * dn_stride]; // [d, moe_inter]
                w.put_lin(&format!("l{l}.experts.{e}.down"), dn, d, moe_inter, dtype)?;
            }
        } else {
            // per-expert 2D (the canonical HF on-disk format; what save_pretrained emits)
            for e in 0..n_exp {
                for (fr, hf) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] {
                    lin(&mut w, &format!("l{l}.experts.{e}.{fr}"), &format!("{p}mlp.experts.{e}.{hf}.weight"))?;
                }
            }
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            lin(&mut w, &format!("l{l}.shared.{proj}"), &format!("{p}mlp.shared_expert.{proj}.weight"))?;
        }
        lin(&mut w, &format!("l{l}.shared_gate"), &format!("{p}mlp.shared_expert_gate.weight"))?;
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// DeepSeek-V3 / Kimi-K2 — MLA (multi-head latent attention) + DeepSeek MoE. MLA compresses q and kv through low-rank
/// down→up projections (q_a/q_b, kv_a/kv_b) with a 128-dim no-RoPE part and a 64-dim shared decoupled-RoPE part, and a
/// distinct v_head_dim. The MoE has a shared always-on expert plus group-limited sigmoid routing (with a learned bias
/// correction). The first `first_k_dense_replace` layers are dense. Experts written one int8 array each (offload).
/// YaRN rope scaling (`rope_parameters`/`rope_scaling` with type "yarn" — every real DeepSeek-V3/R1/Kimi-K2 config)
/// is passed through in `config_f`. Interleaved rotary weights (`rope_interleave`, the DeepSeek default) are
/// de-interleaved here — the permutation is baked into the q_b/q and kv_a rotary rows so the runtime's split-half
/// rope matches the torch interleave path exactly.
fn convert_mla(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
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
    // YaRN rope scaling → config_f[3..]: [yarn, factor, beta_fast, beta_slow, mscale, mscale_all_dim,
    // original_max_pos, truncate, attention_factor(0=derive)]. Accept the new `rope_parameters` nesting or the
    // legacy flat `rope_scaling` (real hub checkpoints); refuse rope types we don't implement rather than ignore them.
    let mut config_f: Vec<f64> = vec![theta, eps, routed_scaling];
    let rp = c.get("rope_parameters").or_else(|| c.get("rope_scaling")).filter(|v| !v.is_null());
    let rope_type = rp.and_then(|v| v.get("rope_type").or_else(|| v.get("type"))).and_then(|t| t.as_str()).unwrap_or("default");
    match rope_type {
        "default" => {}
        "yarn" => {
            let rp = rp.unwrap();
            let g = |k: &str| rp.get(k).and_then(|v| v.as_f64());
            let orig = g("original_max_position_embeddings").unwrap_or(4096.0);
            // factor defaults to the post/pre-scaling context ratio (as upstream) when absent
            let factor = g("factor").unwrap_or_else(|| geti(c, "max_position_embeddings").map_or(1.0, |mp| mp as f64 / orig));
            config_f.extend([1.0, factor, g("beta_fast").unwrap_or(32.0), g("beta_slow").unwrap_or(1.0),
                             g("mscale").unwrap_or(0.0), g("mscale_all_dim").unwrap_or(0.0), orig,
                             if rp.get("truncate").and_then(|v| v.as_bool()).unwrap_or(true) { 1.0 } else { 0.0 },
                             g("attention_factor").unwrap_or(0.0)]);
        }
        other => panic!("convert: mla rope_type {other:?} not supported (default, yarn)"),
    }
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "mla",
        "config": config, "config_f": config_f });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_small(name, &dt, &s, dtype) // standard RMSNorm, weight used directly
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    // rope_interleave (DeepSeek's default, true): torch de-interleaves the rotary slice ([x0,x1,..] → evens‖odds)
    // before a split-half rotation. Baking that permutation into the projection ROWS that produce each rotary slice
    // makes the runtime's plain split-half rope bit-equivalent. `starts` lists each rotary block's first output row.
    let interleave = c.get("rope_interleave").and_then(|v| v.as_bool()).unwrap_or(true);
    let lin_rot = |w: &mut BundleWriter, name: &str, hf: &str, starts: &[usize]| -> std::io::Result<()> {
        let (s, mut dt) = m.read(hf);
        if interleave {
            let (inp, half) = (s[1], qk_rope / 2);
            let mut tmp = vec![0f32; qk_rope * inp];
            for &start in starts {
                for j in 0..half {
                    tmp[j * inp..(j + 1) * inp].copy_from_slice(&dt[(start + 2 * j) * inp..(start + 2 * j + 1) * inp]);
                    tmp[(half + j) * inp..(half + j + 1) * inp]
                        .copy_from_slice(&dt[(start + 2 * j + 1) * inp..(start + 2 * j + 2) * inp]);
                }
                dt[start * inp..(start + qk_rope) * inp].copy_from_slice(&tmp);
            }
        }
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let qkh = qk_nope + qk_rope;
    let q_rot_starts: Vec<usize> = (0..nh).map(|h| h * qkh + qk_nope).collect();
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); w.put_embed("lm_head", &dt, &s, edt, dtype)?;
    }
    // experts ship either packed (experts.gate_up_proj/down_proj 3D) or per-expert Linears — write per-expert gate/up/down either way.
    let write_experts = |w: &mut BundleWriter, p: &str, l: usize| -> std::io::Result<()> {
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
            lin_rot(&mut w, &format!("l{l}.q_b"), &format!("{p}self_attn.q_b_proj.weight"), &q_rot_starts)?;
        } else {
            lin_rot(&mut w, &format!("l{l}.q"), &format!("{p}self_attn.q_proj.weight"), &q_rot_starts)?;
        }
        lin_rot(&mut w, &format!("l{l}.kv_a"), &format!("{p}self_attn.kv_a_proj_with_mqa.weight"), &[kv_lora])?;
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

/// MiniMax-M2 — the RoPE backbone + FULL-WIDTH q/k-norm (RMSNorm over the whole nh·hd / nkv·hd projection, not
/// per-head) + an all-MoE FFN with a sigmoid router (sigmoid scores + bias for the choice, sigmoid scores renormed for
/// the weight; no group limiting, no shared expert). Experts written one int8 array each (offload).
fn convert_minimax(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let nh = geti(c, "num_attention_heads").unwrap();
    let nkv = geti(c, "num_key_value_heads").unwrap_or(nh);
    let d = geti(c, "hidden_size").unwrap();
    let hd = geti(c, "head_dim").unwrap_or(d / nh);
    let nl = geti(c, "num_hidden_layers").unwrap();
    let vocab = geti(c, "vocab_size").unwrap();
    let inter = geti(c, "intermediate_size").unwrap();
    let n_exp = geti(c, "num_local_experts").or_else(|| geti(c, "num_experts")).unwrap();
    let topk = geti(c, "num_experts_per_tok").unwrap();
    let (theta, eps) = rope_theta_eps(c);
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let config: Vec<usize> = vec![nl, nh, nkv, hd, d, vocab, tie as usize, n_exp, topk, inter];
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "minimax",
        "config": config, "config_f": [theta, eps] });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    norm(&mut w, "norm", "model.norm.weight")?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight"); w.put_embed("lm_head", &dt, &s, edt, dtype)?;
    }
    // MiniMax-M2 uses Mixtral-style MoE naming: block_sparse_moe.{gate, e_score_correction_bias,
    // experts.{e}.w1/w2/w3} where w1=gate_proj, w2=down_proj, w3=up_proj.
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        let bsm = format!("{p}block_sparse_moe.");
        norm(&mut w, &format!("l{l}.in_ln"), &format!("{p}input_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.post_ln"), &format!("{p}post_attention_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.q_norm"), &format!("{p}self_attn.q_norm.weight"))?; // full nh*hd width
        norm(&mut w, &format!("l{l}.k_norm"), &format!("{p}self_attn.k_norm.weight"))?; // full nkv*hd width
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
            lin(&mut w, &format!("l{l}.{proj}"), &format!("{p}{proj}.weight"))?;
        }
        lin(&mut w, &format!("l{l}.gate"), &format!("{bsm}gate.weight"))?;
        let (bs, bd) = m.read(&format!("{bsm}e_score_correction_bias"));
        w.put_small(&format!("l{l}.gate_bias"), &bd, &bs, dtype)?;
        for e in 0..n_exp {
            for (fr, hf) in [("gate", "w1"), ("up", "w3"), ("down", "w2")] {
                let (s, dt) = m.read(&format!("{bsm}experts.{e}.{hf}.weight"));
                w.put_lin(&format!("l{l}.experts.{e}.{fr}"), &dt, s[0], s[1], dtype)?;
            }
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

/// Shared Llama/Qwen/Gemma writer: embed (f16) + final norm + per-layer norms (with optional +1 bake) + the
/// q/k/v/o/gate/up/down Linears (transposed, int8 or f16) + optional q/k/v bias.
fn write_layers(w: &mut BundleWriter, c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, nl: usize, tie: bool,
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
    // embed (and the tied unembed) honour the per-role `edt` policy — quantising the largest tensor (opt-in).
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    norm(w, "norm", "model.norm.weight")?;
    if !tie {
        // unembed is read row-wise by rowdot_f32 as (vocab, d) → store raw (NOT transposed), low-precision like embed
        let (s, dt) = m.read("lm_head.weight"); // (vocab, d)
        w.put_embed("lm_head", &dt, &s, edt, dtype)?;
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
        // Qwen3-dense QK-norm: per-head RMSNorm on q/k, present only on Qwen3 (not Llama/Qwen2.5). Standard RMSNorm
        // (no (1+w) bake), so it rides the same `norm` path; skip if this arch already wrote it via `norms` (gemma3).
        for nm in ["self_attn.q_norm", "self_attn.k_norm"] {
            if m.has(&format!("{p}{nm}.weight")) && !norms.iter().any(|(_, h)| *h == nm) {
                norm(w, &format!("l{l}.{nm}"), &format!("{p}{nm}.weight"))?;
            }
        }
    }
    Ok(())
}

/// DeepSeek-V4 (Stage 1: the sliding-only backbone). Refuses checkpoints that actually use the CSA/HCA compressors or
/// hash-routed MoE layers (later stages). Weights: q-LoRA (q_a/q_b + q_a_norm), shared-KV MQA (kv_proj/kv_norm), the
/// grouped o_proj written as one block per group, per-head attention sinks, the two mHC HyperConnections + the final
/// HyperHead (fn/base/scale params), the sqrtsoftplus router (+ e_score_correction_bias) over packed experts, and the
/// always-on shared expert. config packs the dims; config_f the rope-theta/eps/clamp/scale scalars.
fn convert_dsv4(c: &serde_json::Value, m: &Model, dtype: &str, edt: &str, stem: &str) -> std::io::Result<usize> {
    let nl = geti(c, "num_hidden_layers").unwrap();
    let nh = geti(c, "num_attention_heads").unwrap();
    let hd = geti(c, "head_dim").unwrap();
    let d = geti(c, "hidden_size").unwrap();
    let q_lora = geti(c, "q_lora_rank").unwrap();
    // partial_rotary_factor (a fraction) → rope_head_dim = round(head_dim * factor); falls back to default_partial.
    let prf = getf(c, "partial_rotary_factor").or_else(|| getf(c, "default_partial_rotary_factor")).unwrap_or(64.0 / 512.0);
    let rd = (hd as f64 * prf).round() as usize;
    let n_exp = geti(c, "n_routed_experts").or_else(|| geti(c, "num_local_experts")).unwrap();
    let top_k = geti(c, "num_experts_per_tok").unwrap();
    let o_groups = geti(c, "o_groups").unwrap();
    let o_lora = geti(c, "o_lora_rank").unwrap();
    let moe_inter = geti(c, "moe_intermediate_size").or_else(|| geti(c, "intermediate_size")).unwrap();
    let vocab = geti(c, "vocab_size").unwrap();
    let hc = geti(c, "hc_mult").unwrap_or(4);
    let sinkhorn = geti(c, "hc_sinkhorn_iters").unwrap_or(20);
    let window = geti(c, "sliding_window").unwrap();
    // V4 declares _tied_weights_keys(lm_head→embed): a checkpoint that omits lm_head.weight is tied (reload re-ties it),
    // so treat a missing lm_head as tied regardless of the config flag.
    let tie = c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false) || !m.has("lm_head.weight");
    let theta = getf(c, "rope_theta").unwrap_or(10000.0);
    let eps = getf(c, "rms_norm_eps").unwrap_or(1e-6);
    let limit = getf(c, "swiglu_limit").unwrap_or(10.0);
    let rscale = getf(c, "routed_scaling_factor").unwrap_or(1.5);
    let hc_eps = getf(c, "hc_eps").unwrap_or(1e-6);
    // Stage 1 covers the sliding-only / all-`moe` subset. Refuse the compressor + hash-routed regimes loudly.
    let lt = c.get("layer_types").and_then(|v| v.as_array());
    if let Some(a) = lt {
        for (l, v) in a.iter().enumerate() {
            let t = v.as_str().unwrap_or("");
            assert_eq!(t, "sliding_attention",
                "dsv4: layer {l} is {t:?} (CSA/HCA compressor) — not yet supported (Stage 1 is sliding-only).");
        }
    }
    if let Some(a) = c.get("mlp_layer_types").and_then(|v| v.as_array()) {
        for (l, v) in a.iter().enumerate() {
            assert_eq!(v.as_str().unwrap_or("moe"), "moe",
                "dsv4: layer {l} uses hash routing — not yet supported (Stage 1 is moe-only).");
        }
    }

    let config: Vec<usize> = vec![nl, nh, hd, q_lora, rd, d, n_exp, top_k, o_groups, o_lora, moe_inter, vocab, hc, sinkhorn, window, tie as usize];
    let config_f: Vec<f64> = vec![theta, eps, limit, rscale, hc_eps];
    let manifest = serde_json::json!({ "format": "fieldrun-bundle", "version": 1, "arch": "dsv4",
        "config": config, "config_f": config_f });

    let mut w = BundleWriter::new(stem)?;
    let norm = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf);
        w.put_small(name, &dt, &s, dtype)
    };
    let lin = |w: &mut BundleWriter, name: &str, hf: &str| -> std::io::Result<()> {
        let (s, dt) = m.read(hf); // (out, in)
        w.put_lin(name, &dt, s[0], s[1], dtype)
    };
    // embeddings stay low-precision; lm_head raw (vocab, d) for rowdot_f32 when untied.
    let (es, ed) = m.read("model.embed_tokens.weight");
    w.put_embed("embed", &ed, &es, edt, dtype)?;
    if !tie {
        let (s, dt) = m.read("lm_head.weight");
        w.put_embed("lm_head", &dt, &s, edt, dtype)?;
    }
    norm(&mut w, "norm", "model.norm.weight")?;
    // HyperHead (final stream collapse)
    lin(&mut w, "hc_head.hc_fn", "model.hc_head.hc_fn")?;
    {
        let (s, dt) = m.read("model.hc_head.hc_base");
        w.put_small("hc_head.hc_base", &dt, &s, dtype)?;
        let (s2, dt2) = m.read("model.hc_head.hc_scale");
        w.put_small("hc_head.hc_scale", &dt2, &s2, dtype)?;
    }
    let gin = nh * hd / o_groups; // grouped o_proj input-per-group
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        norm(&mut w, &format!("l{l}.input_layernorm"), &format!("{p}input_layernorm.weight"))?;
        norm(&mut w, &format!("l{l}.post_attention_layernorm"), &format!("{p}post_attention_layernorm.weight"))?;
        // attention
        norm(&mut w, &format!("l{l}.self_attn.q_a_norm"), &format!("{p}self_attn.q_a_norm.weight"))?;
        norm(&mut w, &format!("l{l}.self_attn.kv_norm"), &format!("{p}self_attn.kv_norm.weight"))?;
        lin(&mut w, &format!("l{l}.self_attn.q_a_proj"), &format!("{p}self_attn.q_a_proj.weight"))?;
        lin(&mut w, &format!("l{l}.self_attn.q_b_proj"), &format!("{p}self_attn.q_b_proj.weight"))?;
        lin(&mut w, &format!("l{l}.self_attn.kv_proj"), &format!("{p}self_attn.kv_proj.weight"))?;
        lin(&mut w, &format!("l{l}.self_attn.o_b_proj"), &format!("{p}self_attn.o_b_proj.weight"))?;
        // grouped o_a_proj: HF weight (o_groups*o_lora, gin); write one block per group (o_lora, gin).
        let (oas, oad) = m.read(&format!("{p}self_attn.o_a_proj.weight"));
        assert_eq!(oas, vec![o_groups * o_lora, gin], "dsv4 o_a_proj shape");
        for g in 0..o_groups {
            let blk = &oad[g * o_lora * gin..(g + 1) * o_lora * gin];
            w.put_lin(&format!("l{l}.self_attn.o_a_proj.{g}"), blk, o_lora, gin, dtype)?;
        }
        let (sks, skd) = m.read(&format!("{p}self_attn.sinks"));
        w.put_small(&format!("l{l}.self_attn.sinks"), &skd, &sks, dtype)?;
        // mHC HyperConnections (attn + ffn sites)
        for hcn in ["attn_hc", "ffn_hc"] {
            lin(&mut w, &format!("l{l}.{hcn}.fn"), &format!("{p}{hcn}.fn"))?;
            let (bs, bd) = m.read(&format!("{p}{hcn}.base"));
            w.put_small(&format!("l{l}.{hcn}.base"), &bd, &bs, dtype)?;
            let (ss, sd) = m.read(&format!("{p}{hcn}.scale"));
            w.put_small(&format!("l{l}.{hcn}.scale"), &sd, &ss, dtype)?;
        }
        // MoE router + bias
        lin(&mut w, &format!("l{l}.mlp.gate"), &format!("{p}mlp.gate.weight"))?;
        let (bs, bd) = m.read(&format!("{p}mlp.gate.e_score_correction_bias"));
        w.put_small(&format!("l{l}.mlp.e_score_correction_bias"), &bd, &bs, dtype)?;
        // routed experts — saved UNPACKED per expert (Mixtral convention): w1=gate, w3=up, w2=down. One (in,out) array each.
        for e in 0..n_exp {
            lin(&mut w, &format!("l{l}.experts.{e}.gate"), &format!("{p}mlp.experts.{e}.w1.weight"))?;
            lin(&mut w, &format!("l{l}.experts.{e}.up"), &format!("{p}mlp.experts.{e}.w3.weight"))?;
            lin(&mut w, &format!("l{l}.experts.{e}.down"), &format!("{p}mlp.experts.{e}.w2.weight"))?;
        }
        // shared expert (SwiGLU)
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            lin(&mut w, &format!("l{l}.mlp.shared_experts.{proj}"), &format!("{p}mlp.shared_experts.{proj}.weight"))?;
        }
    }
    let n = w.arrays.len();
    w.finish(stem, manifest)?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eos_ids_int_array_none() {
        assert_eq!(eos_ids(&json!({"eos_token_id": 5})), vec![5]);
        assert_eq!(eos_ids(&json!({"eos_token_id": [1, 2, 3]})), vec![1, 2, 3]);
        assert_eq!(eos_ids(&json!({})), Vec::<i64>::new());
    }

    #[test]
    fn geti_getf_basic() {
        let c = json!({"a": 7, "b": 1.5});
        assert_eq!(geti(&c, "a"), Some(7));
        assert_eq!(geti(&c, "missing"), None);
        assert_eq!(getf(&c, "b"), Some(1.5));
    }

    #[test]
    fn gemma3_thetas_flat_nested_default() {
        assert_eq!(gemma3_thetas(&json!({"rope_local_base_freq": 10000.0, "rope_theta": 1000000.0})), (10000.0, 1000000.0));
        let nested = json!({"rope_parameters": {"sliding_attention": {"rope_theta": 1234.0}, "full_attention": {"rope_theta": 9999.0}}});
        assert_eq!(gemma3_thetas(&nested), (1234.0, 9999.0));
        assert_eq!(gemma3_thetas(&json!({})), (10000.0, 1000000.0));
    }

    #[test]
    fn int8_quant_roundtrip() {
        let dir = std::env::temp_dir().join(format!("fr_i8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut w = BundleWriter::new(&stem).unwrap();
        // (in=2, out=2) matrix [[1,2],[3,4]] stored as-is (transpose=false)
        w.put_i8("m", &[1.0, 2.0, 3.0, 4.0], 2, 2, false).unwrap();
        w.finish(&stem, json!({"format": "fieldrun-bundle", "version": 1, "arch": "test", "config": []})).unwrap();
        let b = crate::bundle::Bundle::load(&stem).unwrap();
        let (r0, r1) = (b.weight_row("m", 0), b.weight_row("m", 1));
        assert!((r0[0] - 1.0).abs() < 0.05 && (r0[1] - 2.0).abs() < 0.05, "{r0:?}");
        assert!((r1[0] - 3.0).abs() < 0.05 && (r1[1] - 4.0).abs() < 0.05, "{r1:?}");
    }

    /// The point of q4a: on a non-zero-centred weight (values clustered around +0.5), symmetric int4 wastes range on
    /// the unused negative half, while q4a's per-group `min` captures the offset — so at equal bytes q4a's
    /// reconstruction error must be far lower. Exercises the real convert→load→weight_row path for both dtypes.
    #[test]
    fn q4a_beats_int4_on_offset_data() {
        let (inp, out) = (128usize, 4usize);
        let data: Vec<f32> = (0..inp * out).map(|t| 0.5 + 0.18 * ((t % 37) as f32 / 37.0 - 0.5)).collect(); // ~[0.41, 0.59]
        let dir = std::env::temp_dir().join(format!("fr_q4a_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut w = BundleWriter::new(&stem).unwrap();
        w.put_i4("wi4", &data, inp, out, false).unwrap();
        w.put_q4a("wq4a", &data, inp, out, false).unwrap();
        w.finish(&stem, json!({"format": "fieldrun-bundle", "version": 1, "arch": "test", "config": []})).unwrap();
        let b = crate::bundle::Bundle::load(&stem).unwrap();
        let mse = |name: &str| -> f32 {
            let mut e = 0.0f32;
            for r in 0..inp {
                let row = b.weight_row(name, r);
                for j in 0..out {
                    let d = row[j] - data[r * out + j];
                    e += d * d;
                }
            }
            e / (inp * out) as f32
        };
        let (e_i4, e_q4a) = (mse("wi4"), mse("wq4a"));
        assert!(e_q4a < e_i4 * 0.5, "q4a MSE {e_q4a} should be << int4 MSE {e_i4} on offset data");
    }

    /// Row-major int8 embed (Phase 4b): rows_f32 must reconstruct each vocab row within the per-row int8 step, and
    /// rowdot_f32 / weight_row (the tied-unembed + explain paths) must agree with the dequantised row.
    #[test]
    fn rowi8_embed_roundtrip() {
        let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 1.0, -2.0, 0.5, 0.25, -0.7, 0.8, -0.9, 0.6]; // (vocab=3, d=4)
        let dir = std::env::temp_dir().join(format!("fr_rowi8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut w = BundleWriter::new(&stem).unwrap();
        w.put_embed_i8("embed", &data, 3, 4).unwrap();
        w.finish(&stem, json!({"format": "fieldrun-bundle", "version": 1, "arch": "test", "config": []})).unwrap();
        let b = crate::bundle::Bundle::load(&stem).unwrap();
        let rows = b.rows_f32("embed", &[1]); // row 1 = [1.0,-2.0,0.5,0.25]; amax 2.0 -> step ~0.0157
        for (g, &want) in rows.row(0).iter().zip(&data[4..8]) {
            assert!((g - want).abs() < 0.02, "rows_f32 got {g} want {want}");
        }
        let wr = b.weight_row("embed", 1); // explain path must match rows_f32
        assert!(wr.iter().zip(rows.row(0)).all(|(a, b)| (a - b).abs() < 1e-6), "weight_row != rows_f32");
        let dots = b.rowdot_f32("embed", &[1.0, 1.0, 1.0, 1.0]); // dot with ones = sum of the dequantised row
        assert!((dots[1] - rows.row(0).sum()).abs() < 1e-4, "rowdot {} vs {}", dots[1], rows.row(0).sum());
    }
}
