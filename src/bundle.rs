//! The flat weight bundle — the lm-sae -> fieldrun contract (`pylm/export_bundle.py`).
//!
//! A tiny JSON header (arch, config, and per-array dtype/shape/offset) plus one raw little-endian f32 blob. No zip, no
//! .npy parsing: read the header, slurp the blob, and view each array by offset. (Later: fp16/int8 blobs kept
//! low-precision in RAM and dequantised per matmul — the in-RAM-precision path. This first cut is f32.)

use std::collections::HashMap;

use memmap2::Mmap;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct ArrSpec {
    name: String,
    #[serde(default = "default_dtype")]
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    #[serde(default)]
    group: Option<usize>, // int4 group size (weights along the input dim sharing one scale); None for f32/f16/i8
}

fn default_dtype() -> String {
    "f32".to_string()
}

/// A weight array, kept in its on-disk precision in RAM. f16 arrays (the in-RAM-precision path) are upcast to f32
/// per access, so a big model (e.g. Gemma-2's 256k vocab → ~10 GB f32) stays ~half that resident.
enum Arr {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    I8(I8w), // per-output-column symmetric int8 (scale in sibling "<name>__scale"), repacked for the int8-dot path
    I4(I4w), // group-wise symmetric int4 (2 nibbles/byte, scale per out-col×group), dequantised to f32 on read
}

/// An int8 weight prepared for the int8 matmul: stored transposed to (out, in) so each output column's contiguous `k`
/// values feed one signed-int8 dot (stable NEON `vmull`/`vpadal` on aarch64, scalar elsewhere). Symmetric per-column quant, so no
/// zero-point term — the activation is quantised to signed int8 too and the dot is a plain s8×s8 accumulate.
struct I8w {
    wt: Vec<i8>, // (n, k) row-major: wt[j*k + kk] = W[kk, j]
    k: usize,
    n: usize,
}

/// A group-wise int4 weight: OUTPUT-COLUMN-MAJOR `(n=out, k=in)` packed 2 nibbles/byte along `in`, with an fp16 scale
/// per (output-column, group of `g` input values) in the sibling `<name>__scale`. Half the bytes of int8; dequantised
/// to f32 on read (a block at a time in `mm`). Symmetric `[-7,7]`, two's-complement nibble.
struct I4w {
    packed: Vec<u8>, // (n, ceil(k/2)) row-major: byte (j, kk/2) holds output-col j's input-pos kk nibble
    k: usize,        // in
    n: usize,        // out
    g: usize,        // group size along `in`
}

/// Sign-extend a 4-bit two's-complement value packed in `packed` at logical (output-col `j`, input-pos `i`).
#[inline]
fn i4_nibble(packed: &[u8], row_bytes: usize, j: usize, i: usize) -> i32 {
    let byte = packed[j * row_bytes + i / 2];
    let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    (((nib << 4) as i8) >> 4) as i32
}

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
    arch: String,
    config: Vec<i64>,
    #[serde(default)]
    config_f: Vec<f64>,
    #[serde(default)]
    eos: Vec<i64>,
    #[serde(default)]
    store: Option<serde_json::Value>,
    arrays: Vec<ArrSpec>,
}

/// An MoE expert weight that is NOT parsed into RAM at load — its bytes live in the mmap'd blob and are read +
/// dequantised on demand, so cold experts never occupy RAM (the OS page cache handles the working set). This is the
/// expert-offload contract: per token only the router's top-k experts are touched, so a model with far more expert
/// params than RAM still runs (and a hot expert stays warm in the page cache for free).
struct ExpertSpec {
    offset: usize,
    bytes: usize,
    shape: Vec<usize>,
    dtype: String,
    group: Option<usize>, // int4 group size (None for f16/i8 experts)
}

pub struct Bundle {
    pub arch: String,
    pub config: Vec<i64>,
    pub config_f: Vec<f64>,
    pub eos: Vec<i64>, // end-of-sequence token id(s) from the source config — used to stop API generation
    pub store: Option<serde_json::Value>,
    arrays: HashMap<String, (Vec<usize>, Arr)>,   // parsed once at load, kept in on-disk precision (the resident set)
    experts: HashMap<String, ExpertSpec>,         // MoE experts: read on demand from the mmap (paged, never resident)
    mmap: Mmap,                                    // the blob, kept mapped so expert reads page in on demand
}

pub const FORMAT: &str = "fieldrun-bundle";
pub const VERSION: u32 = 1;
const OUTLIER_T: usize = 32; // activation channels kept in f32 per row in the W8A8 path (outlier-aware quant)

impl Bundle {
    /// Load a fieldrun bundle: `<stem>.fieldrun.json` (manifest) + `<stem>.fieldrun.bin` (blob). f32 arrays hand out
    /// zero-copy views (`arr2`/`arr1`); f16 arrays stay f16 in RAM and upcast on demand (`arr2o`/`arr1o`).
    pub fn load(stem: &str) -> std::io::Result<Bundle> {
        let h: Header = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.fieldrun.json"))?)?;
        if h.format != FORMAT || h.version != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported bundle: {} v{} (this fieldrun reads {FORMAT} v{VERSION})", h.format, h.version),
            ));
        }
        // mmap the blob (not read-into-RAM): dense arrays parse out of it once at load; MoE expert weights stay mapped
        // and page in on demand. For a non-MoE model this reads only the dense pages — same resident footprint as before.
        let file = std::fs::File::open(format!("{stem}.fieldrun.bin"))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut experts = HashMap::new();
        let mut dense: Vec<ArrSpec> = Vec::new();
        for a in h.arrays {
            // MoE expert weights live on disk; their tiny per-column __scale siblings still parse into RAM.
            if a.name.contains(".experts.") && !a.name.ends_with("__scale") {
                experts.insert(a.name, ExpertSpec { offset: a.offset, bytes: a.bytes, shape: a.shape, dtype: a.dtype, group: a.group });
            } else {
                dense.push(a);
            }
        }
        // parse the dense arrays in parallel — the int8 transpose (the slow part of loading a big bundle) fans out
        // across cores, so a multi-GB model loads in a few seconds instead of tens.
        let arrays: HashMap<String, (Vec<usize>, Arr)> = dense
            .into_par_iter()
            .map(|a| {
                let raw = &mmap[a.offset..a.offset + a.bytes];
                let arr = match a.dtype.as_str() {
                    "f32" => Arr::F32(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                    "f16" => Arr::F16(raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect()),
                    "i8" => {
                        let (k, n) = (a.shape[0], a.shape[1]);
                        let mut wt = vec![0i8; k * n];
                        for kk in 0..k {
                            let base = kk * n;
                            for j in 0..n {
                                wt[j * k + kk] = raw[base + j] as i8; // transpose to (out, in) for contiguous-k int8 dots
                            }
                        }
                        Arr::I8(I8w { wt, k, n })
                    }
                    "i4" => {
                        // stored (out, in) output-column-major, 2 nibbles/byte along `in`; kept packed, dequant per mm.
                        let (n, k) = (a.shape[0], a.shape[1]); // (out, in)
                        Arr::I4(I4w { packed: raw.to_vec(), k, n, g: a.group.unwrap_or(32) })
                    }
                    d => panic!("unsupported array dtype {d:?} in bundle"),
                };
                (a.name, (a.shape, arr))
            })
            .collect();
        Ok(Bundle { arch: h.arch, config: h.config, config_f: h.config_f, eos: h.eos, store: h.store, arrays, experts, mmap })
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Arr) {
        self.arrays.get(name).unwrap_or_else(|| panic!("missing array {name}"))
    }

    pub fn has_expert(&self, name: &str) -> bool {
        self.experts.contains_key(name)
    }

    /// Hint the kernel to read-ahead a routed expert's mmap pages (`MADV_WILLNEED`) so the page-in overlaps with compute
    /// on the previously-selected experts. Pure performance hint: a no-op for correctness, for unknown names, and on
    /// non-unix targets. The MoE forward calls this for every active top-k expert up front, so the kernel pages experts
    /// 2..k in while expert 1 is being computed — turning k serial page-fault stalls (the decode bottleneck under expert
    /// offload) into one overlapped readahead. Does not change the resident set: these are exactly the experts this
    /// forward will touch anyway.
    #[cfg(unix)]
    pub fn prefetch(&self, name: &str) {
        if let Some(e) = self.experts.get(name) {
            let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, e.offset, e.bytes);
        }
    }
    #[cfg(not(unix))]
    pub fn prefetch(&self, _name: &str) {}

    /// Read one MoE expert weight on demand from the mmap and dequantise to f32 (i8 via its per-column `__scale`
    /// sibling, which is resident). Cold experts fault in from disk; hot ones stay in the OS page cache. Returns
    /// (shape, data) with the same (in, out) row-major layout `mm` expects.
    pub fn expert_f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let e = self.experts.get(name).unwrap_or_else(|| panic!("missing expert array {name}"));
        let raw = &self.mmap[e.offset..e.offset + e.bytes];
        // Returns (in, out) row-major (what `expert_mm` dots against). i8/f16/f32 store (in, out) already; i4 stores
        // (out, in) packed (group contiguity) and is transposed here while dequantising.
        let (shape, v) = match e.dtype.as_str() {
            "i8" => {
                let (inp, out) = (e.shape[0], e.shape[1]); // stored (in, out) row-major (put_i8 transpose)
                let scale = self.arr1o(&format!("{name}__scale"));
                let mut v = vec![0f32; inp * out];
                for i in 0..inp {
                    for j in 0..out {
                        v[i * out + j] = raw[i * out + j] as i8 as f32 * scale[j];
                    }
                }
                (e.shape.clone(), v)
            }
            "i4" => {
                let (n, k) = (e.shape[0], e.shape[1]); // stored (out, in); dequant to (in, out)
                let g = e.group.unwrap_or(32);
                let (ng, row_bytes) = (k.div_ceil(g), k.div_ceil(2));
                let scale = self.arr1o(&format!("{name}__scale"));
                let mut v = vec![0f32; k * n];
                for j in 0..n {
                    for i in 0..k {
                        v[i * n + j] = i4_nibble(raw, row_bytes, j, i) as f32 * scale[j * ng + i / g];
                    }
                }
                (vec![k, n], v)
            }
            "f16" => (e.shape.clone(), raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()),
            "f32" => (e.shape.clone(), raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            d => panic!("expert dtype {d:?} unsupported"),
        };
        (shape, v)
    }

    /// `x (tokens, in) @ expert_W (in, out)` for an on-demand expert weight — dequantised from the mmap per call.
    pub fn expert_mm(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let (shape, w) = self.expert_f32(name);
        let wv = ArrayView2::from_shape((shape[0], shape[1]), &w).unwrap();
        x.dot(&wv)
    }

    /// Logical row `r` of an offloaded expert weight (stored (in, out) row-major) as f32 — the analogue of
    /// `weight_row` for the mmap'd MoE experts, so explain can name a MoE neuron's promoted tokens (its down-row
    /// projected to the unembed). One expert is paged in per call; explain touches one neuron per layer, so this is
    /// cheap. Falls back to `weight_row` if `name` is actually a dense (non-offloaded) array.
    pub fn expert_row(&self, name: &str, r: usize) -> Vec<f32> {
        if !self.experts.contains_key(name) {
            return self.weight_row(name, r);
        }
        let (shape, w) = self.expert_f32(name);
        let out = shape[1];
        w[r * out..(r + 1) * out].to_vec()
    }

    /// Tier C — routed MLP down-projection. For each row keep only the top `frac` neurons by |activation| and sum just
    /// their down-rows (a sparse axpy), skipping the rest entirely: real conditional compute, ~`frac` of the work.
    /// Numerically identical to zeroing the bottom (1-frac) neurons then a dense down-proj (the pylm `--route-frac`).
    pub fn mm_routed_down(&self, h: &Array2<f32>, name: &str, frac: f32) -> Array2<f32> {
        let (shape, arr) = self.get(name); // down: (ffn, d)
        if matches!(arr, Arr::I8(_) | Arr::I4(_)) {
            return self.mm(h, name); // int8/int4 down is quantised → routing falls back to a dense mm for now
        }
        let (ffn, d) = (shape[0], shape[1]);
        let keep = ((frac * ffn as f32).ceil() as usize).clamp(1, ffn);
        let mut out = Array2::<f32>::zeros((h.nrows(), d));
        for i in 0..h.nrows() {
            let hrow = h.row(i);
            let hrow = hrow.as_slice().unwrap();
            let mut idx: Vec<usize> = (0..ffn).collect();
            idx.select_nth_unstable_by(keep.min(ffn - 1), |&x, &y| {
                hrow[y].abs().partial_cmp(&hrow[x].abs()).unwrap()
            });
            let mut acc = vec![0f32; d];
            for &k in &idx[0..keep] {
                let hk = hrow[k];
                match arr {
                    Arr::F32(v) => acc.iter_mut().zip(&v[k * d..(k + 1) * d]).for_each(|(a, &w)| *a += hk * w),
                    Arr::F16(v) => acc.iter_mut().zip(&v[k * d..(k + 1) * d]).for_each(|(a, w)| *a += hk * w.to_f32()),
                    Arr::I8(_) | Arr::I4(_) => unreachable!(),
                }
            }
            out.row_mut(i).assign(&ArrayView1::from(acc.as_slice()));
        }
        out
    }

    pub fn has(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    /// Whole array as (shape, f32 data) — upcasts f16/copies f32 (i8 panics; the GPU path uses f32/f16 bundles).
    pub fn f32_array(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (shape, arr) = self.get(name);
        (shape.clone(), self.upcast(arr))
    }

    /// Logical row r of a (rows, cols) weight as f32, dtype-agnostic (i8 is dequantised from its transposed store via
    /// the per-column scale). Used for explain's neuron labels so they work on int8 bundles too.
    pub fn weight_row(&self, name: &str, r: usize) -> Vec<f32> {
        let (shape, arr) = self.get(name);
        let cols = shape[1];
        match arr {
            Arr::F32(v) => v[r * cols..(r + 1) * cols].to_vec(),
            Arr::F16(v) => v[r * cols..(r + 1) * cols].iter().map(|h| h.to_f32()).collect(),
            Arr::I8(w8) => {
                let scale = self.arr1o(&format!("{name}__scale"));
                (0..w8.n).map(|j| w8.wt[j * w8.k + r] as f32 * scale[j]).collect()
            }
            Arr::I4(w4) => {
                let scale = self.arr1o(&format!("{name}__scale"));
                let (ng, row_bytes) = (w4.k.div_ceil(w4.g), w4.k.div_ceil(2));
                (0..w4.n).map(|j| i4_nibble(&w4.packed, row_bytes, j, r) as f32 * scale[j * ng + r / w4.g]).collect()
            }
        }
    }

    // Zero-copy f32 views — for f32 bundles (GPT-2 / RoPE). Panics on an f16 array (use arr2o/arr1o).
    pub fn arr2(&self, name: &str) -> ArrayView2<'_, f32> {
        let (shape, arr) = self.get(name);
        match arr {
            Arr::F32(v) => ArrayView2::from_shape((shape[0], shape[1]), v).unwrap(),
            _ => panic!("arr2: {name} is not f32; use arr2o (owned, upcast) or mm (quantised)"),
        }
    }

    /// 1D weight as an owned f32 vector — upcasts f16 (norms/biases are tiny under fp16/int8 bundles, so copying is
    /// free) so the kernels are dtype-agnostic. (The big 2D weights go through `mm`/`weight_row`, not here.)
    pub fn arr1(&self, name: &str) -> Array1<f32> {
        self.arr1o(name)
    }

    // Owned f32 — upcasts f16 (or copies f32) per call. The in-RAM-precision path (Gemma keeps weights f16 in RAM).
    pub fn arr2o(&self, name: &str) -> Array2<f32> {
        let (shape, arr) = self.get(name);
        Array2::from_shape_vec((shape[0], shape[1]), self.upcast(arr)).unwrap()
    }

    // NB: the f32/f16 GEMM goes through ndarray's `.dot()`, which routes to a tuned cblas (sgemm) when built with a
    // BLAS backend (`--features accelerate`/`openblas`/`blis`) — far faster for dense models; the pure-Rust column-block
    // path is the default + faithful reference. int8 keeps its own dot regardless. See the `[features]` in Cargo.toml.
    /// Fused matmul `a (seq, K) @ W (K, N) -> (seq, N)`. For an f16 weight this converts on the fly in a cache-friendly
    /// `ikj` loop (contiguous W-row and output-row access) — no f32 copy of the weight, which is the real cost of the
    /// in-RAM-precision path (a fresh upcast per matmul is ~GBs of alloc/write per forward). f32 falls back to ndarray.
    pub fn mm(&self, a: &Array2<f32>, name: &str) -> Array2<f32> {
        let (shape, arr) = self.get(name);
        let (k, n) = (shape[0], shape[1]);
        match arr {
            Arr::F32(v) => {
                let w = ArrayView2::from_shape((k, n), v).unwrap();
                // With a BLAS backend (--features accelerate/openblas/blis), hand the whole GEMM to cblas sgemm, which
                // is tuned + threads internally — far faster than the pure-Rust kernel on big dense models.
                #[cfg(feature = "blas")]
                {
                    a.dot(&w)
                }
                // Pure-Rust default: parallelise over output-column blocks (each independent + exact). Single-stream
                // generation gets all cores per forward; under the scoring loop's outer rayon it work-steals.
                #[cfg(not(feature = "blas"))]
                {
                    let block = 512.min(n.max(1));
                    let nblocks = n.div_ceil(block);
                    if nblocks <= 1 {
                        return a.dot(&w);
                    }
                    let mut blocks: Vec<(usize, Array2<f32>)> = (0..nblocks)
                        .into_par_iter()
                        .map(|bi| {
                            let c0 = bi * block;
                            let bw = block.min(n - c0);
                            (c0, a.dot(&w.slice(s![.., c0..c0 + bw])))
                        })
                        .collect();
                    let mut out = Array2::<f32>::zeros((a.nrows(), n));
                    for (c0, ob) in blocks.drain(..) {
                        let bw = ob.ncols();
                        out.slice_mut(s![.., c0..c0 + bw]).assign(&ob);
                    }
                    out
                }
            }
            // int8 W8A8: dynamically quantise each activation row to signed int8, take a signed int8·int8 dot
            // (stable NEON `vmull`/`vpadal` on aarch64, scalar elsewhere), then dequant: out = scale_a*scale_w[j]*Σ a·w + corr.
            // Symmetric per-column quant means no zero-point term. Activations go to int8 too, so this is lossier than
            // the f16-activation dequant path — it trades a little accuracy for the on-core int8 dot.
            Arr::I8(w8) => {
                let scale = self.arr1o(&format!("{name}__scale"));
                let (kk, nn) = (w8.k, w8.n);
                let t = OUTLIER_T.min(kk); // keep the T largest-magnitude activation channels in f32
                let mut out = Array2::<f32>::zeros((a.nrows(), nn));
                for i in 0..a.nrows() {
                    let arow = a.row(i);
                    let arow = arow.as_slice().unwrap();
                    // outlier-aware activation quant: the int8 scale fits the BULK (outliers excluded), and the few
                    // outlier channels are added back exactly in f32 — the massive-activation outliers are what wreck
                    // a naive per-token scale (TurboQuant's insight), so handling them recovers most of the W8A8 loss.
                    let absv: Vec<f32> = arow.iter().map(|v| v.abs()).collect();
                    let mut idx: Vec<usize> = (0..kk).collect();
                    // partition so idx[0..t] are the t largest-|a| channels; skip when t==kk (every channel is an
                    // outlier → the whole dot is taken exactly in f32 via `corr`). select_nth_unstable_by needs t<len.
                    if t < kk {
                        idx.select_nth_unstable_by(t, |&x, &y| absv[y].partial_cmp(&absv[x]).unwrap());
                    }
                    let bulk_max = if t < kk { absv[idx[t]] } else { 0.0 };
                    let sa = if bulk_max > 0.0 { bulk_max / 127.0 } else { 1.0 };
                    // signed-int8 activations; outlier channels zeroed here and added back exactly in f32 via `corr`.
                    let mut a8: Vec<i8> = arow.iter().map(|&v| ((v / sa).round() as i32).clamp(-127, 127) as i8).collect();
                    let ov: Vec<(usize, f32)> = idx[0..t].iter().map(|&o| { a8[o] = 0; (o, arow[o]) }).collect();
                    let orow: Vec<f32> = (0..nn)
                        .into_par_iter()
                        .map(|j| {
                            let base = j * kk;
                            let acc = i8dot(&a8, &w8.wt[base..base + kk]);
                            let corr: f32 = ov.iter().map(|&(ch, av)| av * w8.wt[base + ch] as f32).sum();
                            scale[j] * (sa * acc as f32 + corr)
                        })
                        .collect();
                    out.row_mut(i).assign(&ArrayView1::from(orow.as_slice()));
                }
                out
            }
            // group-wise int4: unpack two nibbles/byte + apply the per-group scale into a local f32 block, then GEMM the
            // block (a W4A_f32 path — only the weight is quantised, like the f16 path). Half the bytes of int8 on disk;
            // a NEON s4×s8 int dot is a later optimization. Output columns are independent → fan blocks out across cores.
            Arr::I4(w4) => {
                let scale = self.arr1o(&format!("{name}__scale")); // (n, ng) f32
                let (k, nn, g) = (w4.k, w4.n, w4.g);
                let (ng, row_bytes) = (k.div_ceil(g), k.div_ceil(2));
                let packed = &w4.packed;
                let block = 512.min(nn.max(1));
                let mut blocks: Vec<(usize, Array2<f32>)> = (0..nn.div_ceil(block))
                    .into_par_iter()
                    .map(|bi| {
                        let (c0, bw) = (bi * block, block.min(nn - bi * block));
                        let mut buf = vec![0f32; k * bw]; // (k, bw) for a.dot
                        for col in 0..bw {
                            let j = c0 + col;
                            for kk in 0..k {
                                buf[kk * bw + col] = i4_nibble(packed, row_bytes, j, kk) as f32 * scale[j * ng + kk / g];
                            }
                        }
                        let wblock = ArrayView2::from_shape((k, bw), &buf).unwrap();
                        (c0, a.dot(&wblock))
                    })
                    .collect();
                let mut out = Array2::<f32>::zeros((a.nrows(), nn));
                for (c0, ob) in blocks.drain(..) {
                    let bw = ob.ncols();
                    out.slice_mut(s![.., c0..c0 + bw]).assign(&ob);
                }
                out
            }
            Arr::F16(v) => {
                // Upcast W one column-block at a time into a local f32 buffer, then GEMM the block — this keeps the
                // vectorised matmul while bounding the f32 upcast to one block (so a multi-GB f16 weight is never
                // fully materialised as f32). Block width 512.
                let block = 512.min(n.max(1));
                // With BLAS: upcast + sgemm each block serially (cblas threads internally; one f32 block buffer live).
                #[cfg(feature = "blas")]
                {
                    let mut out = Array2::<f32>::zeros((a.nrows(), n));
                    let mut c0 = 0;
                    while c0 < n {
                        let bw = block.min(n - c0);
                        let mut buf = vec![0f32; k * bw];
                        for kk in 0..k {
                            let wrow = &v[kk * n + c0..kk * n + c0 + bw];
                            f16_to_f32(wrow, &mut buf[kk * bw..kk * bw + bw]);
                        }
                        let wblock = ArrayView2::from_shape((k, bw), &buf).unwrap();
                        out.slice_mut(s![.., c0..c0 + bw]).assign(&a.dot(&wblock));
                        c0 += bw;
                    }
                    out
                }
                // Pure-Rust default: blocks are independent (disjoint output columns), so run them in parallel.
                #[cfg(not(feature = "blas"))]
                {
                    let nblocks = n.div_ceil(block);
                    let mut blocks: Vec<(usize, Array2<f32>)> = (0..nblocks)
                        .into_par_iter()
                        .map(|bi| {
                            let c0 = bi * block;
                            let bw = block.min(n - c0);
                            let mut buf = vec![0f32; k * bw];
                            for kk in 0..k {
                                let wrow = &v[kk * n + c0..kk * n + c0 + bw];
                                f16_to_f32(wrow, &mut buf[kk * bw..kk * bw + bw]);
                            }
                            let wblock = ArrayView2::from_shape((k, bw), &buf).unwrap();
                            (c0, a.dot(&wblock))
                        })
                        .collect();
                    let mut out = Array2::<f32>::zeros((a.nrows(), n));
                    for (c0, ob) in blocks.drain(..) {
                        let bw = ob.ncols();
                        out.slice_mut(s![.., c0..c0 + bw]).assign(&ob);
                    }
                    out
                }
            }
        }
    }

    pub fn arr1o(&self, name: &str) -> Array1<f32> {
        let (_, arr) = self.get(name);
        Array1::from_vec(self.upcast(arr))
    }

    fn upcast(&self, arr: &Arr) -> Vec<f32> {
        match arr {
            Arr::F32(v) => v.clone(),
            Arr::F16(v) => v.iter().map(|h| h.to_f32()).collect(),
            Arr::I8(_) | Arr::I4(_) => panic!("upcast: quantised weight needs its scale; go through mm()"),
        }
    }

    /// Upcast only the given rows of a (rows, cols) matrix — for an embedding lookup, so a 256k-row table is never
    /// materialised in f32 (the OOM trap when 16 forwards each upcast the whole 2.36 GB embed).
    pub fn rows_f32(&self, name: &str, ids: &[i64]) -> Array2<f32> {
        let (shape, arr) = self.get(name);
        let d = shape[1];
        let mut out = Array2::<f32>::zeros((ids.len(), d));
        for (t, &id) in ids.iter().enumerate() {
            let base = id as usize * d;
            match arr {
                Arr::F32(v) => out.row_mut(t).iter_mut().zip(&v[base..base + d]).for_each(|(o, &s)| *o = s),
                Arr::F16(v) => out.row_mut(t).iter_mut().zip(&v[base..base + d]).for_each(|(o, s)| *o = s.to_f32()),
                Arr::I8(_) | Arr::I4(_) => panic!("rows_f32: quantised embed unsupported (embed stays f16)"),
            }
        }
        out
    }

    /// For each row r of a (rows, cols) matrix W, compute dot(W[r], x) → (rows,). Keeps W in its stored precision, so
    /// the tied unembed over a huge vocab streams f16 weights without a (vocab, d) f32 allocation.
    pub fn rowdot_f32(&self, name: &str, x: &[f32]) -> Vec<f32> {
        let (shape, arr) = self.get(name);
        let (rows, d) = (shape[0], shape[1]);
        // The tied unembed over a 256k vocab is ~the biggest per-token cost; rows are independent → fan out over cores.
        (0..rows)
            .into_par_iter()
            .map(|r| {
                let base = r * d;
                match arr {
                    Arr::F32(v) => v[base..base + d].iter().zip(x).map(|(&w, &xi)| w * xi).sum(),
                    Arr::F16(v) => v[base..base + d].iter().zip(x).map(|(w, &xi)| w.to_f32() * xi).sum(),
                    Arr::I8(_) | Arr::I4(_) => panic!("rowdot_f32: quantised unembed unsupported (embed stays f16)"),
                }
            })
            .collect()
    }
}

/// Dot product of signed-int8 activations with signed-int8 weights, accumulating in i32. Hand-vectorised per target
/// with **stable** intrinsics, runtime-dispatched; every path is **bit-exact** to the scalar fallback (i8×i8 products
/// fit in i16 and the i32 sum never overflows for our row widths, so reordering the adds is exact — it cannot move an
/// argmax, so the faithfulness gate is unaffected):
///  - **aarch64**: `vmull_s8` (s8×s8 → s16) then `vpadalq_s16` (pairwise-add into i32), 16 lanes/iter. (We avoid the
///    one-instruction `sdot`/`vdotq_s32` — it's gated behind the unstable `stdarch_neon_dotprod` feature = nightly.)
///  - **x86-64**: sign-extend i8→i16 (`cvtepi8_epi16`) then `madd_epi16` (i16×i16 → i32 pairwise) into an i32 vector
///    accumulator. AVX-512-BW does 32 lanes/iter, AVX2 16 lanes/iter. AVX-512 intrinsics are stable since Rust 1.89
///    (the project MSRV), AVX2 since 1.27 — no nightly. This replaces the old scalar-only x86 path.
/// NEON is baseline on aarch64 so its check always passes; the scalar fallback keeps any other CPU correct.
fn i8dot(a: &[i8], w: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { sdot_neon(a, w) };
        }
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "_scalar_i8_bench")))]
    {
        if std::arch::is_x86_feature_detected!("avx512bw") && std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { i8dot_avx512bw(a, w) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { i8dot_avx2(a, w) };
        }
    }
    a.iter().zip(w).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sdot_neon(a: &[i8], w: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let len = a.len();
    let chunks = len / 16;
    let mut acc = vdupq_n_s32(0);
    for c in 0..chunks {
        let av = vld1q_s8(a.as_ptr().add(c * 16));
        let wv = vld1q_s8(w.as_ptr().add(c * 16));
        // s8×s8 -> s16 products for the low and high 8 lanes, then pairwise-add each into the i32 accumulator.
        let p_lo = vmull_s8(vget_low_s8(av), vget_low_s8(wv)); // int16x8
        let p_hi = vmull_s8(vget_high_s8(av), vget_high_s8(wv)); // int16x8
        acc = vpadalq_s16(acc, p_lo); // acc[i] += p_lo[2i] + p_lo[2i+1]
        acc = vpadalq_s16(acc, p_hi);
    }
    let mut sum = vaddvq_s32(acc); // horizontal add of the 4 lanes
    for k in (chunks * 16)..len {
        sum += a[k] as i32 * w[k] as i32; // tail (k not a multiple of 16)
    }
    sum
}

/// AVX2 signed-int8 dot: sign-extend 16 i8 → 16 i16 (`cvtepi8_epi16`), `madd_epi16` to 8 i32 pairwise products,
/// accumulate, horizontal-sum, scalar tail. Bit-exact to scalar (integer, no overflow for our widths).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i8dot_avx2(a: &[i8], w: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 16;
    let mut acc = _mm256_setzero_si256();
    for c in 0..chunks {
        let av = _mm_loadu_si128(a.as_ptr().add(c * 16) as *const __m128i); // 16 i8
        let wv = _mm_loadu_si128(w.as_ptr().add(c * 16) as *const __m128i);
        let a16 = _mm256_cvtepi8_epi16(av); // 16 i16 (sign-extended)
        let w16 = _mm256_cvtepi8_epi16(wv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16)); // 8 i32 pairwise products
    }
    // horizontal sum of the 8 i32 lanes
    let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
    let mut sum = _mm_cvtsi128_si32(s32);
    for k in (chunks * 16)..len {
        sum += a[k] as i32 * w[k] as i32; // tail (k not a multiple of 16)
    }
    sum
}

/// AVX-512-BW signed-int8 dot: same shape as AVX2 at 32 lanes/iter (`_mm512_cvtepi8_epi16` → `_mm512_madd_epi16` →
/// `_mm512_reduce_add_epi32`). Stable since Rust 1.89 (the MSRV). Bit-exact to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f")]
unsafe fn i8dot_avx512bw(a: &[i8], w: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 32;
    let mut acc = _mm512_setzero_si512();
    for c in 0..chunks {
        let av = _mm256_loadu_si256(a.as_ptr().add(c * 32) as *const __m256i); // 32 i8
        let wv = _mm256_loadu_si256(w.as_ptr().add(c * 32) as *const __m256i);
        let a16 = _mm512_cvtepi8_epi16(av); // 32 i16 (sign-extended)
        let w16 = _mm512_cvtepi8_epi16(wv);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(a16, w16)); // 16 i32 pairwise products
    }
    let mut sum = _mm512_reduce_add_epi32(acc);
    for k in (chunks * 32)..len {
        sum += a[k] as i32 * w[k] as i32; // tail (k not a multiple of 32)
    }
    sum
}

/// Convert a contiguous f16 slice to f32 into `dst`. Uses F16C hardware conversion (`_mm256_cvtph_ps`, 8 lanes/instr)
/// on x86 when available, else the scalar `half` path. IEEE f16→f32 is exact (every f16 is representable in f32), so
/// this is **bit-identical** to the scalar loop — it just does the upcast 8-wide instead of one `to_f32` call at a
/// time. `half::f16` is `repr(transparent)` over `u16`, so the slice reinterprets without a copy.
#[inline]
fn f16_to_f32(src: &[half::f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
            unsafe {
                f16_to_f32_f16c(src, dst);
            }
            return;
        }
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn f16_to_f32_f16c(src: &[half::f16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 8;
    let sp = src.as_ptr() as *const u16; // f16 is repr(transparent) over u16
    let dp = dst.as_mut_ptr();
    for c in 0..chunks {
        let h = _mm_loadu_si128(sp.add(c * 8) as *const __m128i); // 8 f16 (128 bits)
        _mm256_storeu_ps(dp.add(c * 8), _mm256_cvtph_ps(h)); // -> 8 f32
    }
    for i in (chunks * 8)..len {
        *dp.add(i) = (*src.get_unchecked(i)).to_f32(); // tail (len not a multiple of 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal bundle (manifest + blob) to `stem` for tests. Arrays: (name, dtype f32|f16, shape, data).
    fn write_bundle(stem: &str, arch: &str, config: &[i64], arrays: &[(&str, &str, Vec<usize>, Vec<f32>)]) {
        let mut bin: Vec<u8> = Vec::new();
        let mut specs: Vec<serde_json::Value> = Vec::new();
        for (name, dtype, shape, data) in arrays {
            let offset = bin.len();
            match *dtype {
                "f32" => for &v in data { bin.extend_from_slice(&v.to_le_bytes()); },
                "f16" => for &v in data { bin.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()); },
                d => panic!("test dtype {d}"),
            }
            specs.push(serde_json::json!({ "name": name, "dtype": dtype, "shape": shape, "offset": offset,
                "bytes": bin.len() - offset }));
        }
        let manifest = serde_json::json!({ "format": FORMAT, "version": VERSION, "arch": arch, "config": config,
            "config_f": [], "arrays": specs });
        std::fs::write(format!("{stem}.fieldrun.bin"), &bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
    }

    fn tmp_stem(tag: &str) -> String {
        let dir = std::env::temp_dir().join(format!("fr_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("b").to_string_lossy().into_owned()
    }

    #[test]
    fn mm_f32_exact() {
        let stem = tmp_stem("mm32");
        // W (k=2, n=3) row-major [[1,2,3],[4,5,6]]
        write_bundle(&stem, "test", &[], &[("w", "f32", vec![2, 3], vec![1., 2., 3., 4., 5., 6.])]);
        let b = Bundle::load(&stem).unwrap();
        let a = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let out = b.mm(&a, "w"); // [1+4, 2+5, 3+6]
        assert_eq!(out.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn mm_f16_close_to_f32() {
        let stem = tmp_stem("mm16");
        write_bundle(&stem, "test", &[], &[("w", "f16", vec![2, 2], vec![0.5, -1.5, 2.0, 0.25])]);
        let b = Bundle::load(&stem).unwrap();
        let a = Array2::from_shape_vec((1, 2), vec![2.0, 4.0]).unwrap();
        let out = b.mm(&a, "w"); // [2*0.5+4*2.0, 2*-1.5+4*0.25] = [9.0, -2.0]
        assert!((out[[0, 0]] - 9.0).abs() < 1e-2 && (out[[0, 1]] + 2.0).abs() < 1e-2, "{out:?}");
    }

    #[test]
    fn rows_and_rowdot() {
        let stem = tmp_stem("emb");
        // embed (vocab=3, d=2)
        write_bundle(&stem, "test", &[], &[("embed", "f32", vec![3, 2], vec![0., 1., 2., 3., 4., 5.])]);
        let b = Bundle::load(&stem).unwrap();
        let rows = b.rows_f32("embed", &[2, 0]);
        assert_eq!(rows.row(0).to_vec(), vec![4.0, 5.0]);
        assert_eq!(rows.row(1).to_vec(), vec![0.0, 1.0]);
        // rowdot: dot(embed[r], x) for x=[1,1] -> [0+1, 2+3, 4+5] = [1,5,9]
        assert_eq!(b.rowdot_f32("embed", &[1.0, 1.0]), vec![1.0, 5.0, 9.0]);
        // weight_row
        assert_eq!(b.weight_row("embed", 1), vec![2.0, 3.0]);
    }

    #[test]
    fn i8dot_matches_manual() {
        // a length that is not a multiple of 16 exercises the NEON tail loop / scalar path identically.
        let a: Vec<i8> = vec![1, -2, 3, 4, -5, 6, -7, 8, 9, -10, 11, -12, 13, 14, -15, 16, -17, 18];
        let w: Vec<i8> = vec![-1, 2, -3, 4, 5, -6, 7, 8, -9, 10, -11, 12, 13, -14, 15, 16, 17, -18];
        let want: i32 = a.iter().zip(&w).map(|(&x, &y)| x as i32 * y as i32).sum();
        assert_eq!(i8dot(&a, &w), want);
    }

    /// The SIMD i8dot (NEON / AVX2 / AVX-512, whichever the host dispatches to) must be **bit-exact** to the scalar
    /// reference across many lengths — including non-multiples of 16/32 (tail), all-extreme values (overflow headroom),
    /// and the longest realistic row width. Bit-exactness is what lets the SIMD path ship without touching the gate.
    #[test]
    fn i8dot_simd_vs_scalar() {
        let scalar = |a: &[i8], w: &[i8]| -> i32 { a.iter().zip(w).map(|(&x, &y)| x as i32 * y as i32).sum() };
        // a cheap deterministic PRNG so we don't pull in a dep; covers lengths that stress every lane count + tail.
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as i8 // full i8 range incl. -128
        };
        for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 896, 4864] {
            let a: Vec<i8> = (0..len).map(|_| next()).collect();
            let w: Vec<i8> = (0..len).map(|_| next()).collect();
            assert_eq!(i8dot(&a, &w), scalar(&a, &w), "len {len}");
        }
        // worst-case magnitude (all ±127) at the widest row — confirms the i32 accumulator never overflows.
        let a = vec![127i8; 4864];
        let w = vec![-128i8; 4864];
        assert_eq!(i8dot(&a, &w), scalar(&a, &w));
    }

    /// The F16C upcast must be bit-identical to `half::to_f32` for EVERY f16 bit pattern (subnormals, inf, nan, ±max)
    /// — IEEE f16→f32 is exact, so the SIMD path can't perturb a weight. Also covers non-multiple-of-8 tail lengths.
    #[test]
    fn f16_to_f32_simd_vs_scalar() {
        let all: Vec<half::f16> = (0u16..=u16::MAX).map(half::f16::from_bits).collect();
        for &len in &[0usize, 1, 7, 8, 9, 15, 16, 17, all.len()] {
            let src = &all[..len];
            let mut got = vec![0f32; len];
            f16_to_f32(src, &mut got);
            for (g, s) in got.iter().zip(src) {
                let want = s.to_f32();
                assert!(g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()), "f16 {:#06x}", s.to_bits());
            }
        }
    }
}
