//! The flat weight bundle — the lm-sae -> fieldrun contract (`pylm/export_bundle.py`).
//!
//! A tiny JSON header (arch, config, and per-array dtype/shape/offset) plus one raw little-endian f32 blob. No zip, no
//! .npy parsing: read the header, slurp the blob, and view each array by offset. (Later: fp16/int8 blobs kept
//! low-precision in RAM and dequantised per matmul — the in-RAM-precision path. This first cut is f32.)

use std::collections::HashMap;

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
}

fn default_dtype() -> String {
    "f32".to_string()
}

/// A weight array, kept in its on-disk precision in RAM. f16 arrays (the in-RAM-precision path) are upcast to f32
/// per access, so a big model (e.g. Gemma-2's 256k vocab → ~10 GB f32) stays ~half that resident.
enum Arr {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
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
    store: Option<serde_json::Value>,
    arrays: Vec<ArrSpec>,
}

pub struct Bundle {
    pub arch: String,
    pub config: Vec<i64>,
    pub config_f: Vec<f64>,
    pub store: Option<serde_json::Value>,
    arrays: HashMap<String, (Vec<usize>, Arr)>,   // parsed once at load, kept in on-disk precision
}

pub const FORMAT: &str = "fieldrun-bundle";
pub const VERSION: u32 = 1;

impl Bundle {
    /// Load a fieldrun bundle: `<stem>.fieldrun.json` (manifest) + `<stem>.fieldrun.bin` (blob). f32 arrays hand out
    /// zero-copy views (`arr2`/`arr1`); f16 arrays stay f16 in RAM and upcast on demand (`arr2o`/`arr1o`).
    pub fn load(stem: &str) -> std::io::Result<Bundle> {
        let h: Header = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.fieldrun.json"))?)?;
        if h.format != FORMAT || h.version != VERSION {
            panic!("unsupported bundle: {} v{} (this fieldrun reads {FORMAT} v{VERSION})", h.format, h.version);
        }
        let data = std::fs::read(format!("{stem}.fieldrun.bin"))?;
        let mut arrays = HashMap::new();
        for a in h.arrays {
            let raw = &data[a.offset..a.offset + a.bytes];
            let arr = match a.dtype.as_str() {
                "f32" => Arr::F32(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                "f16" => Arr::F16(raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect()),
                d => panic!("unsupported array dtype {d:?} in bundle"),
            };
            arrays.insert(a.name, (a.shape, arr));
        }
        Ok(Bundle { arch: h.arch, config: h.config, config_f: h.config_f, store: h.store, arrays })
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Arr) {
        self.arrays.get(name).unwrap_or_else(|| panic!("missing array {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    // Zero-copy f32 views — for f32 bundles (GPT-2 / RoPE). Panics on an f16 array (use arr2o/arr1o).
    pub fn arr2(&self, name: &str) -> ArrayView2<f32> {
        let (shape, arr) = self.get(name);
        match arr {
            Arr::F32(v) => ArrayView2::from_shape((shape[0], shape[1]), v).unwrap(),
            Arr::F16(_) => panic!("arr2: {name} is f16; use arr2o (owned, upcast)"),
        }
    }

    pub fn arr1(&self, name: &str) -> ArrayView1<f32> {
        let (_, arr) = self.get(name);
        match arr {
            Arr::F32(v) => ArrayView1::from(v.as_slice()),
            Arr::F16(_) => panic!("arr1: {name} is f16; use arr1o (owned, upcast)"),
        }
    }

    // Owned f32 — upcasts f16 (or copies f32) per call. The in-RAM-precision path (Gemma keeps weights f16 in RAM).
    pub fn arr2o(&self, name: &str) -> Array2<f32> {
        let (shape, arr) = self.get(name);
        Array2::from_shape_vec((shape[0], shape[1]), self.upcast(arr)).unwrap()
    }

    /// Fused matmul `a (seq, K) @ W (K, N) -> (seq, N)`. For an f16 weight this converts on the fly in a cache-friendly
    /// `ikj` loop (contiguous W-row and output-row access) — no f32 copy of the weight, which is the real cost of the
    /// in-RAM-precision path (a fresh upcast per matmul is ~GBs of alloc/write per forward). f32 falls back to ndarray.
    pub fn mm(&self, a: &Array2<f32>, name: &str) -> Array2<f32> {
        let (shape, arr) = self.get(name);
        let (k, n) = (shape[0], shape[1]);
        match arr {
            Arr::F32(v) => a.dot(&ArrayView2::from_shape((k, n), v).unwrap()),
            Arr::F16(v) => {
                // Upcast W one column-block at a time into a local f32 buffer, then SIMD-dot the block (ndarray) —
                // keeps the vectorised matmul while bounding upcast allocation to one block. Blocks are independent
                // (disjoint output columns), so they run in parallel; under the scoring loop's outer rayon this just
                // work-steals, but in single-stream generation it gives the per-forward matmul all the cores.
                let block = 512.min(n);
                let nblocks = n.div_ceil(block);
                let mut blocks: Vec<(usize, Array2<f32>)> = (0..nblocks)
                    .into_par_iter()
                    .map(|bi| {
                        let c0 = bi * block;
                        let bw = block.min(n - c0);
                        let mut buf = vec![0f32; k * bw];
                        for kk in 0..k {
                            let wrow = &v[kk * n + c0..kk * n + c0 + bw];
                            for (b, w) in buf[kk * bw..kk * bw + bw].iter_mut().zip(wrow) {
                                *b = w.to_f32();
                            }
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

    pub fn arr1o(&self, name: &str) -> Array1<f32> {
        let (_, arr) = self.get(name);
        Array1::from_vec(self.upcast(arr))
    }

    fn upcast(&self, arr: &Arr) -> Vec<f32> {
        match arr {
            Arr::F32(v) => v.clone(),
            Arr::F16(v) => v.iter().map(|h| h.to_f32()).collect(),
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
                }
            })
            .collect()
    }
}
