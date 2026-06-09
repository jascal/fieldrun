//! The flat weight bundle — the lm-sae -> fieldrun contract (`pylm/export_bundle.py`).
//!
//! A tiny JSON header (arch, config, and per-array dtype/shape/offset) plus one raw little-endian f32 blob. No zip, no
//! .npy parsing: read the header, slurp the blob, and view each array by offset. (Later: fp16/int8 blobs kept
//! low-precision in RAM and dequantised per matmul — the in-RAM-precision path. This first cut is f32.)

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
        (0..rows)
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
