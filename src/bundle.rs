//! The flat weight bundle — the lm-sae -> fieldrun contract (`pylm/export_bundle.py`).
//!
//! A tiny JSON header (arch, config, and per-array dtype/shape/offset) plus one raw little-endian f32 blob. No zip, no
//! .npy parsing: read the header, slurp the blob, and view each array by offset. (Later: fp16/int8 blobs kept
//! low-precision in RAM and dequantised per matmul — the in-RAM-precision path. This first cut is f32.)

use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2};
use serde::Deserialize;

#[derive(Deserialize)]
struct ArrSpec {
    name: String,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
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
    arrays: HashMap<String, (Vec<usize>, Vec<f32>)>,   // parsed once at load; views handed out, zero-copy
}

pub const FORMAT: &str = "fieldrun-bundle";
pub const VERSION: u32 = 1;

impl Bundle {
    /// Load a fieldrun bundle: `<stem>.fieldrun.json` (manifest) + `<stem>.fieldrun.bin` (raw f32 blob). The blob is
    /// decoded to f32 once here; `arr2`/`arr1` then hand out zero-copy views, so the forward pass does no re-parsing.
    pub fn load(stem: &str) -> std::io::Result<Bundle> {
        let h: Header = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.fieldrun.json"))?)?;
        if h.format != FORMAT || h.version != VERSION {
            panic!("unsupported bundle: {} v{} (this fieldrun reads {FORMAT} v{VERSION})", h.format, h.version);
        }
        let data = std::fs::read(format!("{stem}.fieldrun.bin"))?;
        let mut arrays = HashMap::new();
        for a in h.arrays {
            let raw = &data[a.offset..a.offset + a.bytes];
            let v: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            arrays.insert(a.name, (a.shape, v));
        }
        Ok(Bundle { arch: h.arch, config: h.config, config_f: h.config_f, store: h.store, arrays })
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.arrays.get(name).unwrap_or_else(|| panic!("missing array {name}"))
    }

    pub fn arr2(&self, name: &str) -> ArrayView2<f32> {
        let (shape, v) = self.get(name);
        ArrayView2::from_shape((shape[0], shape[1]), v).unwrap()
    }

    pub fn arr1(&self, name: &str) -> ArrayView1<f32> {
        let (_, v) = self.get(name);
        ArrayView1::from(v.as_slice())
    }
}
