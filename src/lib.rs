//! Library surface: the flat-bundle loader + the encoder-only BERT kernel, for downstream crates that
//! embed fieldrun as an inference library (e.g. sgiandubh's `bert_ffi` staticlib for the neural-expert
//! package). Deliberately minimal — the decoder archs, CLI, and explain machinery stay binary-only.
#[path = "bundle.rs"]
pub mod bundle;
#[path = "ternary.rs"]
mod ternary;
#[path = "bert.rs"]
pub mod bert;

pub use bert::Bert;
pub use bundle::Bundle;
