//! The runtime kernel interface — one decompiled-LLM forward pass, dispatched by `arch` in the bundle manifest.
//! Every kernel (GPT-2, RoPE family, Gemma-2) mirrors its pylm numpy reference and is held behind this trait so the
//! scoring loop is architecture-agnostic. `Sync` so rayon can fan independent forwards across cores.

pub trait Model: Sync {
    /// Top-1 next-token id for a context.
    fn predict(&self, ids: &[i64]) -> i64;
}
