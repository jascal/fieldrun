//! The runtime kernel interface — one decompiled-LLM forward pass, dispatched by `arch` in the bundle manifest.
//! Every kernel (GPT-2, RoPE family, Gemma-2) mirrors its pylm numpy reference and is held behind this trait so the
//! scoring loop is architecture-agnostic. `Sync` so rayon can fan independent forwards across cores.

pub trait Model: Sync {
    /// Top-1 next-token id for a context.
    fn predict(&self, ids: &[i64]) -> i64;

    /// Explain the prediction (composition-side circuits + features). Default None; GPT-2 implements it.
    fn explain(&self, _ids: &[i64]) -> Option<crate::explain::Explanation> {
        None
    }

    /// Greedy generation up to `max_tokens`, stopping early at any `eos` id (the stop token is NOT included in the
    /// output). `emit(id)` is called for each generated token *as it is produced* (for streaming / a live chat);
    /// returning `false` (e.g. the HTTP client disconnected) stops generation. Default: naive — recompute the whole
    /// context every token (O(n)/token); kernels with a KV-cache override this for O(1)/token decode.
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let mut ctx = prompt.to_vec();
        let mut out = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let t = self.predict(&ctx);
            if eos.contains(&t) {
                break;
            }
            out.push(t);
            if !emit(t) {
                break;
            }
            ctx.push(t);
        }
        out
    }

    /// Fixed-length greedy generation (exactly `n_new` tokens, no early stop) — used by the CLI `--generate`
    /// KV-cache-vs-naive benchmark. KV-cache kernels override for speed; otherwise delegates to `generate_stream`.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        self.generate_stream(prompt, n_new, &[], &mut |_| true)
    }
}
