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

    /// Greedy autoregressive generation of `n_new` tokens after `prompt`. The default recomputes the whole forward per
    /// token (O(context) work each step); kernels override with a KV-cache so each step processes only the new token.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let mut ctx = prompt.to_vec();
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let t = self.predict(&ctx);
            ctx.push(t);
            out.push(t);
        }
        out
    }
}
