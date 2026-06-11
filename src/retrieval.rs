//! Tier A — retrieval. A faithful Rust port of pylm's `lm.py`: the decompiled LM as a handful of symbolic idioms
//! over a flat-file store, no neural net. Same arbitration, same idioms, same token space as the Python reference:
//!
//!   INDUCTION (in-context copy) — the longest local context (>= `min_accept` tokens) that recurs earlier in the
//!     sequence; predict the token that followed it. A pure slice scan, the keystone idiom.
//!   N-GRAM backoff — 4-gram -> trigram -> bigram -> unigram successor tables, a flat dict lookup.
//!   GRAMMAR (closed-class skeleton) — collapse content tokens to one `O` symbol, keep function-words/punct, look up
//!     the grammatical skeleton -> successor table. Fires below the lexical n-gram, above the unigram floor.
//!
//! Loaded from the same `store.json` the Python tier uses, so the two are bit-for-bit comparable (the faithfulness
//! gate: Rust predictions must equal Python `lm.py` predictions on the same store + ids).

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

fn default_min_induction() -> usize { 3 }
fn default_min_accept() -> usize { 2 }

#[derive(Deserialize)]
pub struct Store {
    #[serde(default)]
    quad: HashMap<String, Vec<i64>>,
    tri: HashMap<String, Vec<i64>>,
    bi: HashMap<String, Vec<i64>>,
    uni: Vec<i64>,
    #[serde(rename = "min_induction_match", default = "default_min_induction")]
    min_induction: usize,
    #[serde(rename = "min_induction_accept", default = "default_min_accept")]
    min_accept: usize,
    #[serde(default)]
    closed_ids: Vec<i64>,
    #[serde(default)]
    skel: HashMap<String, Vec<i64>>,
    #[serde(skip)]
    closed: HashSet<i64>,
}

/// Knobs for the retrieval-pruned-head candidate set (Phase 8b). Each field caps how many tokens a source contributes,
/// so the sweep can trade candidate-set size (∝ unembed compute) against coverage (= top-1 agreement with the full head).
#[derive(Clone, Copy)]
pub struct CandCfg {
    pub recent: usize, // last-N distinct context tokens (copy/induction window) — context-only, needs no store
    pub induction: usize, // tokens following recurrences of the context tail (in-context copy) — context-only
    pub quad: usize,   // top-j 4-gram successors (store)
    pub tri: usize,    // top-j trigram successors (store)
    pub bi: usize,     // top-j bigram successors (store)
    pub skel: usize,   // top-j grammar-skeleton successors (store)
    pub uni: usize,    // top-m unigram (globally frequent) floor (store)
    pub closed: bool,  // include the whole closed-class set (function words / punct) (store)
}

/// Context-only candidate tokens — recent distinct tokens + in-context induction copy. Needs NO store, so it works for
/// any model/tokenizer (the "the model copies from its own context" prior, which dominates real next-token coverage).
pub fn context_candidates(ctx: &[i64], recent: usize, induction: usize, out: &mut Vec<i64>) {
    let n = ctx.len();
    for &t in ctx.iter().rev().take(recent) {
        out.push(t);
    }
    // induction: for each span (longest first), find earlier recurrences of the tail and take the following token(s).
    if induction > 0 {
        let mut found = 0usize;
        'spans: for span in (1..=3.min(n.saturating_sub(1))).rev() {
            let tail = &ctx[n - span..];
            let mut i = n as isize - span as isize - 1;
            while i >= 0 {
                let iu = i as usize;
                if &ctx[iu..iu + span] == tail {
                    out.push(ctx[iu + span]);
                    found += 1;
                    if found >= induction {
                        break 'spans;
                    }
                }
                i -= 1;
            }
        }
    }
}

impl Store {
    pub fn load(path: &str) -> std::io::Result<Store> {
        let text = std::fs::read_to_string(path)?;
        let mut s: Store = serde_json::from_str(&text)?;
        s.closed = s.closed_ids.iter().copied().collect();
        Ok(s)
    }

    /// Retrieval-pruned-head candidate set (Phase 8b): the union of tokens the KB thinks can plausibly be next for
    /// `ctx` — context-only (recent + induction) plus store-driven (n-gram successors, grammar skeleton, closed class,
    /// unigram floor), each capped by `cfg`. Deduplicated, order-stable (highest-value sources first). The full-vocab
    /// unembed then scores only these (`Bundle::rowdot_f32_subset`); coverage of the full head's argmax = top-1 fidelity.
    pub fn candidates(&self, ctx: &[i64], cfg: &CandCfg) -> Vec<i64> {
        let mut out: Vec<i64> = Vec::new();
        context_candidates(ctx, cfg.recent, cfg.induction, &mut out);
        let n = ctx.len();
        let take = |v: &[i64], j: usize, out: &mut Vec<i64>| out.extend(v.iter().take(j).copied());
        if cfg.quad > 0 && n >= 3 {
            if let Some(v) = self.quad.get(&format!("{},{},{}", ctx[n - 3], ctx[n - 2], ctx[n - 1])) {
                take(v, cfg.quad, &mut out);
            }
        }
        if cfg.tri > 0 && n >= 2 {
            if let Some(v) = self.tri.get(&format!("{},{}", ctx[n - 2], ctx[n - 1])) {
                take(v, cfg.tri, &mut out);
            }
        }
        if cfg.bi > 0 && n >= 1 {
            if let Some(v) = self.bi.get(&ctx[n - 1].to_string()) {
                take(v, cfg.bi, &mut out);
            }
        }
        if cfg.skel > 0 && !self.skel.is_empty() {
            for sn in [3usize, 2usize] {
                if n >= sn {
                    let skel: Vec<String> = ctx[n - sn..]
                        .iter()
                        .map(|t| if self.closed.contains(t) { t.to_string() } else { "O".to_string() })
                        .collect();
                    if let Some(v) = self.skel.get(&format!("{}:{}", sn, skel.join("/"))) {
                        take(v, cfg.skel, &mut out);
                    }
                }
            }
        }
        if cfg.closed {
            out.extend(self.closed_ids.iter().copied());
        }
        if cfg.uni > 0 {
            take(&self.uni, cfg.uni, &mut out);
        }
        // dedup, order-stable (first occurrence wins → highest-value source kept).
        let mut seen = HashSet::with_capacity(out.len());
        out.retain(|&t| seen.insert(t));
        out
    }

    /// Next-token prediction for a token-id context: `(top-1 id, which idiom fired)` — mirrors `lm.py predict_explain`.
    pub fn predict(&self, ctx: &[i64]) -> (i64, String) {
        if let Some(hit) = self.induction(ctx) {
            return hit;
        }
        self.ngram(ctx)
    }

    fn induction(&self, ctx: &[i64]) -> Option<(i64, String)> {
        let mut span = self.min_induction;
        while span >= self.min_accept {
            if ctx.len() > span {
                let tail = &ctx[ctx.len() - span..];
                // search earlier occurrences from the most recent backwards (in-context copy of the latest match)
                let mut i = ctx.len() as isize - span as isize - 1;
                while i >= 0 {
                    let iu = i as usize;
                    if &ctx[iu..iu + span] == tail {
                        return Some((ctx[iu + span], format!("induction-{span}")));
                    }
                    i -= 1;
                }
            }
            span -= 1;
        }
        None
    }

    fn ngram(&self, ctx: &[i64]) -> (i64, String) {
        let n = ctx.len();
        if !self.quad.is_empty() && n >= 3 {
            let key = format!("{},{},{}", ctx[n - 3], ctx[n - 2], ctx[n - 1]);
            if let Some(v) = self.quad.get(&key) {
                if !v.is_empty() {
                    return (v[0], "quad".into());
                }
            }
        }
        if n >= 2 {
            let key = format!("{},{}", ctx[n - 2], ctx[n - 1]);
            if let Some(v) = self.tri.get(&key) {
                if !v.is_empty() {
                    return (v[0], "trigram".into());
                }
            }
        }
        if let Some(v) = self.bi.get(&ctx[n - 1].to_string()) {
            if !v.is_empty() {
                return (v[0], "bigram".into());
            }
        }
        if let Some(hit) = self.grammar(ctx) {
            return hit;
        }
        (self.uni[0], "unigram".into())
    }

    fn grammar(&self, ctx: &[i64]) -> Option<(i64, String)> {
        if self.skel.is_empty() {
            return None;
        }
        for n in [3usize, 2usize] {
            if ctx.len() >= n {
                let skel: Vec<String> = ctx[ctx.len() - n..]
                    .iter()
                    .map(|t| if self.closed.contains(t) { t.to_string() } else { "O".to_string() })
                    .collect();
                let key = format!("{}:{}", n, skel.join("/"));
                if let Some(v) = self.skel.get(&key) {
                    if !v.is_empty() {
                        return Some((v[0], format!("grammar-{n}")));
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_json(j: &str) -> Store {
        let mut s: Store = serde_json::from_str(j).unwrap();
        s.closed = s.closed_ids.iter().copied().collect();
        s
    }

    #[test]
    fn induction_copies_token_after_recurrence() {
        // "1 2 3 1 2" — the last "1 2" recurred at the start; predict what followed it there (3)
        let s = from_json(r#"{"tri":{},"bi":{},"uni":[0]}"#);
        let (p, tag) = s.predict(&[1, 2, 3, 1, 2]);
        assert_eq!(p, 3);
        assert!(tag.starts_with("induction"), "got {tag}");
    }

    #[test]
    fn ngram_bigram_fallback_when_no_induction() {
        let s = from_json(r#"{"tri":{},"bi":{"5":[9]},"uni":[0]}"#);
        let (p, tag) = s.predict(&[7, 5]);
        assert_eq!((p, tag.as_str()), (9, "bigram"));
    }

    #[test]
    fn unigram_is_the_floor() {
        let s = from_json(r#"{"tri":{},"bi":{},"uni":[42]}"#);
        let (p, tag) = s.predict(&[7]);
        assert_eq!((p, tag.as_str()), (42, "unigram"));
    }

    #[test]
    fn context_candidates_recent_and_induction() {
        // "1 2 3 1 2" — recent picks the last distinct tokens; induction copies the token after the recurring tail.
        let mut out = Vec::new();
        context_candidates(&[1, 2, 3, 1, 2], 3, 2, &mut out);
        // recent(3): last three are 2,1,3 (reverse order). induction: tail "1 2" recurred at start → next token 3.
        assert!(out.contains(&2) && out.contains(&1) && out.contains(&3));
        // induction with no recurrence contributes nothing beyond recent.
        let mut out2 = Vec::new();
        context_candidates(&[9, 8, 7], 1, 4, &mut out2);
        assert_eq!(out2[0], 7); // most-recent first
    }

    #[test]
    fn candidates_union_dedups_and_includes_ngrams() {
        let s = from_json(r#"{"quad":{},"tri":{},"bi":{"2":[9,2,5]},"uni":[42],"closed_ids":[7]}"#);
        let cfg = CandCfg { recent: 4, induction: 0, quad: 0, tri: 0, bi: 3, skel: 0, uni: 1, closed: true };
        let c = s.candidates(&[1, 2], &cfg);
        // recent {1,2} + bigram successors of "2" {9,2,5} + closed {7} + uni {42}, deduped (2 appears once).
        assert!(c.contains(&9) && c.contains(&5) && c.contains(&7) && c.contains(&42));
        assert_eq!(c.iter().filter(|&&t| t == 2).count(), 1, "deduped");
    }
}
