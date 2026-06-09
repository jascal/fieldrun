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

impl Store {
    pub fn load(path: &str) -> std::io::Result<Store> {
        let text = std::fs::read_to_string(path)?;
        let mut s: Store = serde_json::from_str(&text)?;
        s.closed = s.closed_ids.iter().copied().collect();
        Ok(s)
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
}
