//! Margin-gated retrieval-pruned output head — the serve-path form of Phase 8b (`--pruned-head`).
//!
//! Per decode step the KB proposes a small candidate set (n-gram / induction / grammar / closed-class, ~540 tokens
//! at the default config); the unembed scores ONLY those rows (`rowdot_f32_subset`, a gather instead of the full
//! (vocab, d) stream). The pruned argmax is accepted iff the **normalized margin** between the in-set top-2,
//! `(L_t − L_v)/‖U_t − U_v‖` — the exact Euclidean distance from the final residual to the t–v bisector facet of the
//! unembedding's power diagram (FINDINGS §5b) — clears `threshold`; below it, the caller falls back to the full head.
//! Rationale (FINDINGS): covered tokens sit deep in their cell (RETRIEVED ≈ 2.2–2.8, SELECTED ≈ 1.3–1.5) while
//! COMPOSED tokens — the ones whose argmax the candidate set misses — sit thin (≈ 1.0–1.2), so a thin in-set margin
//! flags exactly the steps where the pruned head can't be trusted. The gate is a calibrated accuracy-vs-speed knob
//! (like `--route-frac` / `--kv-int8`), opt-in and off by default; `--gate-check` measures its top-1 agreement
//! against the full head (the faithfulness-gate form for a deliberately lossy mode).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::retrieval::{CandCfg, Store};

pub struct HeadGate {
    store: Store,
    cfg: CandCfg,
    threshold: f32,
    accepted: AtomicU64,
    fallback: AtomicU64,
}

/// Indices of the best and runner-up FINITE logits (subset scores use −∞ for out-of-range candidate ids — never
/// pick or margin against those). `None` if fewer than two finite entries (no margin can be formed).
pub fn top2(lg: &[f32]) -> Option<(usize, usize)> {
    let mut i1: Option<usize> = None;
    let mut i2: Option<usize> = None;
    for (i, &v) in lg.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if i1.is_none_or(|a| v > lg[a]) {
            i2 = i1;
            i1 = Some(i);
        } else if i2.is_none_or(|b| v > lg[b]) {
            i2 = Some(i);
        }
    }
    Some((i1?, i2?))
}

/// `(L_t − L_v)/‖U_t − U_v‖` — the exact distance from the final residual to the t–v bisector facet (same quantity
/// as `--probe-facet`, which validated the geometry). Degenerate rows (‖U_t − U_v‖² ≤ 1e-4, no facet) → −∞ so the
/// gate always falls back on them.
pub fn normalized_margin(lt: f32, lv: f32, ut: &[f32], uv: &[f32]) -> f32 {
    let d2: f32 = ut.iter().zip(uv).map(|(a, b)| (a - b) * (a - b)).sum();
    if d2 <= 1e-4 {
        return f32::NEG_INFINITY;
    }
    (lt - lv) / d2.sqrt()
}

impl HeadGate {
    pub fn new(store: Store, threshold: f32) -> HeadGate {
        // The "generous~512" config from the --prune-head sweep (~540 candidates ⇒ ~85% argmax coverage on natural
        // text, FINDINGS §1) — the operating point the margin gate was characterized at.
        let cfg = CandCfg { recent: 128, induction: 4, quad: 16, tri: 16, bi: 16, skel: 16, uni: 256, closed: true };
        HeadGate { store, cfg, threshold, accepted: AtomicU64::new(0), fallback: AtomicU64::new(0) }
    }

    /// Try the pruned head for one decode step. `score(cands)` returns subset logits aligned to `cands`
    /// (`Bundle::rowdot_f32_subset`); `row(v)` returns unembedding row v (`Bundle::weight_row`). `Some(token)` =
    /// the in-set argmax cleared the margin gate; `None` = run the full head. Counters track the accept rate.
    pub fn try_pruned(&self, ctx: &[i64], score: &dyn Fn(&[i64]) -> Vec<f32>, row: &dyn Fn(usize) -> Vec<f32>) -> Option<i64> {
        let cands = self.store.candidates(ctx, &self.cfg);
        let picked = (|| {
            if cands.len() < 2 {
                return None;
            }
            let lg = score(&cands);
            let (i, j) = top2(&lg)?;
            let (t, v) = (cands[i] as usize, cands[j] as usize);
            if normalized_margin(lg[i], lg[j], &row(t), &row(v)) >= self.threshold {
                Some(cands[i])
            } else {
                None
            }
        })();
        match picked {
            Some(_) => self.accepted.fetch_add(1, Ordering::Relaxed),
            None => self.fallback.fetch_add(1, Ordering::Relaxed),
        };
        picked
    }

    /// (accepted, fallback) decode-step counts since construction.
    pub fn stats(&self) -> (u64, u64) {
        (self.accepted.load(Ordering::Relaxed), self.fallback.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(j: &str) -> Store {
        serde_json::from_str(j).unwrap()
    }

    #[test]
    fn top2_picks_best_and_runner_up_skipping_non_finite() {
        assert_eq!(top2(&[1.0, 5.0, 3.0]), Some((1, 2)));
        assert_eq!(top2(&[f32::NEG_INFINITY, 2.0, 4.0]), Some((2, 1)));
        assert_eq!(top2(&[f32::NEG_INFINITY, 2.0]), None); // one finite entry — no margin
        assert_eq!(top2(&[]), None);
        // ties keep the first occurrence as top-1 (strict > to displace)
        assert_eq!(top2(&[7.0, 7.0]), Some((0, 1)));
    }

    #[test]
    fn normalized_margin_is_facet_distance() {
        // U_t=(1,0), U_v=(0,1): ‖U_t−U_v‖=√2; logit gap 2 → margin 2/√2 = √2
        let m = normalized_margin(3.0, 1.0, &[1.0, 0.0], &[0.0, 1.0]);
        assert!((m - 2.0f32 / 2.0f32.sqrt()).abs() < 1e-6);
        // identical rows: no facet between t and v → −∞ (always falls back)
        assert_eq!(normalized_margin(3.0, 1.0, &[1.0, 0.0], &[1.0, 0.0]), f32::NEG_INFINITY);
    }

    #[test]
    fn gate_accepts_wide_margin_and_falls_back_thin() {
        // bigram store: after token 5 the candidates are {9, 7} (+ recent context tokens).
        let g = HeadGate::new(store(r#"{"tri":{},"bi":{"5":[9,7]},"uni":[0]}"#), 1.0);
        // orthogonal unit rows → margin == logit gap / √2
        let row = |v: usize| {
            let mut r = vec![0.0f32; 16];
            r[v] = 1.0;
            r
        };
        // wide: token 9 beats everything by 10 → accepted, and the pick is 9
        let wide = |c: &[i64]| c.iter().map(|&t| if t == 9 { 10.0 } else { 0.0 }).collect::<Vec<f32>>();
        assert_eq!(g.try_pruned(&[1, 5], &wide, &row), Some(9));
        // thin: top-2 within 0.1 → margin 0.1/√2 < 1.0 → fallback
        let thin = |c: &[i64]| c.iter().map(|&t| if t == 9 { 0.1 } else { 0.0 }).collect::<Vec<f32>>();
        assert_eq!(g.try_pruned(&[1, 5], &thin, &row), None);
        assert_eq!(g.stats(), (1, 1));
    }

    #[test]
    fn gate_falls_back_when_no_margin_can_be_formed() {
        // empty context + empty store at a tiny config → fewer than 2 candidates
        let g = HeadGate::new(store(r#"{"tri":{},"bi":{},"uni":[0]}"#), 1.0);
        assert_eq!(g.try_pruned(&[], &|_| vec![], &|_| vec![]), None);
        // out-of-range candidates score −∞ (rowdot_f32_subset contract): only one finite → fallback, not a pick
        let g2 = HeadGate::new(store(r#"{"tri":{},"bi":{"5":[9,7]},"uni":[0]}"#), 0.0);
        let oob = |c: &[i64]| c.iter().map(|&t| if t == 9 { 1.0 } else { f32::NEG_INFINITY }).collect::<Vec<f32>>();
        assert_eq!(g2.try_pruned(&[1, 5], &oob, &|_| vec![0.0; 4]), None);
    }
}
