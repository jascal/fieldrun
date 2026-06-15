//! Corpus-level expert bucketing: an accumulator that ingests per-token irreducible atoms (from the
//! Density-Minimization descent, `explain::decompose_descent`) and clusters the corpus working set into hub-anchored
//! experts. Shared by the batch sweep (`--corpus-decompose`) and the incremental serve/REPL path (`--bucket`), so a
//! "much bigger corpus" is handled by streaming atoms in — the binding cost is the per-token explain forward; the atoms
//! themselves are tiny (~3 small ints each, ~72 bytes/token), so accumulating them is cheap.

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// A scored source's identity: (kind: 0 = attention head, 1 = MLP neuron, layer, index-within-layer).
pub type Circuit = (u8, usize, usize);

/// Run the descent at one position and return its irreducible atom as circuit identities. `None` if the arch does not
/// expose the substrate (rope/Qwen only). The single shared entry point for both the batch and the incremental paths.
pub fn atom_at(lm: &dyn crate::model::Model, ctx: &[i64], k: usize) -> Option<Vec<Circuit>> {
    let ex = lm.explain_decomp(ctx, k)?;
    let sub = ex.decomp.as_ref()?;
    let r = crate::explain::decompose_descent(sub);
    Some(r.atom.iter().map(|&i| { let s = &sub.sources[i]; (s.kind, s.layer, s.idx) }).collect())
}

/// Online accumulator of per-token atoms over a corpus. `render` clusters the current contents into `experts`
/// hub-anchored buckets and returns the report body — recomputed from scratch each call (cheap; atoms are tiny), so the
/// clustering always reflects everything ingested so far. Memory grows ~72 bytes/token with the corpus.
#[derive(Default)]
pub struct CorpusBuckets {
    atoms: Vec<Vec<Circuit>>,
}

impl CorpusBuckets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, atom: Vec<Circuit>) {
        self.atoms.push(atom);
    }

    pub fn n_tokens(&self) -> usize {
        self.atoms.len()
    }

    /// Compute the clustering: partition the corpus working set into `experts` hub-anchored buckets. The top-`experts`
    /// circuits by frequency are the expert ANCHORS (the recurring hubs); each other circuit joins the anchor-expert it
    /// co-fires with most (co-fire = same atom); circuits that never co-fire with a hub fall into the residual bucket
    /// (index `e`). Returns the FULL assignment (`expert_of`) + per-expert sizes/token-routes + routing stats — the
    /// object both `render` (summary) and `partition` (the concrete expert→circuit sets) build from. `None` if empty.
    fn cluster(&self, experts: usize) -> Option<Cluster> {
        let atoms = &self.atoms;
        let n = atoms.len();
        if n == 0 {
            return None;
        }
        let mut freq: HashMap<Circuit, usize> = HashMap::new();
        for a in atoms {
            for &id in a {
                *freq.entry(id).or_default() += 1;
            }
        }
        let distinct = freq.len();
        let mut ranked: Vec<(Circuit, usize)> = freq.iter().map(|(&id, &f)| (id, f)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let e = experts.min(ranked.len());
        let mut expert_of: HashMap<Circuit, usize> = HashMap::new();
        for (k, &(id, _)) in ranked.iter().take(e).enumerate() {
            expert_of.insert(id, k);
        }
        let mut cooc: HashMap<Circuit, HashMap<usize, usize>> = HashMap::new();
        for a in atoms {
            let present: Vec<usize> = a.iter().filter_map(|id| expert_of.get(id).copied()).collect();
            if present.is_empty() {
                continue;
            }
            for &id in a {
                if expert_of.contains_key(&id) {
                    continue;
                }
                let m = cooc.entry(id).or_default();
                for &ex in &present {
                    *m.entry(ex).or_default() += 1;
                }
            }
        }
        let residual = e;
        for (&id, m) in &cooc {
            let best = m.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))).map(|(&ex, _)| ex).unwrap_or(residual);
            expert_of.entry(id).or_insert(best);
        }
        for (&id, _) in &freq {
            expert_of.entry(id).or_insert(residual);
        }
        let mut size = vec![0usize; e + 1];
        for (_, &ex) in &expert_of {
            size[ex] += 1;
        }
        let total: usize = atoms.iter().map(|a| a.len()).sum();
        let (mut span1, mut spanned, mut span_sum, mut cost_sum) = (0usize, 0usize, 0usize, 0usize);
        let mut tok_per_expert = vec![0usize; e + 1];
        for a in atoms {
            if a.is_empty() {
                continue;
            }
            let mut touched: HashSet<usize> = HashSet::new();
            for &id in a {
                touched.insert(expert_of[&id]);
            }
            span_sum += touched.len();
            spanned += 1;
            if touched.len() == 1 {
                span1 += 1;
            }
            cost_sum += touched.iter().map(|&ex| size[ex]).sum::<usize>();
            let route = a.iter().max_by_key(|id| freq[*id]).map(|id| expert_of[id]).unwrap();
            tok_per_expert[route] += 1;
        }
        Some(Cluster { e, n, distinct, total, ranked, expert_of, size, tok_per_expert, span1, spanned, span_sum, cost_sum })
    }

    /// Render the clustering as a human-readable report body (summary stats + the per-expert anchors/sizes/routes).
    pub fn render(&self, experts: usize) -> String {
        let mut out = String::new();
        let c = match self.cluster(experts) {
            Some(c) => c,
            None => {
                let _ = writeln!(out, "  (no atoms accumulated yet)");
                return out;
            }
        };
        let pct = |x: f32| 100.0 * x;
        let _ = writeln!(out, "  N tokens                  {}", c.n);
        let _ = writeln!(out, "  Σ|A_t| total firings      {}   (reuse 1 − |C|/Σ = {:.0}%)", c.total, pct(c.reuse()));
        let _ = writeln!(out, "  |C| distinct circuits     {}", c.distinct);
        let _ = writeln!(out, "  E experts                 {}   (+ residual: {} circuits never co-firing with a hub)", c.e, c.size[c.e]);
        let _ = writeln!(out, "  span 1 (top-1 routable)   {:.0}%   ← the deciding atom fits in ONE expert", pct(c.span1_frac()));
        let _ = writeln!(out, "  mean experts / token      {:.2}", c.mean_span());
        let _ = writeln!(out, "  active circuits / token   {:.1} of {}   → {:.0}% fewer computed (static oracle-router proxy)", c.mean_active(), c.distinct, pct(c.active_saving()));
        for k in 0..c.e {
            let (ak, al, ai) = c.ranked[k].0;
            let kind = if ak == 0 { "head" } else { "neuron" };
            let _ = writeln!(out, "    e{k:<2} anchor {kind:<6} L{al:<2} #{ai:<6}  {:>4} circuits  {:>6} tokens ({:.0}%)", c.size[k], c.tok_per_expert[k], 100.0 * c.tok_per_expert[k] as f32 / c.n.max(1) as f32);
        }
        out
    }

    /// The concrete expert PARTITION as a serializable object: each expert's anchor + the full list of circuits assigned
    /// to it (the residual bucket last) + per-token routing stats. This is the build artifact — the actual expert→circuit
    /// sets a router/weight-chunk pager would consume — not just the summary. Empty (no experts) if nothing ingested.
    pub fn partition(&self, experts: usize) -> PartitionJson {
        let c = match self.cluster(experts) {
            Some(c) => c,
            None => return PartitionJson::default(),
        };
        // group circuits by their assigned expert (0..e plus residual = e).
        let mut by_expert: Vec<Vec<Circuit>> = vec![Vec::new(); c.e + 1];
        for (&id, &ex) in &c.expert_of {
            by_expert[ex].push(id);
        }
        for v in by_expert.iter_mut() {
            v.sort();
        }
        let mut buckets = Vec::with_capacity(c.e + 1);
        for ex in 0..=c.e {
            let anchor = if ex < c.e { Some(cj(c.ranked[ex].0)) } else { None }; // residual has no anchor
            buckets.push(ExpertJson {
                id: ex,
                residual: ex == c.e,
                anchor,
                size: c.size[ex],
                tokens: c.tok_per_expert[ex],
                circuits: by_expert[ex].iter().map(|&x| cj(x)).collect(),
            });
        }
        PartitionJson {
            n_tokens: c.n,
            distinct_circuits: c.distinct,
            experts: c.e,
            span1_frac: c.span1_frac(),
            mean_experts_per_token: c.mean_span(),
            mean_active_circuits: c.mean_active(),
            reuse: c.reuse(),
            buckets,
        }
    }
}

/// The computed clustering (internal): every circuit assigned to an expert (0..e) or the residual bucket (index e).
struct Cluster {
    e: usize,
    n: usize,
    distinct: usize,
    total: usize,
    ranked: Vec<(Circuit, usize)>,
    expert_of: HashMap<Circuit, usize>,
    size: Vec<usize>,
    tok_per_expert: Vec<usize>,
    span1: usize,
    spanned: usize,
    span_sum: usize,
    cost_sum: usize,
}

impl Cluster {
    fn reuse(&self) -> f32 {
        if self.total > 0 { 1.0 - self.distinct as f32 / self.total as f32 } else { 0.0 }
    }
    fn span1_frac(&self) -> f32 {
        if self.spanned > 0 { self.span1 as f32 / self.spanned as f32 } else { 0.0 }
    }
    fn mean_span(&self) -> f32 {
        if self.spanned > 0 { self.span_sum as f32 / self.spanned as f32 } else { 0.0 }
    }
    fn mean_active(&self) -> f32 {
        if self.spanned > 0 { self.cost_sum as f32 / self.spanned as f32 } else { 0.0 }
    }
    fn active_saving(&self) -> f32 {
        if self.distinct > 0 { 1.0 - self.mean_active() / self.distinct as f32 } else { 0.0 }
    }
}

/// A circuit identity in serializable form.
#[derive(Serialize, Clone)]
pub struct CircuitJson {
    pub kind: &'static str,
    pub layer: usize,
    pub idx: usize,
}

fn cj(c: Circuit) -> CircuitJson {
    CircuitJson { kind: if c.0 == 0 { "head" } else { "neuron" }, layer: c.1, idx: c.2 }
}

/// One expert in the emitted partition: its anchor hub + the full circuit set assigned to it.
#[derive(Serialize)]
pub struct ExpertJson {
    pub id: usize,
    pub residual: bool,
    pub anchor: Option<CircuitJson>,
    pub size: usize,
    pub tokens: usize,
    pub circuits: Vec<CircuitJson>,
}

/// The full emitted partition: the concrete expert→circuit assignment + routing stats (serializes to JSON).
#[derive(Serialize, Default)]
pub struct PartitionJson {
    pub n_tokens: usize,
    pub distinct_circuits: usize,
    pub experts: usize,
    pub span1_frac: f32,
    pub mean_experts_per_token: f32,
    pub mean_active_circuits: f32,
    pub reuse: f32,
    pub buckets: Vec<ExpertJson>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cluster_and_route() {
        let mut b = CorpusBuckets::new();
        // two recurring hub circuits (h0, n0) plus per-token private circuits; atoms co-fire h0+n0 often.
        let h0 = (0u8, 23, 1);
        let n0 = (1u8, 23, 2539);
        for t in 0..20 {
            b.ingest(vec![h0, n0, (1u8, t, t)]); // each token: the two hubs + one private neuron
        }
        assert_eq!(b.n_tokens(), 20);
        let rep = b.render(2);
        // 2 experts anchored on the two hubs; every atom touches both hubs ⇒ NOT top-1 routable (span 2).
        assert!(rep.contains("E experts                 2"));
        assert!(rep.contains("|C| distinct circuits     22")); // 2 hubs + 20 private
    }

    #[test]
    fn empty_render_is_safe() {
        let b = CorpusBuckets::new();
        assert!(b.render(8).contains("no atoms"));
    }
}
