//! Corpus-level expert bucketing: an accumulator that ingests per-token irreducible atoms (from the
//! Density-Minimization descent, `explain::decompose_descent`) and clusters the corpus working set into hub-anchored
//! experts. Shared by the batch sweep (`--corpus-decompose`) and the incremental serve/REPL path (`--bucket`), so a
//! "much bigger corpus" is handled by streaming atoms in — the binding cost is the per-token explain forward; the atoms
//! themselves are tiny (~3 small ints each, ~72 bytes/token), so accumulating them is cheap.

use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

/// A scored source's identity: (kind: 0 = attention head, 1 = MLP neuron, layer, index-within-layer).
pub type Circuit = (u8, usize, usize);

/// Config for the incremental serve/REPL bucketing (`--bucket`): the descent's competitor count + the expert count.
#[derive(Clone, Copy)]
pub struct BucketOpts {
    pub k: usize,
    pub experts: usize,
}

/// Run the descent at one position and return its irreducible atom + the model's predicted token. `None` if the arch
/// does not expose the substrate (rope/Qwen only). The shared entry point for the batch, incremental, and DL paths.
pub fn atom_and_pred_at(lm: &dyn crate::model::Model, ctx: &[i64], k: usize) -> Option<(Vec<Circuit>, i64)> {
    let ex = lm.explain_decomp(ctx, k)?;
    let sub = ex.decomp.as_ref()?;
    let r = crate::explain::decompose_descent(sub);
    let atom = r.atom.iter().map(|&i| { let s = &sub.sources[i]; (s.kind, s.layer, s.idx) }).collect();
    Some((atom, ex.model_predicts))
}

/// Just the irreducible atom (the predicted token is dropped). Used by `--corpus-decompose` / `--query-decompose`.
pub fn atom_at(lm: &dyn crate::model::Model, ctx: &[i64], k: usize) -> Option<Vec<Circuit>> {
    atom_and_pred_at(lm, ctx, k).map(|(a, _)| a)
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
        Some(Cluster { e, n, distinct, total, freq, ranked, expert_of, size, tok_per_expert, span1, spanned, span_sum, cost_sum })
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

    /// Emit a Soufflé-compatible Datalog LOOKUP/SELECTION model derived from the partition. The expert partition is
    /// RELATIONS (`expert`/`anchor`); routing (`selected(sig,e)`) and decision (`predict(sig,tok)`) are LOOKUP tables
    /// over a context signature (the previous token id), compiled from the corpus; rules apply the lookup (`decode`)
    /// and check it reproduces the model's decode (`hit`). The header reports per-expert decision entropy
    /// `H(pred|expert)` — ≈0 marks a lookup-exact (retrievable) expert, >0 the computed residue (the forge tax). A
    /// corpus-derived lookup model (generalizes by signature match), NOT the dense forward pass (that is logic_whole.rs
    /// / LO3a — exact but non-compact). `sig[i]`/`pred[i]` align with the i-th ingested atom (prev token + decode).
    pub fn emit_datalog(&self, experts: usize, sig: &[i64], pred: &[i64], test_frac: f32) -> String {
        let mut out = String::new();
        let c = match self.cluster(experts) {
            None => {
                out.push_str("// (no atoms accumulated — nothing to emit)\n");
                return out;
            }
            Some(c) => c,
        };
        let m = self.atoms.len().min(sig.len()).min(pred.len());
        // per-token route (top-1 expert), aligned with ingest order.
        let routes: Vec<usize> = self
            .atoms
            .iter()
            .map(|a| if a.is_empty() { c.e } else { c.expert_of[a.iter().max_by_key(|x| c.freq[*x]).unwrap()] })
            .collect();
        // TRAIN/TEST split — held-out generalization vs in-sample memorization. The lookup is compiled from the first
        // (1−test_frac) of the corpus and evaluated on the held-out tail; test_frac<=0 ⇒ pure in-sample (train==all).
        let test_frac = test_frac.clamp(0.0, 0.9);
        let train_n = if test_frac <= 0.0 { m } else { (((1.0 - test_frac) * m as f32).round() as usize).clamp(1, m) };
        let test_n = m - train_n;
        // compile the lookup tables from TRAIN positions only: signature → {expert counts}, signature → {pred counts}.
        let mut sig_e: BTreeMap<i64, HashMap<usize, usize>> = BTreeMap::new();
        let mut sig_p: BTreeMap<i64, HashMap<i64, usize>> = BTreeMap::new();
        for i in 0..train_n {
            *sig_e.entry(sig[i]).or_default().entry(routes[i]).or_default() += 1;
            *sig_p.entry(sig[i]).or_default().entry(pred[i]).or_default() += 1;
        }
        let plur_e: BTreeMap<i64, usize> = sig_e.iter().map(|(&s, mm)| (s, *mm.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).unwrap().0)).collect();
        let plur_p: BTreeMap<i64, i64> = sig_p.iter().map(|(&s, mm)| (s, *mm.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).unwrap().0)).collect();
        // IN-SAMPLE accuracy (train → train) vs HELD-OUT (train → test, with coverage of unseen signatures).
        let (mut in_hit, mut in_rhit) = (0usize, 0usize);
        for i in 0..train_n {
            if plur_p.get(&sig[i]) == Some(&pred[i]) { in_hit += 1; }
            if plur_e.get(&sig[i]) == Some(&routes[i]) { in_rhit += 1; }
        }
        let (mut ho_hit, mut ho_rhit, mut covered) = (0usize, 0usize, 0usize);
        for i in train_n..m {
            if plur_p.contains_key(&sig[i]) { covered += 1; } // signature seen in train ⇒ a lookup exists
            if plur_p.get(&sig[i]) == Some(&pred[i]) { ho_hit += 1; }
            if plur_e.get(&sig[i]) == Some(&routes[i]) { ho_rhit += 1; }
        }
        // per-expert decision entropy H(pred|expert): ~0 = lookup-exact, >0 = computed residue.
        let mut e_pred: Vec<HashMap<i64, usize>> = vec![HashMap::new(); c.e + 1];
        for i in 0..m {
            *e_pred[routes[i]].entry(pred[i]).or_default() += 1;
        }
        let entropy = |mm: &HashMap<i64, usize>| -> f64 {
            let t: usize = mm.values().sum();
            if t == 0 {
                return 0.0;
            }
            mm.values().map(|&x| { let p = x as f64 / t as f64; -p * p.log2() }).sum()
        };
        let kind = |k: u8| if k == 0 { "head" } else { "neuron" };
        let pc = |x: usize, d: usize| if d > 0 { 100.0 * x as f32 / d as f32 } else { 0.0 };
        // ---- header ------------------------------------------------------------------------------------------
        let _ = writeln!(out, "// fieldrun EXPERTS-DL — a Datalog LOOKUP/SELECTION model from the density-minimization partition.");
        let _ = writeln!(out, "// The expert partition is RELATIONS; routing selected(sig,e) and decision predict(sig,tok) are LOOKUP");
        let _ = writeln!(out, "// tables over a context signature (the previous token id), compiled from the corpus — a corpus-derived");
        let _ = writeln!(out, "// lookup model (generalizes by signature match), NOT the dense forward pass (logic_whole.rs/LO3a, exact");
        let _ = writeln!(out, "// but non-compact). Run: souffle <this>.dl -D-   (outputs: selected, predict, decode, routed, hit_train, hit_test).");
        let _ = writeln!(out, "//");
        let _ = writeln!(out, "// N tokens={}  |C| circuits={}  E experts={} (+residual idx {})", c.n, c.distinct, c.e, c.e);
        let _ = writeln!(out, "// split: train={train_n}  test={test_n} (test_frac={test_frac:.2})  train signatures={}", sig_e.len());
        let _ = writeln!(out, "// IN-SAMPLE (train): predict==decode {:.0}%   selected==route {:.0}%   [optimistic — memorizes singleton signatures]", pc(in_hit, train_n), pc(in_rhit, train_n));
        if test_n > 0 {
            let _ = writeln!(out, "// HELD-OUT (test):  predict==decode {:.0}%   selected==route {:.0}%   (coverage {:.0}% of test sigs seen in train; accuracy among covered {:.0}%)", pc(ho_hit, test_n), pc(ho_rhit, test_n), pc(covered, test_n), pc(ho_hit, covered));
        } else {
            let _ = writeln!(out, "// HELD-OUT: none (test_frac=0 → in-sample only; pass --dl-test-frac 0.2 for generalization)");
        }
        let _ = writeln!(out, "// per-expert decision entropy H(pred|expert) — ~0 bits = lookup-exact (retrievable); >0 = computed residue:");
        for e in 0..c.e {
            let (ak, al, ai) = c.ranked[e].0;
            let _ = writeln!(out, "//   e{e:<2} anchor {} L{al} #{ai}: H={:.2} bits over {} tokens", kind(ak), entropy(&e_pred[e]), c.tok_per_expert[e]);
        }
        let _ = writeln!(out, "//   residual(e{}): H={:.2} bits over {} tokens", c.e, entropy(&e_pred[c.e]), c.tok_per_expert[c.e]);
        let _ = writeln!(out);
        // ---- partition relations ------------------------------------------------------------------------------
        let _ = writeln!(out, ".decl expert(e:number, kind:symbol, layer:number, idx:number)");
        let mut by_expert: Vec<Vec<Circuit>> = vec![Vec::new(); c.e + 1];
        for (&id, &ex) in &c.expert_of {
            by_expert[ex].push(id);
        }
        for v in by_expert.iter_mut() {
            v.sort();
        }
        for (ex, v) in by_expert.iter().enumerate() {
            for &(k, l, i) in v {
                let _ = writeln!(out, "expert({ex},\"{}\",{l},{i}).", kind(k));
            }
        }
        let _ = writeln!(out, ".decl anchor(e:number, kind:symbol, layer:number, idx:number)");
        for e in 0..c.e {
            let (k, l, i) = c.ranked[e].0;
            let _ = writeln!(out, "anchor({e},\"{}\",{l},{i}).", kind(k));
        }
        // ---- corpus observations ------------------------------------------------------------------------------
        let _ = writeln!(out, ".decl obs(pos:number, sig:number, route:number, pred:number, split:number)   // split: 0=train 1=test");
        for i in 0..m {
            let split = if i < train_n { 0 } else { 1 };
            let _ = writeln!(out, "obs({i},{},{},{},{split}).", sig[i], routes[i], pred[i]);
        }
        // ---- compiled lookup / selection ----------------------------------------------------------------------
        let _ = writeln!(out, ".decl selected(sig:number, e:number)    // SELECTION: the routed expert for a signature");
        for (&s, &e) in &plur_e {
            let _ = writeln!(out, "selected({s},{e}).");
        }
        let _ = writeln!(out, ".decl predict(sig:number, tok:number)   // LOOKUP: the decided token for a signature");
        for (&s, &p) in &plur_p {
            let _ = writeln!(out, "predict({s},{p}).");
        }
        // ---- rules: apply the lookup + check it reproduces the model's decode ---------------------------------
        let _ = writeln!(out, ".decl decode(pos:number, tok:number)");
        let _ = writeln!(out, "decode(P,Tok) :- obs(P,Sig,_,_,_), predict(Sig,Tok).");
        let _ = writeln!(out, ".decl routed(pos:number, e:number)");
        let _ = writeln!(out, "routed(P,E) :- obs(P,Sig,_,_,_), selected(Sig,E).");
        let _ = writeln!(out, ".decl hit_train(pos:number)   // train-derived lookup reproduces the decode (IN-SAMPLE)");
        let _ = writeln!(out, "hit_train(P) :- obs(P,_,_,D,0), decode(P,T), T=D.");
        let _ = writeln!(out, ".decl hit_test(pos:number)    // ... on the HELD-OUT tail (generalization)");
        let _ = writeln!(out, "hit_test(P) :- obs(P,_,_,D,1), decode(P,T), T=D.");
        let _ = writeln!(out, ".output selected\n.output predict\n.output decode\n.output routed\n.output hit_train\n.output hit_test");
        out
    }
}

/// The computed clustering (internal): every circuit assigned to an expert (0..e) or the residual bucket (index e).
struct Cluster {
    e: usize,
    n: usize,
    distinct: usize,
    total: usize,
    freq: HashMap<Circuit, usize>,
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

    #[test]
    fn emit_datalog_has_relations_rules_and_stats() {
        let mut b = CorpusBuckets::new();
        let n0 = (1u8, 23, 2539);
        for t in 0..6 {
            b.ingest(vec![n0, (1u8, t, t)]); // a shared hub + a per-token private neuron
        }
        let sig: Vec<i64> = (0..6).map(|t| (t % 2) as i64).collect();
        let pred: Vec<i64> = (0..6).map(|_| 100i64).collect(); // constant decode ⇒ lookup is exact (100%)
        let dl = b.emit_datalog(2, &sig, &pred, 0.0); // test_frac=0 ⇒ in-sample only
        for needle in [
            ".decl expert(e:number",
            ".decl selected(sig:number",
            ".decl predict(sig:number",
            "decode(P,Tok) :- obs(P,Sig,_,_,_), predict(Sig,Tok).",
            "hit_train(P) :- obs(P,_,_,D,0), decode(P,T), T=D.",
            "hit_test(P) :- obs(P,_,_,D,1), decode(P,T), T=D.",
            ".output hit_test",
        ] {
            assert!(dl.contains(needle), "emitted Datalog missing: {needle}");
        }
        // pred is constant ⇒ predict(sig)==decode everywhere ⇒ the header reports 100% in-sample accuracy.
        assert!(dl.contains("IN-SAMPLE (train): predict==decode 100%"), "in-sample accuracy stat wrong:\n{dl}");
    }

    #[test]
    fn emit_datalog_holds_out_unseen_signatures() {
        // train sigs {0,1} all decode to 100; the test tail has an UNSEEN sig 2 ⇒ not covered ⇒ held-out misses it.
        let mut b = CorpusBuckets::new();
        for _ in 0..10 { b.ingest(vec![(1u8, 23, 1)]); }
        let sig: Vec<i64> = vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 2]; // last two positions are the unseen-signature test tail
        let pred: Vec<i64> = vec![100; 10];
        let dl = b.emit_datalog(1, &sig, &pred, 0.2); // test_frac 0.2 ⇒ train=8, test=2 (both sig 2, unseen)
        assert!(dl.contains("split: train=8  test=2"), "split header wrong:\n{dl}");
        // sig 2 never appears in train ⇒ 0% coverage ⇒ 0% held-out accuracy despite 100% in-sample.
        assert!(dl.contains("HELD-OUT (test):  predict==decode 0%"), "held-out should miss unseen sigs:\n{dl}");
        assert!(dl.contains("coverage 0%"), "coverage should be 0 for unseen sigs:\n{dl}");
    }
}
