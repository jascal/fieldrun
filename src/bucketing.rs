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

    /// The full circuit→expert assignment `(e, expert_of)` (e = residual-bucket index). For the contrib-over-expert emit:
    /// a decision's scored circuits are grouped by their corpus-expert to sum per-expert contributions.
    pub fn expert_map(&self, experts: usize) -> (usize, HashMap<Circuit, usize>) {
        match self.cluster(experts) {
            None => (0, HashMap::new()),
            Some(c) => (c.e, c.expert_of),
        }
    }

    /// The top-1 expert each ingested atom routes to (aligned with ingest order); returns `(e, routes)` where `e` is the
    /// residual-bucket index (empty / un-anchored atoms route there). For per-expert interpretability + the contrib emit.
    pub fn routes(&self, experts: usize) -> (usize, Vec<usize>) {
        match self.cluster(experts) {
            None => (0, Vec::new()),
            Some(c) => {
                let r = self.atoms.iter().map(|a| if a.is_empty() { c.e } else { c.expert_of[a.iter().max_by_key(|x| c.freq[*x]).unwrap()] }).collect();
                (c.e, r)
            }
        }
    }

    /// Compute the clustering: partition the corpus working set into `experts` hub-anchored buckets. The top-`experts`
    /// circuits by frequency are the expert ANCHORS (the recurring hubs); each other circuit joins the anchor-expert it
    /// co-fires with most (co-fire = same atom); circuits that never co-fire with a hub fall into the residual bucket
    /// (index `e`). Returns the FULL assignment (`expert_of`) + per-expert sizes/token-routes + routing stats — the
    /// object both `render` (summary) and `partition` (the concrete expert→circuit sets) build from. `None` if empty.
    fn cluster(&self, experts: usize) -> Option<Cluster> {
        cluster_atoms(&self.atoms, experts)
    }

    /// Recursively sub-bucket the residual: cluster, then re-cluster the residual circuits (restricting atoms to them)
    /// into finer experts, up to `depth` levels or until the residual has < `min_circuits` circuits. Returns the leaf
    /// experts sorted by token-load (descending) with hierarchical labels (`e3`, `r.e1`, `r.r.e0`, …) + a `route` per
    /// ingested atom (index into the returned leaves) — the path that resolves the collapsed tail into domain experts.
    pub fn recursive(&self, experts: usize, depth: usize, min_circuits: usize) -> (Vec<RecExpert>, Vec<usize>) {
        // leaf_of: every circuit → its final hierarchical leaf label. Built top-down, residual descending each level.
        let mut leaf_of: HashMap<Circuit, String> = HashMap::new();
        // universe = the circuits still being clustered at this level; restricted atoms keep only those.
        let mut universe: Option<HashSet<Circuit>> = None; // None = all circuits (level 0)
        let mut prefix = String::new();
        for level in 0..depth.max(1) {
            let restricted: Vec<Vec<Circuit>> = match &universe {
                None => self.atoms.clone(),
                Some(u) => self.atoms.iter().map(|a| a.iter().copied().filter(|c| u.contains(c)).collect()).collect(),
            };
            let c = match cluster_atoms(&restricted, experts) {
                Some(c) => c,
                None => break,
            };
            let mut residual_circuits: Vec<Circuit> = Vec::new();
            for (&id, &ex) in &c.expert_of {
                if ex == c.e {
                    residual_circuits.push(id);
                } else {
                    leaf_of.insert(id, format!("{prefix}e{ex}"));
                }
            }
            let last = level + 1 >= depth.max(1) || residual_circuits.len() < min_circuits || c.e == 0;
            if last {
                for id in residual_circuits {
                    leaf_of.insert(id, format!("{prefix}residual"));
                }
                break;
            }
            universe = Some(residual_circuits.into_iter().collect());
            prefix.push_str("r.");
        }
        // depth = how many "r." prefixes the leaf label carries (= recursion level it was resolved at).
        self.finalize_tree(leaf_of, &|l: &str| l.matches("r.").count())
    }

    /// Build the leaf list (label + depth + circuit count + token load) and the per-atom route from a circuit→leaf-label
    /// map. `depth_of` computes a leaf's tree depth from its label. A token routes to the leaf of its atom's highest-
    /// global-frequency circuit. Leaves are returned sorted by token-load (the `--tree` printer regroups them by depth).
    fn finalize_tree(&self, leaf_of: HashMap<Circuit, String>, depth_of: &dyn Fn(&str) -> usize) -> (Vec<RecExpert>, Vec<usize>) {
        let mut freq: HashMap<Circuit, usize> = HashMap::new();
        for a in &self.atoms {
            for &id in a {
                *freq.entry(id).or_default() += 1;
            }
        }
        let mut idx_of: HashMap<String, usize> = HashMap::new();
        let mut leaves: Vec<RecExpert> = Vec::new();
        for lab in leaf_of.values() {
            if !idx_of.contains_key(lab) {
                idx_of.insert(lab.clone(), leaves.len());
                leaves.push(RecExpert { label: lab.clone(), depth: depth_of(lab), n_circuits: 0, tokens: 0 });
            }
        }
        for lab in leaf_of.values() {
            leaves[idx_of[lab]].n_circuits += 1;
        }
        let route: Vec<usize> = self.atoms.iter().map(|a| {
            a.iter().max_by_key(|id| freq[*id]).and_then(|id| leaf_of.get(id)).map(|l| idx_of[l]).unwrap_or(0)
        }).collect();
        for &r in &route {
            leaves[r].tokens += 1;
        }
        let mut order: Vec<usize> = (0..leaves.len()).collect();
        order.sort_by(|&a, &b| leaves[b].tokens.cmp(&leaves[a].tokens).then(leaves[a].label.cmp(&leaves[b].label)));
        let new_pos: HashMap<usize, usize> = order.iter().enumerate().map(|(np, &old)| (old, np)).collect();
        let sorted: Vec<RecExpert> = order.iter().map(|&o| leaves[o].clone()).collect();
        let route2: Vec<usize> = route.iter().map(|r| new_pos[r]).collect();
        (sorted, route2)
    }

    /// BALANCED tree: recursive `branch`-way bisection of the circuit set by co-occurrence — pick `branch` mutually-distant
    /// high-frequency seeds, assign each circuit to the seed it co-fires with most, capped to ~equal group sizes, and
    /// recurse until a group has ≤ `leaf_size` circuits. Low branching (default binary) + size balance ⇒ a DEEP, even tree
    /// (depth ~log_branch|C|, routing O(depth)) rather than the flat one-wide-level greedy tree. Returns leaves + routes.
    pub fn balanced(&self, branch: usize, leaf_size: usize) -> (Vec<RecExpert>, Vec<usize>) {
        let branch = branch.max(2);
        let mut freq: HashMap<Circuit, usize> = HashMap::new();
        for a in &self.atoms {
            for &id in a {
                *freq.entry(id).or_default() += 1;
            }
        }
        let all: Vec<Circuit> = {
            let mut v: Vec<Circuit> = freq.keys().copied().collect();
            v.sort();
            v
        };
        let mut leaf_of: HashMap<Circuit, String> = HashMap::new();
        self.split_balanced(&all, branch, leaf_size.max(1), &freq, &mut Vec::new(), &mut leaf_of);
        self.finalize_tree(leaf_of, &|l: &str| l.len())
    }

    /// One node of the balanced recursion: split `circuits` into `branch` size-balanced co-occurrence groups, recurse.
    /// `path` is the current node address (one char per level); the leaf label is the full path string.
    fn split_balanced(&self, circuits: &[Circuit], branch: usize, leaf_size: usize, freq: &HashMap<Circuit, usize>, path: &mut Vec<u8>, leaf_of: &mut HashMap<Circuit, String>) {
        if circuits.len() <= leaf_size || circuits.len() < branch {
            let lab = if path.is_empty() { "root".to_string() } else { String::from_utf8_lossy(path).into_owned() };
            for &c in circuits {
                leaf_of.insert(c, lab.clone());
            }
            return;
        }
        let cset: HashSet<Circuit> = circuits.iter().copied().collect();
        let seeds = self.pick_seeds(circuits, branch, freq, &cset);
        let groups = self.assign_balanced(circuits, &seeds, &cset);
        for (i, g) in groups.into_iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            path.push(b'0' + i as u8);
            self.split_balanced(&g, branch, leaf_size, freq, path, leaf_of);
            path.pop();
        }
    }

    /// Pick `branch` seeds: the highest-frequency circuit, then greedily the most frequent circuit that least co-fires
    /// with the seeds already chosen (frequent AND distant) — so the seeds anchor well-separated regions of the graph.
    fn pick_seeds(&self, circuits: &[Circuit], branch: usize, freq: &HashMap<Circuit, usize>, cset: &HashSet<Circuit>) -> Vec<Circuit> {
        let mut seeds: Vec<Circuit> = vec![*circuits.iter().max_by_key(|c| freq[*c]).unwrap()];
        while seeds.len() < branch.min(circuits.len()) {
            // co-occurrence of every candidate with the chosen seeds.
            let seedset: HashSet<Circuit> = seeds.iter().copied().collect();
            let mut cooc: HashMap<Circuit, u32> = circuits.iter().map(|&c| (c, 0u32)).collect();
            for a in &self.atoms {
                let present: Vec<Circuit> = a.iter().copied().filter(|c| cset.contains(c)).collect();
                let has_seed = present.iter().any(|c| seedset.contains(c));
                if !has_seed {
                    continue;
                }
                for &c in &present {
                    if let Some(v) = cooc.get_mut(&c) {
                        *v += 1;
                    }
                }
            }
            // maximise freq / (1 + cooc_with_seeds): frequent but distant. Skip already-chosen seeds.
            let next = circuits.iter().filter(|c| !seedset.contains(c)).max_by(|&&a, &&b| {
                let sa = freq[&a] as f64 / (1.0 + cooc[&a] as f64);
                let sb = freq[&b] as f64 / (1.0 + cooc[&b] as f64);
                sa.partial_cmp(&sb).unwrap().then(a.cmp(&b))
            });
            match next {
                Some(&c) => seeds.push(c),
                None => break,
            }
        }
        seeds
    }

    /// Assign each circuit to the seed it co-fires with most, capped to ~equal group sizes (size balance). Circuits with
    /// the clearest preference are placed first; ties / full groups fall to the next-best group with room.
    fn assign_balanced(&self, circuits: &[Circuit], seeds: &[Circuit], cset: &HashSet<Circuit>) -> Vec<Vec<Circuit>> {
        let branch = seeds.len();
        let seed_idx: HashMap<Circuit, usize> = seeds.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut cooc: HashMap<Circuit, Vec<u32>> = circuits.iter().map(|&c| (c, vec![0u32; branch])).collect();
        for a in &self.atoms {
            let present: Vec<Circuit> = a.iter().copied().filter(|c| cset.contains(c)).collect();
            let seeds_here: Vec<usize> = present.iter().filter_map(|c| seed_idx.get(c).copied()).collect();
            if seeds_here.is_empty() {
                continue;
            }
            for &c in &present {
                let v = cooc.get_mut(&c).unwrap();
                for &si in &seeds_here {
                    v[si] += 1;
                }
            }
        }
        // confidence = top co-occurrence minus the runner-up (place the most decisively-assigned circuits first).
        let conf = |c: &Circuit| -> i64 {
            let mut s = cooc[c].clone();
            s.sort_unstable_by(|a, b| b.cmp(a));
            s[0] as i64 - *s.get(1).unwrap_or(&0) as i64
        };
        let mut order: Vec<Circuit> = circuits.to_vec();
        order.sort_by(|a, b| conf(b).cmp(&conf(a)).then(a.cmp(b)));
        let cap = circuits.len().div_ceil(branch);
        let mut groups: Vec<Vec<Circuit>> = vec![Vec::new(); branch];
        for c in order {
            let mut prefs: Vec<usize> = (0..branch).collect();
            prefs.sort_by(|&x, &y| cooc[&c][y].cmp(&cooc[&c][x]).then(groups[x].len().cmp(&groups[y].len())));
            let g = prefs.iter().copied().find(|&gi| groups[gi].len() < cap).unwrap_or(prefs[0]);
            groups[g].push(c);
        }
        groups
    }

    /// A runtime RESIDENCY profile: experts sorted by token-load (descending) with cumulative coverage. The hot
    /// high-load experts that together cover `resident_cov` of tokens are the always-resident working set; the long tail
    /// of low-load buckets pages in on demand — the MoE residency policy. The frequency distribution IS the policy.
    pub fn residency(&self, experts: usize, resident_cov: f32) -> String {
        let mut out = String::new();
        let c = match self.cluster(experts) {
            Some(c) => c,
            None => {
                let _ = writeln!(out, "  (no atoms accumulated yet)");
                return out;
            }
        };
        let n: usize = c.tok_per_expert.iter().sum();
        if n == 0 {
            return out;
        }
        let mut v: Vec<(usize, usize, bool)> = (0..=c.e).map(|e| (e, c.tok_per_expert[e], e == c.e)).collect();
        v.retain(|x| x.1 > 0);
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let thresh = (resident_cov * n as f32).ceil() as usize;
        let (mut cum, mut resident_set) = (0usize, 0usize);
        let _ = writeln!(out, "  rank  expert      load   share   cumul   residency");
        for (rank, &(e, load, is_res)) in v.iter().enumerate() {
            let resident = cum < thresh; // experts needed to reach resident_cov coverage are the hot resident set
            cum += load;
            if resident {
                resident_set += 1;
            }
            let label = if is_res { "residual".to_string() } else { format!("e{e}") };
            let _ = writeln!(out, "  {:>3}   {:<10} {:>5}  {:>4.0}%   {:>4.0}%   {}", rank + 1, label, load, 100.0 * load as f32 / n as f32, 100.0 * cum as f32 / n as f32, if resident { "RESIDENT" } else { "paged tail" });
        }
        let _ = writeln!(out, "  → hot resident set: {resident_set} expert(s) cover ~{:.0}% of tokens; {} buckets are the paged long tail (loaded on demand).", 100.0 * resident_cov, v.len().saturating_sub(resident_set));
        out
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

/// Core hub-anchored clustering over a given atom list (so it can run on a residual-restricted universe for recursion).
///
/// A deterministic, embedding-free heuristic — a SOUND grouping of the (density-min-derived) atoms, not a claim of a
/// globally optimal partition:
///   1. count each circuit's frequency across the atoms (a circuit = `(kind, layer, idx)`);
///   2. the top-`experts` circuits by frequency are the expert ANCHORS (the recurring hubs);
///   3. every other circuit joins the anchor it CO-FIRES with most (co-fire = appears in the same atom);
///   4. circuits that never co-fire with any anchor fall into the residual bucket (index `e`).
/// Every ordering uses a stable sort with a circuit-id tiebreak (freq desc, then id; co-occurrence desc, then expert
/// id), so the partition is fully reproducible for a given atom set — no randomness, no embedding distance.
fn cluster_atoms(atoms: &[Vec<Circuit>], experts: usize) -> Option<Cluster> {
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

/// Summary metrics for a built tree (leaves carry depth + token-load), for comparing tree algorithms: size, depth (max +
/// the load-weighted mean ROUTING depth), and load balance (hottest-leaf share + normalized load entropy where 1.0 =
/// perfectly even). A balanced low-branch tree should show a higher mean-depth but a far lower hottest-leaf and a
/// load-balance near 1.0 vs the flat greedy tree.
pub fn tree_metrics(leaves: &[RecExpert]) -> String {
    let active: Vec<&RecExpert> = leaves.iter().filter(|l| l.tokens > 0).collect();
    let n: usize = active.iter().map(|l| l.tokens).sum();
    if n == 0 || active.is_empty() {
        return "  (no routed tokens)".to_string();
    }
    let maxd = active.iter().map(|l| l.depth).max().unwrap_or(0);
    let mean_depth: f64 = active.iter().map(|l| l.depth as f64 * l.tokens as f64).sum::<f64>() / n as f64;
    let max_share = active.iter().map(|l| l.tokens).max().unwrap_or(0) as f64 / n as f64;
    let h: f64 = active.iter().map(|l| { let p = l.tokens as f64 / n as f64; -p * p.log2() }).sum();
    let balance = h / (active.len() as f64).log2().max(1e-9);
    let mean_circ: f64 = active.iter().map(|l| l.n_circuits as f64).sum::<f64>() / active.len() as f64;
    format!("  leaves={}  max-depth={maxd}  mean-routing-depth={mean_depth:.2}  hottest-leaf={:.0}%  load-balance={balance:.2}  mean-circuits/leaf={mean_circ:.1}", active.len(), 100.0 * max_share)
}

/// One leaf expert from a tree build: a hierarchical label, its recursion depth, circuit count, token load.
#[derive(Clone)]
pub struct RecExpert {
    pub label: String,
    pub depth: usize,
    pub n_circuits: usize,
    pub tokens: usize,
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

    #[test]
    fn partition_emits_expert_circuit_sets() {
        let mut b = CorpusBuckets::new();
        let (h0, n0) = ((0u8, 23, 1), (1u8, 23, 2539)); // two recurring hubs
        for t in 0..10 {
            b.ingest(vec![h0, n0, (1u8, t, t)]); // + a per-token private circuit
        }
        let p = b.partition(2);
        assert_eq!(p.experts, 2);
        assert_eq!(p.n_tokens, 10);
        assert_eq!(p.distinct_circuits, 12); // 2 hubs + 10 private
        // every circuit is assigned to exactly one bucket (Σ sizes == distinct).
        assert_eq!(p.buckets.iter().map(|e| e.size).sum::<usize>(), p.distinct_circuits);
        // the residual bucket is last and flagged; the anchored experts carry an anchor.
        assert!(p.buckets.last().unwrap().residual);
        assert!(p.buckets[0].anchor.is_some());
        assert!(!p.buckets[0].circuits.is_empty());
    }

    #[test]
    fn residency_splits_hot_and_tail() {
        let mut b = CorpusBuckets::new();
        let hot = (1u8, 23, 1);
        for _ in 0..50 {
            b.ingest(vec![hot]); // a dominant always-on expert
        }
        for t in 0..5 {
            b.ingest(vec![(1u8, 9, t)]); // a few rare singletons → the paged tail
        }
        let r = b.residency(8, 0.9);
        assert!(r.contains("RESIDENT"), "the hot expert must be marked resident:\n{r}");
        assert!(r.contains("hot resident set"), "a hot/paged summary line is expected:\n{r}");
    }

    #[test]
    fn routes_match_expert_map() {
        let mut b = CorpusBuckets::new();
        let h = (0u8, 1, 1);
        for _ in 0..8 {
            b.ingest(vec![h]); // every atom is the same hub ⇒ all route to the hub's expert
        }
        let (_e, map) = b.expert_map(2);
        assert!(map.contains_key(&h));
        let (_e2, routes) = b.routes(2);
        assert_eq!(routes.len(), 8);
        assert!(routes.iter().all(|&r| r == map[&h]), "hub-only atoms must all route to the hub's expert");
    }

    #[test]
    fn recursive_splits_the_residual() {
        let mut b = CorpusBuckets::new();
        let hub = (1u8, 23, 1);
        let (a1, a2) = ((1u8, 1, 1), (1u8, 1, 2)); // residual group A (co-fire with each other, not the hub)
        let (z1, z2) = ((1u8, 2, 1), (1u8, 2, 2)); // residual group B
        for _ in 0..10 { b.ingest(vec![hub]); } // hub-only atoms ⇒ e0
        for _ in 0..6 { b.ingest(vec![a1, a2]); } // group A ⇒ residual at L0, its own sub-expert at L1
        for _ in 0..6 { b.ingest(vec![z1, z2]); } // group B ⇒ residual at L0, its own sub-expert deeper
        let (leaves, route) = b.recursive(1, 3, 1); // 1 expert/level, depth 3 ⇒ the residual resolves into sub-experts
        assert!(leaves.len() >= 3, "recursion should resolve the residual into ≥2 sub-experts; got {:?}", leaves.iter().map(|l| &l.label).collect::<Vec<_>>());
        assert!(leaves.iter().any(|l| l.label.starts_with("r.")), "expected hierarchical r.* leaf labels");
        assert_eq!(route.len(), b.n_tokens()); // one route per ingested atom
    }

    #[test]
    fn balanced_bisects_two_clusters() {
        let mut b = CorpusBuckets::new();
        let a = [(1u8, 1, 0), (1u8, 1, 1), (1u8, 1, 2)]; // cluster A: these three co-fire together
        let bb = [(1u8, 2, 0), (1u8, 2, 1), (1u8, 2, 2)]; // cluster B: co-fire together, never with A
        for _ in 0..10 { b.ingest(a.to_vec()); }
        for _ in 0..10 { b.ingest(bb.to_vec()); }
        let (leaves, route) = b.balanced(2, 1); // binary split, single-circuit leaves
        assert_eq!(route.len(), 20);
        // a binary balanced tree over 6 circuits must split the root — every leaf sits below it (depth >= 1).
        assert!(leaves.iter().all(|l| l.depth >= 1), "balanced tree must split the root");
        assert!(tree_metrics(&leaves).contains("load-balance="));
        // the two disjoint co-occurrence clusters must land in DIFFERENT leaves (the bisection separated them).
        assert_ne!(route[0], route[10], "the two clusters should route to different leaves");
    }

    #[test]
    fn balanced_terminates_on_a_clique() {
        // a single all-co-firing clique (no natural split) must still terminate and partition every circuit.
        let mut b = CorpusBuckets::new();
        let clique: Vec<Circuit> = (0..8).map(|i| (1u8, 0, i)).collect();
        for _ in 0..5 { b.ingest(clique.clone()); }
        let (leaves, route) = b.balanced(2, 2); // leaf-size 2
        assert_eq!(route.len(), 5);
        assert!(!leaves.is_empty());
        assert_eq!(leaves.iter().map(|l| l.n_circuits).sum::<usize>(), 8); // every circuit assigned exactly once
    }
}
