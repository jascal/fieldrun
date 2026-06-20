//! recursion → recursive Datalog (interpretability-FIRST: full faithfulness to the model, in symbolic form).
//!
//! Parse the arithmetic s-expression recovered from the recursion-explain token stream, then emit a RUNNABLE,
//! depth-bounded recursive Soufflé program that reproduces the MODEL'S answer — right or wrong:
//!
//!   eval(N,V,B) :- B>0, node(N,"+",A,C), eval(A,X,B-1), eval(C,Y,B-1), V=X+Y.   // recurse while budget remains
//!   eval(N,V,0) :- retrieved(N,V).                                              // CUT: out of depth → memorized value
//!
//! The budget B is the model's effective recursion depth (≈ its layer budget); the CUT clause is what the semiring /
//! forge-tax core was carrying — a retrieved value, made legible. We ABDUCE the broken program from the model's output:
//! find the DEEPEST depth D at which "recurse to D, take a memorized (context-literal) value at the frontier" reproduces
//! the model's exact answer. Correctness (the real math) is a secondary annotation; faithfulness to the model is the law.

pub enum Node {
    Leaf(i64),
    Op(char, Box<Node>, Box<Node>, usize, usize),
}

pub fn parse(atoms: &[(String, usize)], pos: &mut usize) -> Option<Node> {
    let (a, tok) = atoms.get(*pos)?.clone();
    if a == "(" {
        let open = tok;
        *pos += 1;
        let op = atoms.get(*pos)?.0.chars().next()?;
        *pos += 1;
        let mut kids = Vec::new();
        while atoms.get(*pos).map(|x| x.0.as_str()) != Some(")") {
            if *pos >= atoms.len() {
                return None;
            }
            kids.push(parse(atoms, pos)?);
        }
        let close = atoms[*pos].1;
        *pos += 1;
        if kids.len() < 2 {
            return kids.into_iter().next();
        }
        let mut it = kids.into_iter();
        let mut acc = it.next().unwrap();
        for k in it {
            acc = Node::Op(op, Box::new(acc), Box::new(k), open, close);
        }
        Some(acc)
    } else if let Ok(v) = a.parse::<i64>() {
        *pos += 1;
        Some(Node::Leaf(v))
    } else {
        *pos += 1;
        parse(atoms, pos)
    }
}

pub fn parse_target(atoms: &[(String, usize)]) -> Option<Node> {
    let mut pos = 0;
    let mut last = None;
    while pos < atoms.len() {
        if atoms[pos].0 == "(" {
            let mut p = pos;
            match parse(atoms, &mut p) {
                Some(n @ Node::Op(..)) => { last = Some(n); pos = p; }
                _ => pos += 1,
            }
        } else {
            pos += 1;
        }
    }
    last
}

// ── flat representation (index = node id, in pre-order to match the emitted facts) ──
pub enum Flat {
    Leaf(i64),
    Op(char, usize, usize), // op, left id, right id
}

fn flatten(n: &Node, out: &mut Vec<Flat>, depth: &mut Vec<usize>, d: usize) -> usize {
    let id = out.len();
    out.push(Flat::Leaf(0));
    depth.push(d);
    match n {
        Node::Leaf(v) => out[id] = Flat::Leaf(*v),
        Node::Op(op, a, b, ..) => {
            let ai = flatten(a, out, depth, d + 1);
            let bi = flatten(b, out, depth, d + 1);
            out[id] = Flat::Op(*op, ai, bi);
        }
    }
    id
}

fn apply(op: char, x: i64, y: i64) -> Option<i64> {
    Some(match op {
        '+' => x + y,
        '-' => x - y,
        '*' => x * y,
        '/' => { if y == 0 { return None; } x / y }
        _ => return None,
    })
}

/// Evaluate node `idx` with recursion budget `b`. At budget 0 an Op is CUT → its abduced/retrieved value.
fn eval_bounded(nodes: &[Flat], idx: usize, b: i64, cuts: &std::collections::HashMap<usize, i64>) -> Option<i64> {
    match &nodes[idx] {
        Flat::Leaf(v) => Some(*v),
        Flat::Op(op, a, c) => {
            if b <= 0 {
                cuts.get(&idx).copied()
            } else {
                apply(*op, eval_bounded(nodes, *a, b - 1, cuts)?, eval_bounded(nodes, *c, b - 1, cuts)?)
            }
        }
    }
}

pub struct Abduction {
    pub depth: i64,                       // deepest recursion budget that reproduces the model (effective depth)
    pub cuts: Vec<(usize, i64, bool)>,    // (node id, retrieved value, from_context_literal)
}

/// Abduce the model's broken program: the DEEPEST D such that recursing to D and taking memorized values at the cut
/// frontier reproduces `answer`. Cut values prefer `literals` (a value retrieved from context); else any small int.
pub fn abduce(nodes: &[Flat], depth: &[usize], answer: i64, literals: &[i64]) -> Option<Abduction> {
    let maxd = *depth.iter().max().unwrap_or(&0) as i64;
    let mut lits: Vec<i64> = literals.to_vec();
    lits.sort_unstable();
    lits.dedup();
    let fallback: Vec<i64> = (0..=50).collect();
    for d in (0..=maxd).rev() {
        let frontier: Vec<usize> = (0..nodes.len())
            .filter(|&i| depth[i] as i64 == d && matches!(nodes[i], Flat::Op(..)))
            .collect();
        if frontier.len() > 2 {
            continue; // keep abduction tractable; a shallower D will have a smaller frontier
        }
        // empty frontier = full recursion to this depth — faithful iff it already reproduces the model
        if frontier.is_empty() {
            if eval_bounded(nodes, 0, d, &std::collections::HashMap::new()) == Some(answer) {
                return Some(Abduction { depth: d, cuts: vec![] });
            }
            continue;
        }
        // try context literals first (a genuine retrieval), then a small-int fallback
        for (src, pool) in [(true, &lits), (false, &fallback)] {
            let assign = match frontier.len() {
                1 => pool.iter().find_map(|&v| {
                    let cuts = std::collections::HashMap::from([(frontier[0], v)]);
                    (eval_bounded(nodes, 0, d, &cuts) == Some(answer)).then_some(vec![v])
                }),
                2 => {
                    let mut found = None;
                    'outer: for &v0 in pool {
                        for &v1 in pool {
                            let cuts = std::collections::HashMap::from([(frontier[0], v0), (frontier[1], v1)]);
                            if eval_bounded(nodes, 0, d, &cuts) == Some(answer) {
                                found = Some(vec![v0, v1]);
                                break 'outer;
                            }
                        }
                    }
                    found
                }
                _ => None,
            };
            if let Some(vals) = assign {
                let cuts = frontier.iter().zip(&vals).map(|(&i, &v)| (i, v, src)).collect();
                return Some(Abduction { depth: d, cuts });
            }
        }
    }
    None
}

/// Emit the depth-bounded recursive Datalog program. `model_answer` is the model's ACTUAL answer (the faithfulness
/// target); `literals` are the integers in the input context (candidate retrieved values for the cuts).
pub fn emit(root: &Node, model_answer: Option<i64>, literals: &[i64]) -> String {
    let mut nodes = Vec::new();
    let mut depth = Vec::new();
    flatten(root, &mut nodes, &mut depth, 0);
    let correct = eval_bounded(&nodes, 0, i64::MAX, &Default::default());
    let abd = model_answer.and_then(|a| abduce(&nodes, &depth, a, literals));

    let mut o = String::new();
    o.push_str("// recursion → depth-bounded recursive Datalog  (fieldrun --recursion-explain --datalog · Soufflé)\n");
    o.push_str("// FAITHFUL to the model (interpretability first): a depth-bounded recursive evaluator whose CUT clause\n");
    o.push_str("// (out of budget → a memorized/retrieved value) reproduces the model's answer — right or wrong. The\n");
    o.push_str("// budget B is the model's effective recursion depth; the cut is the semiring/forge-tax core made legible.\n");
    o.push_str("// Run:  souffle -D- this.dl\n\n");
    o.push_str(".decl leaf(n:symbol, v:number)\n.decl node(n:symbol, op:symbol, a:symbol, b:symbol)\n");
    o.push_str(".decl retrieved(n:symbol, v:number)\n.decl eval(n:symbol, v:number, b:number)\n");
    o.push_str(".decl answer(v:number)\n.decl model_answer(v:number)\n");
    o.push_str(".decl reproduces(v:number)\n.decl diverges(prog:number, model:number)\n\n");

    o.push_str("// ---- parse tree (recovered from the recursion-explain bracket folds) ----\n");
    for (i, f) in nodes.iter().enumerate() {
        match f {
            Flat::Leaf(v) => o.push_str(&format!("leaf(\"n{i}\", {v}).\n")),
            Flat::Op(op, a, b) => o.push_str(&format!("node(\"n{i}\", \"{op}\", \"n{a}\", \"n{b}\").\n")),
        }
    }
    let dbud = abd.as_ref().map(|x| x.depth).unwrap_or(i64::from(*depth.iter().max().unwrap_or(&0) as i32));
    if let Some(a) = &abd {
        o.push_str("\n// ---- the CUTS: where the model ran out of recursion budget and RETRIEVED a value (abduced) ----\n");
        for &(i, v, from_ctx) in &a.cuts {
            let why = if from_ctx { "memorized literal from context (a genuine retrieval)" } else { "value the model held (not a clean context literal)" };
            o.push_str(&format!("retrieved(\"n{i}\", {v}).   // BROKEN: did not recurse here — {why}\n"));
        }
        if a.cuts.is_empty() {
            o.push_str("// (none — the model recursed all the way; no cut needed)\n");
        }
    }
    match model_answer {
        Some(a) => o.push_str(&format!("model_answer({a}).\n")),
        None => o.push_str("// model_answer: the model's answer did not decode to an integer\n"),
    }

    o.push_str("\n// ---- the depth-bounded RECURSIVE evaluator (cut = retrieval at the budget) ----\n");
    o.push_str("eval(N,V,B) :- B <= 0, retrieved(N,V).                                  // CUT (the semiring core)\n");
    o.push_str("eval(N,V,B) :- B > 0, leaf(N,V).\n");
    o.push_str("eval(N,V,B) :- B > 0, node(N,\"+\",A,C), eval(A,X,B-1), eval(C,Y,B-1), V = X + Y.\n");
    o.push_str("eval(N,V,B) :- B > 0, node(N,\"-\",A,C), eval(A,X,B-1), eval(C,Y,B-1), V = X - Y.\n");
    o.push_str("eval(N,V,B) :- B > 0, node(N,\"*\",A,C), eval(A,X,B-1), eval(C,Y,B-1), V = X * Y.\n");
    o.push_str("eval(N,V,B) :- B > 0, node(N,\"/\",A,C), eval(A,X,B-1), eval(C,Y,B-1), Y != 0, V = X / Y.\n\n");
    o.push_str(&format!("answer(V)     :- eval(\"n0\", V, {dbud}).        // run at the model's effective depth\n"));
    o.push_str("reproduces(V)  :- answer(V), model_answer(V).             // FAITHFUL: symbolic == model ✓\n");
    o.push_str("diverges(P,M)  :- answer(P), model_answer(M), P != M.\n\n");
    o.push_str(".output answer\n.output reproduces\n.output diverges\n");

    // verdict comment
    let correct_s = correct.map(|v| v.to_string()).unwrap_or_else(|| "?".into());
    o.push_str(&format!("\n// correct math: {correct_s}"));
    match (&abd, model_answer) {
        (Some(a), Some(m)) => {
            let prog = eval_bounded(&nodes, 0, a.depth,
                &a.cuts.iter().map(|&(i, v, _)| (i, v)).collect());
            let faith = if prog == Some(m) { "FAITHFUL" } else { "NOT reproduced" };
            o.push_str(&format!(" · model: {m} · effective recursion depth D={} ({} cut{}) · {faith}\n",
                a.depth, a.cuts.len(), if a.cuts.len() == 1 { "" } else { "s" }));
            if correct != Some(m) {
                o.push_str("// → the model is mathematically WRONG; the faithful program is the BROKEN (early-cut) recursion above.\n");
            }
        }
        (None, Some(m)) => o.push_str(&format!(" · model: {m} · could not abduce a depth-cut reproducing it (semiring fallback territory)\n")),
        _ => o.push_str(" · (model answer not an integer)\n"),
    }
    o
}
