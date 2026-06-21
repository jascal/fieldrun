//! Probe/experiment harness for the recursion work: expression generators + parse-tree walkers used by the
//! `--recursion-explain` sub-modes (`--measure`, `--discover`, `--induce`). Kept OUT of main.rs (which is arg
//! dispatch) and OUT of recursion_dl.rs (which is the Datalog artifact) — this module is the *experiments*.

use crate::flag;
use crate::recursion_dl;

/// Small xorshift PRNG (deterministic; `Date.now`/`rand` are intentionally avoided so sweeps are reproducible).
pub fn xorshift(r: &mut u64) -> u64 {
    *r ^= *r << 13;
    *r ^= *r >> 7;
    *r ^= *r << 17;
    *r
}

/// A random arithmetic s-expression of exactly `depth` nesting, all sub-results in [0, maxv]. Returns (string, value).
pub fn gen_arith(depth: usize, maxv: i64, r: &mut u64) -> (String, i64) {
    if depth == 0 {
        let v = (xorshift(r) % (maxv as u64 + 1)) as i64;
        return (v.to_string(), v);
    }
    for _ in 0..400 {
        let op = ['+', '-', '*'][(xorshift(r) % 3) as usize];
        let (mut dl, mut dr) = (depth - 1, (xorshift(r) % depth as u64) as usize);
        if xorshift(r) & 1 == 1 {
            std::mem::swap(&mut dl, &mut dr);
        }
        let (ls, lv) = gen_arith(dl, maxv, r);
        let (rs, rv) = gen_arith(dr, maxv, r);
        if op == '-' && lv < rv {
            continue;
        }
        let v = match op { '+' => lv + rv, '-' => lv - rv, '*' => lv * rv, _ => 0 };
        if !(0..=maxv).contains(&v) {
            continue;
        }
        return (format!("({op} {ls} {rs})"), v);
    }
    let v = (xorshift(r) % (maxv as u64 + 1)) as i64;
    (v.to_string(), v)
}

/// Mean and population std-dev of a slice — for seed-variance reporting (`--seeds K` re-runs with K seeds).
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() { return (0.0, 0.0); }
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

/// A random depth-`depth` expression over an arbitrary operator set `ops`, all sub-results in [0, maxv] (or {0,1} for
/// Boolean). Generalises gen_arith to any family — lets --measure gauge WHICH recursive functions a model can do.
pub fn gen_family(depth: usize, ops: &[char], maxv: i64, r: &mut u64) -> (String, i64) {
    let ev = |op: char, x: i64, y: i64| -> Option<i64> {
        match op {
            '+' => Some(x + y), '-' => (x >= y).then_some(x - y), '*' => Some(x * y),
            '/' => (y != 0 && x % y == 0).then(|| x / y),
            '<' => Some(x.min(y)), '>' => Some(x.max(y)), '&' => Some(x & y), '|' => Some(x | y), '^' => Some(x ^ y),
            '%' => (y != 0).then(|| x % y), _ => None,
        }
    };
    if depth == 0 || ops.is_empty() {
        let v = (xorshift(r) % (maxv as u64 + 1)) as i64;
        return (v.to_string(), v);
    }
    for _ in 0..400 {
        let op = ops[(xorshift(r) % ops.len() as u64) as usize];
        let (mut dl, mut dr) = (depth - 1, (xorshift(r) % depth as u64) as usize);
        if xorshift(r) & 1 == 1 { std::mem::swap(&mut dl, &mut dr); }
        let (ls, lv) = gen_family(dl, ops, maxv, r);
        let (rs, rv) = gen_family(dr, ops, maxv, r);
        if let Some(v) = ev(op, lv, rv) {
            if (0..=maxv).contains(&v) { return (format!("({op} {ls} {rs})"), v); }
        }
    }
    let v = (xorshift(r) % (maxv as u64 + 1)) as i64;
    (v.to_string(), v)
}

/// Collect every Op node (as a reference), pre-order — so each can be graded against its true value.
pub fn collect_ops<'a>(n: &'a recursion_dl::Node, out: &mut Vec<&'a recursion_dl::Node>) {
    if let recursion_dl::Node::Op(_, a, b, ..) = n {
        out.push(n);
        collect_ops(a, out);
        collect_ops(b, out);
    }
}

/// Collect the leaf values of a subtree (the genuine input operands of THIS expression).
pub fn collect_leaves(n: &recursion_dl::Node, out: &mut Vec<i64>) {
    match n {
        recursion_dl::Node::Leaf(v) => out.push(*v),
        recursion_dl::Node::Op(_, a, b, ..) => { collect_leaves(a, out); collect_leaves(b, out); }
    }
}

/// The true value of a subtree (ground truth for grading the trace reads — correctness is a secondary annotation).
pub fn true_eval(n: &recursion_dl::Node) -> Option<i64> {
    match n {
        recursion_dl::Node::Leaf(v) => Some(*v),
        recursion_dl::Node::Op(op, a, b, ..) => {
            let (x, y) = (true_eval(a)?, true_eval(b)?);
            Some(match op {
                '+' => x + y, '-' => x - y, '*' => x * y, '/' => { if y == 0 { return None; } x / y },
                '<' => x.min(y), '>' => x.max(y), '&' => x & y, '|' => x | y, '^' => x ^ y,
                '%' => { if y == 0 { return None; } x % y }, _ => return None,
            })
        }
    }
}

/// A controlled depth-2 expression `(rop (lop a b) (rop2 c d))` — both children are genuine Op subtrees, so the
/// structural positions root/left/right are unambiguous. All sub-results in [0, maxv]. Returns the string.
pub fn gen_depth2(ops: &[char], maxv: i64, r: &mut u64) -> Option<String> {
    let ev = |op: char, x: i64, y: i64| -> Option<i64> {
        match op {
            '+' => Some(x + y), '*' => Some(x * y), '-' => (x >= y).then_some(x - y),
            '<' => Some(x.min(y)), '>' => Some(x.max(y)), '&' => Some(x & y), '|' => Some(x | y), '^' => Some(x ^ y),
            _ => None,
        }
    };
    for _ in 0..500 {
        let pick = |r: &mut u64| ops[(xorshift(r) % ops.len() as u64) as usize];
        let leaf = |r: &mut u64| (xorshift(r) % (maxv as u64 + 1)) as i64;
        let (rop, lop, rop2) = (pick(r), pick(r), pick(r));
        let (a, b, c, d) = (leaf(r), leaf(r), leaf(r), leaf(r));
        let lv = match ev(lop, a, b) { Some(v) if (0..=maxv).contains(&v) => v, _ => continue };
        let rv = match ev(rop2, c, d) { Some(v) if (0..=maxv).contains(&v) => v, _ => continue };
        if ev(rop, lv, rv).is_none() { continue; }
        return Some(format!("({rop} ({lop} {a} {b}) ({rop2} {c} {d}))"));
    }
    None
}

/// A nested expression using a SINGLE operator `op` (semantics `opf`), all sub-results in [0, maxv].
/// Used by --discover to probe the model with a (possibly novel) symbol whose meaning we do not assume.
pub fn gen_op(depth: usize, op: char, opf: fn(i64, i64) -> Option<i64>, maxv: i64, r: &mut u64) -> Option<(String, i64)> {
    if depth == 0 {
        let v = (xorshift(r) % (maxv as u64 + 1)) as i64;
        return Some((v.to_string(), v));
    }
    for _ in 0..400 {
        let (mut dl, mut dr) = (depth - 1, (xorshift(r) % depth as u64) as usize);
        if xorshift(r) & 1 == 1 { std::mem::swap(&mut dl, &mut dr); }
        let (ls, lv) = gen_op(dl, op, opf, maxv, r)?;
        let (rs, rv) = gen_op(dr, op, opf, maxv, r)?;
        if let Some(v) = opf(lv, rv) {
            if (0..=maxv).contains(&v) {
                return Some((format!("({op} {ls} {rs})"), v));
            }
        }
    }
    None
}

/// Atomize a token-id sequence into (atom, source-token-index) pairs (splits BPE merges; recognises the operator set).
pub fn atomize_ids(tg: &crate::api::TextGen, ids: &[i64]) -> Vec<(String, usize)> {
    let mut atoms: Vec<(String, usize)> = Vec::new();
    for (ti, &id) in ids.iter().enumerate() {
        let mut num = String::new();
        for ch in tg.decode(&[id]).chars() {
            if ch.is_ascii_digit() { num.push(ch); }
            else {
                if !num.is_empty() { atoms.push((std::mem::take(&mut num), ti)); }
                if matches!(ch, '(' | ')' | '+' | '-' | '*' | '/' | '&' | '|' | '^' | '<' | '>' | '%') { atoms.push((ch.to_string(), ti)); }
            }
        }
        if !num.is_empty() { atoms.push((num, ti)); }
    }
    atoms
}

/// A matched pair of depth-2 exprs differing ONLY in the LEFT subtree's value — same right child, same root/left ops,
/// all single-digit operands so the two tokenize identically and positions align. For the causal interchange test.
/// Returns (exprA, exprB, ansA, ansB, leftValA, leftValB); root op in {+,-}, both answers single-digit and distinct.
pub fn gen_patch_pair(r: &mut u64) -> Option<(String, String, i64, i64, i64, i64)> {
    let pick9 = |r: &mut u64| (xorshift(r) % 10) as i64;
    for _ in 0..3000 {
        let rop = if xorshift(r) & 1 == 0 { '+' } else { '-' };
        let rop2 = ['+', '*'][(xorshift(r) % 2) as usize];
        let (c, d) = (pick9(r), pick9(r));
        let rv = match rop2 { '+' => c + d, '*' => c * d, _ => continue };
        if !(0..=9).contains(&rv) { continue; }
        let (la, lb) = (pick9(r), pick9(r));
        if la == lb { continue; }
        let ans = |lv: i64| -> Option<i64> { let a = match rop { '+' => lv + rv, '-' => lv - rv, _ => return None }; (0..=9).contains(&a).then_some(a) };
        let (ansa, ansb) = match (ans(la), ans(lb)) { (Some(a), Some(b)) if a != b => (a, b), _ => continue };
        let left_str = |lv: i64, r: &mut u64| -> Option<String> {
            let a1 = (xorshift(r) % (lv as u64 + 1)) as i64;
            let a2 = lv - a1;
            (0..=9).contains(&a2).then(|| format!("(+ {a1} {a2})"))
        };
        let (lsa, lsb) = match (left_str(la, r), left_str(lb, r)) { (Some(x), Some(y)) => (x, y), _ => continue };
        return Some((format!("({rop} {lsa} ({rop2} {c} {d}))"), format!("({rop} {lsb} ({rop2} {c} {d}))"), ansa, ansb, la, lb));
    }
    None
}

/// A matched pair like gen_patch_pair but with the RIGHT subtree a left-nested `+` CHAIN of `hold+1` leaves — so the
/// left value must be carried across a longer computation before the root consumes it. `hold` = right-subtree depth =
/// the HOLD DISTANCE. A,B differ only in the (shallow) left subtree; both answers single-digit and distinct.
pub fn gen_hold_pair(hold: usize, r: &mut u64) -> Option<(String, String, i64, i64)> {
    if hold < 1 { return None; }
    let pick9 = |r: &mut u64| (xorshift(r) % 10) as i64;
    let k = hold + 1; // leaves in the right chain
    for _ in 0..4000 {
        let rop = if xorshift(r) & 1 == 0 { '+' } else { '-' };
        let rv = pick9(r);
        // partition rv into k leaves (each 0..9, sum == rv)
        let mut rem = rv;
        let mut leaves = Vec::with_capacity(k);
        for i in 0..k {
            let v = if i == k - 1 { rem } else { (xorshift(r) % (rem as u64 + 1)) as i64 };
            leaves.push(v);
            rem -= v;
        }
        let mut chain = format!("(+ {} {})", leaves[0], leaves[1]);
        for &lf in &leaves[2..] { chain = format!("(+ {chain} {lf})"); }
        let (la, lb) = (pick9(r), pick9(r));
        if la == lb { continue; }
        let ans = |lv: i64| -> Option<i64> { let a = match rop { '+' => lv + rv, '-' => lv - rv, _ => return None }; (0..=9).contains(&a).then_some(a) };
        let (ansa, ansb) = match (ans(la), ans(lb)) { (Some(a), Some(b)) if a != b => (a, b), _ => continue };
        let lstr = |lv: i64, r: &mut u64| -> Option<String> { let x = (xorshift(r) % (lv as u64 + 1)) as i64; let y = lv - x; (0..=9).contains(&y).then(|| format!("(+ {x} {y})")) };
        let (lsa, lsb) = match (lstr(la, r), lstr(lb, r)) { (Some(x), Some(y)) => (x, y), _ => continue };
        return Some((format!("({rop} {lsa} {chain})"), format!("({rop} {lsb} {chain})"), ansa, ansb));
    }
    None
}

/// LIST-RECURSION battery — gauge which RECURSIVE functions (over lists/sequences, not just numeric expressions) the
/// model can do. These are the canonical recursive programs (length/last/member/fold), closer to what an LLM actually
/// does than arithmetic. Single-token-answer folds so they grade in the same framework; accuracy-by-list-length is the
/// list-depth cliff (the analog of arithmetic's nesting-depth D*).
pub fn run_list_measure(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --list-measure needs a tokenizer next to {stem}"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(30);
    let lmin: usize = 3;
    let lmax: usize = flag(args, "--lmax").and_then(|s| s.parse().ok()).unwrap_or(6);
    let base_seed: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15);
    let seeds: usize = flag(args, "--seeds").and_then(|s| s.parse().ok()).unwrap_or(1);
    type LFn = (&'static str, &'static str, fn(&[i64]) -> Option<i64>);
    let fns: [LFn; 6] = [
        ("last",  "last 3 7 2 5 = 5\nlast 1 8 4 = 4\nlast 6 0 9 2 = 2\n", |l| l.last().copied()),
        ("first", "first 3 7 2 5 = 3\nfirst 1 8 4 = 1\nfirst 6 0 9 2 = 6\n", |l| l.first().copied()),
        ("len",   "len 3 7 2 5 = 4\nlen 1 8 4 = 3\nlen 6 0 9 2 5 = 5\n", |l| Some(l.len() as i64)),
        ("max",   "max 3 7 2 5 = 7\nmax 1 8 4 = 8\nmax 6 0 9 2 = 9\n", |l| l.iter().max().copied()),
        ("min",   "min 3 7 2 5 = 2\nmin 1 8 4 = 1\nmin 6 0 9 2 = 0\n", |l| l.iter().min().copied()),
        ("sum",   "sum 3 1 2 = 6\nsum 4 0 1 = 5\nsum 2 3 1 = 6\n", |l| { let s: i64 = l.iter().sum(); (s <= 9).then_some(s) }),
    ];
    eprintln!("[fieldrun] list-recursion battery · {n} lists/fn × {seeds} seeds · len {lmin}..{lmax} · {stem}");
    println!("# list-recursion battery — which RECURSIVE (list-fold) functions can the model do? ({stem})");
    println!("  function  accuracy(mean±sd over {seeds} seeds)   acc-by-list-length(pooled)");
    for (name, prime, truth) in fns {
        let mut accs: Vec<f64> = Vec::new();
        let mut bylen: std::collections::BTreeMap<usize, (usize, usize)> = std::collections::BTreeMap::new();
        for si in 0..seeds {
            let mut rng: u64 = (base_seed ^ (si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
            let (mut tot, mut ok) = (0usize, 0usize);
            let mut tries = 0;
            while tot < n && tries < n * 30 {
                tries += 1;
                let len = lmin + (xorshift(&mut rng) % (lmax - lmin + 1) as u64) as usize;
                let list: Vec<i64> = (0..len).map(|_| (xorshift(&mut rng) % 10) as i64).collect();
                let ans = match truth(&list) { Some(a) if (0..=9).contains(&a) => a, _ => continue };
                let listing = list.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
                let mut g = tg.encode(&format!("{prime}{name} {listing} ="), false);
                let mut cont = String::new();
                for _ in 0..3 { let t = lm.predict(&g); let s = tg.decode(&[t]); if s.contains('\n') { break; } cont.push_str(&s); g.push(t); }
                let pred: Option<i64> = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
                tot += 1;
                let e = bylen.entry(len).or_insert((0, 0));
                e.1 += 1;
                if pred == Some(ans) { ok += 1; e.0 += 1; }
            }
            accs.push(100.0 * ok as f64 / tot.max(1) as f64);
        }
        let (m, s) = mean_std(&accs);
        let bl: Vec<String> = bylen.iter().map(|(l, (o, mm))| format!("{l}:{:.0}%", 100.0 * *o as f64 / (*mm).max(1) as f64)).collect();
        println!("  {name:<8}  {m:>4.0}% ± {s:<3.0}   {}", bl.join("  "));
    }
    println!("\n→ which RECURSIVE functions the model computes over LISTS (folds), not just numeric expressions;");
    println!("  acc-by-list-length = the list-depth cliff. These map onto textbook recursive Datalog (len/last/member/fold).");
}

/// DUMP (task, list, model-output, truth) JSONL for the list battery — the input to the offline bottom-up synthesizer
/// (RULE_EXTRACTION_PROPOSAL §8). The synthesizer fits a program to the MODEL'S output (faithful), so we dump the
/// model's actual answer per list; `truth` is the textbook value (for grading discovery, not a fit target).
pub fn run_list_dump(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --list-dump needs a tokenizer next to {stem}"); return; } };
    let path = match flag(args, "--list-dump") { Some(p) => p, None => { eprintln!("[fieldrun] --list-dump needs a path"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(120);
    let lmin: usize = 3;
    let lmax: usize = flag(args, "--lmax").and_then(|s| s.parse().ok()).unwrap_or(7);
    let mut rng: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
    type LFn = (&'static str, &'static str, fn(&[i64]) -> Option<i64>);
    let fns: [LFn; 6] = [
        ("last",  "last 3 7 2 5 = 5\nlast 1 8 4 = 4\nlast 6 0 9 2 = 2\n", |l| l.last().copied()),
        ("first", "first 3 7 2 5 = 3\nfirst 1 8 4 = 1\nfirst 6 0 9 2 = 6\n", |l| l.first().copied()),
        ("len",   "len 3 7 2 5 = 4\nlen 1 8 4 = 3\nlen 6 0 9 2 5 = 5\n", |l| Some(l.len() as i64)),
        ("max",   "max 3 7 2 5 = 7\nmax 1 8 4 = 8\nmax 6 0 9 2 = 9\n", |l| l.iter().max().copied()),
        ("min",   "min 3 7 2 5 = 2\nmin 1 8 4 = 1\nmin 6 0 9 2 = 0\n", |l| l.iter().min().copied()),
        ("sum",   "sum 3 1 2 = 6\nsum 4 0 1 = 5\nsum 2 3 1 = 6\n", |l| { let s: i64 = l.iter().sum(); (s <= 9).then_some(s) }),
    ];
    let mut out = String::new();
    let mut total = 0usize;
    eprintln!("[fieldrun] list-dump · {n} lists/task · len {lmin}..{lmax} → {path}");
    for (name, prime, truth) in fns {
        let (mut got, mut tries) = (0usize, 0usize);
        while got < n && tries < n * 40 {
            tries += 1;
            let len = lmin + (xorshift(&mut rng) % (lmax - lmin + 1) as u64) as usize;
            let list: Vec<i64> = (0..len).map(|_| (xorshift(&mut rng) % 10) as i64).collect();
            let tv = match truth(&list) { Some(a) if (0..=9).contains(&a) => a, _ => continue };
            let listing = list.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            let mut g = tg.encode(&format!("{prime}{name} {listing} ="), false);
            let mut cont = String::new();
            for _ in 0..3 { let t = lm.predict(&g); let s = tg.decode(&[t]); if s.contains('\n') { break; } cont.push_str(&s); g.push(t); }
            let mo: Option<i64> = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
            if let Some(mo) = mo {
                let ls = list.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                out.push_str(&format!("{{\"task\":\"{name}\",\"list\":[{ls}],\"out\":{mo},\"truth\":{tv}}}\n"));
                got += 1;
                total += 1;
            }
        }
    }
    match std::fs::write(path, &out) {
        Ok(_) => eprintln!("[fieldrun] wrote {total} records → {path}"),
        Err(e) => eprintln!("[fieldrun] cannot write {path}: {e}"),
    }
}

/// FAITHFUL cross-attribution: for each ASKED list-function, score what the model ACTUALLY computes against EVERY
/// candidate (incl. broken variants) — the best match is the model's real function, which may not be the one we named.
/// Faithfulness, not correctness: we fit the model's output, "wrong"-by-textbook included. Off-diagonal > diagonal =
/// the model is computing a DIFFERENT function than asked (the first whiff of discovery, within the named set).
pub fn run_list_attribute(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --list-attribute needs a tokenizer next to {stem}"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(40);
    let (lmin, lmax): (usize, usize) = (3, flag(args, "--lmax").and_then(|s| s.parse().ok()).unwrap_or(6));
    let mut rng: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
    type LFn = (&'static str, &'static str, fn(&[i64]) -> Option<i64>);
    // candidate set incl. a couple BROKEN variants (max-but-first, sum-mod-10) the model might actually be doing
    let cand: [LFn; 8] = [
        ("last",  "", |l| l.last().copied()),
        ("first", "", |l| l.first().copied()),
        ("len",   "", |l| Some(l.len() as i64)),
        ("max",   "", |l| l.iter().max().copied()),
        ("min",   "", |l| l.iter().min().copied()),
        ("sum",   "", |l| Some(l.iter().sum())),
        ("sum%10", "", |l| Some(l.iter().sum::<i64>() % 10)),       // broken: drops the carry
        ("2nd",   "", |l| l.get(1).copied()),                       // broken: off-by-one
    ];
    let asked: [LFn; 6] = [
        ("last",  "last 3 7 2 5 = 5\nlast 1 8 4 = 4\nlast 6 0 9 2 = 2\n", |l| l.last().copied()),
        ("first", "first 3 7 2 5 = 3\nfirst 1 8 4 = 1\nfirst 6 0 9 2 = 6\n", |l| l.first().copied()),
        ("len",   "len 3 7 2 5 = 4\nlen 1 8 4 = 3\nlen 6 0 9 2 5 = 5\n", |l| Some(l.len() as i64)),
        ("max",   "max 3 7 2 5 = 7\nmax 1 8 4 = 8\nmax 6 0 9 2 = 9\n", |l| l.iter().max().copied()),
        ("min",   "min 3 7 2 5 = 2\nmin 1 8 4 = 1\nmin 6 0 9 2 = 0\n", |l| l.iter().min().copied()),
        ("sum",   "sum 3 1 2 = 6\nsum 4 0 1 = 5\nsum 2 3 1 = 6\n", |l| { let s: i64 = l.iter().sum(); (s <= 9).then_some(s) }),
    ];
    eprintln!("[fieldrun] list cross-attribution · {n} lists/asked-fn · {stem}");
    println!("# list cross-attribution — what function is the model ACTUALLY computing? ({stem})");
    let hdr: Vec<&str> = cand.iter().map(|c| c.0).collect();
    println!("  asked\\fits   {}", hdr.iter().map(|h| format!("{h:>6}")).collect::<Vec<_>>().join(""));
    for (aname, prime, _) in asked {
        let mut io: Vec<(Vec<i64>, i64)> = Vec::new();
        let mut tries = 0;
        while io.len() < n && tries < n * 30 {
            tries += 1;
            let len = lmin + (xorshift(&mut rng) % (lmax - lmin + 1) as u64) as usize;
            let list: Vec<i64> = (0..len).map(|_| (xorshift(&mut rng) % 10) as i64).collect();
            let listing = list.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            let mut g = tg.encode(&format!("{prime}{aname} {listing} ="), false);
            let mut cont = String::new();
            for _ in 0..3 { let t = lm.predict(&g); let s = tg.decode(&[t]); if s.contains('\n') { break; } cont.push_str(&s); g.push(t); }
            if let Ok(v) = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<i64>() {
                io.push((list, v));
            }
        }
        let scores: Vec<f64> = cand.iter().map(|(_, _, f)| {
            let (mut m, mut d) = (0usize, 0usize);
            for (l, obs) in &io { if let Some(v) = f(l) { d += 1; if v == *obs { m += 1; } } }
            if d > 0 { 100.0 * m as f64 / d as f64 } else { 0.0 }
        }).collect();
        let best = scores.iter().cloned().fold(0.0, f64::max);
        let row: Vec<String> = scores.iter().zip(&cand).map(|(s, c)| {
            let mark = if *s == best && *s > 0.0 { "*" } else { " " };
            let _ = c;
            format!("{s:>5.0}{mark}")
        }).collect();
        println!("  {aname:<10}  {}", row.join(""));
    }
    println!("\n→ each row = the model's outputs when ASKED that function, scored against every candidate (* = best fit).");
    println!("  diagonal best = model computes what we asked; OFF-diagonal best = it's doing a DIFFERENT/broken function.");
}

/// A matched variable-binding pair: `a=3 b=7 c=2 a=` (query the FIRST var). A and B bind the SAME variables to the SAME
/// values EXCEPT the queried var, which differs — so the model must HOLD the queried binding across the others and
/// RETRIEVE it. No deep computation → not depth-limited. Returns (strA, strB, ansA, ansB). The two differ at exactly one
/// token (the queried var's value), so positions align and the diff position is the binding site.
pub fn gen_bind_pair(nvars: usize, r: &mut u64) -> Option<(String, String, i64, i64)> {
    let letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    let nv = nvars.clamp(2, letters.len());
    for _ in 0..2000 {
        // distinct letters
        let mut ls: Vec<char> = letters.to_vec();
        for i in (1..ls.len()).rev() { let j = (xorshift(r) % (i as u64 + 1)) as usize; ls.swap(i, j); }
        let vars = &ls[..nv];
        // distinct values 0..9
        let mut vals: Vec<i64> = Vec::new();
        let mut tries = 0;
        while vals.len() < nv && tries < 200 { tries += 1; let v = (xorshift(r) % 10) as i64; if !vals.contains(&v) { vals.push(v); } }
        if vals.len() < nv { continue; }
        let va = vals[0];
        let vb = match (0..10).map(|_| (xorshift(r) % 10) as i64).find(|x| *x != va && !vals[1..].contains(x)) { Some(x) => x, None => continue };
        let body = |v0: i64| -> String {
            let mut parts: Vec<String> = vars.iter().zip(&vals).map(|(c, &v)| format!("{c}={v}")).collect();
            parts[0] = format!("{}={v0}", vars[0]);
            format!("{} {}=", parts.join(" "), vars[0])
        };
        return Some((body(va), body(vb), va, vb));
    }
    None
}

/// VARIABLE-BINDING RETRIEVAL causal test (consuming-context test #2; depth-free). Matched pairs differ only in the
/// queried variable's bound value; patch A's residual at the BINDING SITE (the one differing token) into B at each
/// layer and see if B's RETRIEVED answer flips A→. A high flip rate = the binding site causally carries the held value
/// into retrieval (held-for-retrieval, the consumer/retrieval circuit fires) — the test the arithmetic hold-sweep
/// couldn't run because of the depth ceiling.
pub fn run_bind_patch(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --bind-patch needs a tokenizer next to {stem}"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(50);
    let nvars: usize = flag(args, "--nvars").and_then(|s| s.parse().ok()).unwrap_or(3);
    let base_seed: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15);
    let seeds: usize = flag(args, "--seeds").and_then(|s| s.parse().ok()).unwrap_or(1);
    let layers: Vec<usize> = match flag(args, "--layers") {
        Some(s) => s.split(',').filter_map(|x| x.parse().ok()).collect(),
        None => vec![0, 4, 8, 12, 16, 20, 24],
    };
    const PRIME: &str = "k=2 t=9 k=2\nr=6 s=1 r=6\nm=4 n=7 p=3 m=4\n";
    let gen_ans = |ids: &[i64], patch: Option<(usize, &Vec<usize>, &Vec<Vec<f32>>)>| -> Option<i64> {
        let mut g = ids.to_vec();
        let mut cont = String::new();
        for _ in 0..3 {
            let t = match patch { Some((l, ps, ds)) => lm.predict_patched(&g, l, ps, ds)?, None => lm.predict(&g) };
            let s = tg.decode(&[t]);
            if s.contains('\n') { break; }
            cont.push_str(&s);
            g.push(t);
        }
        cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
    };
    let mut layer_rates: std::collections::BTreeMap<usize, Vec<f64>> = layers.iter().map(|&l| (l, Vec::new())).collect();
    let mut base_rates: Vec<f64> = Vec::new();
    eprintln!("[fieldrun] bind-patch · {n} pairs × {seeds} seeds · {nvars} vars · layers {layers:?} · {stem}");
    for si in 0..seeds {
    let mut rng: u64 = (base_seed ^ (si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    let (mut total, mut base_ok) = (0usize, 0usize);
    let mut hits: std::collections::BTreeMap<usize, usize> = layers.iter().map(|&l| (l, 0)).collect();
    for _ in 0..n {
        let (sa, sb, ansa, ansb) = match gen_bind_pair(nvars, &mut rng) { Some(x) => x, None => continue };
        let aids = tg.encode(&format!("{PRIME}{sa}"), false);
        let bids = tg.encode(&format!("{PRIME}{sb}"), false);
        if aids.len() != bids.len() { continue; }
        // the binding site = the single token where A and B differ (the queried var's value)
        let diff: Vec<usize> = (0..aids.len()).filter(|&i| aids[i] != bids[i]).collect();
        if diff.len() != 1 { continue; }
        let pos = vec![diff[0]];
        let ares = match lm.residuals_at(&aids, &pos) { Some(r) => r, None => { eprintln!("[fieldrun] arch has no residuals_at"); return; } };
        total += 1;
        if gen_ans(&bids, None) == Some(ansb) { base_ok += 1; }
        for &l in &layers {
            if l >= ares[0].len() || ares[0][l].is_empty() { continue; }
            let donor = vec![ares[0][l].clone()];
            if gen_ans(&bids, Some((l, &pos, &donor))) == Some(ansa) { *hits.get_mut(&l).unwrap() += 1; }
        }
    }
    let pc = |x: usize| 100.0 * x as f64 / total.max(1) as f64;
    for &l in &layers { layer_rates.get_mut(&l).unwrap().push(pc(*hits.get(&l).unwrap())); }
    base_rates.push(pc(base_ok));
    }
    let (bm, bs) = mean_std(&base_rates);
    println!("# bind-patch — VARIABLE-BINDING retrieval, causal ({stem}) · n={n}/seed × {seeds} seeds · {nvars} vars");
    println!("# baseline: B retrieves its OWN value {bm:.0}% ± {bs:.0} (sd over seeds)");
    println!("  layer   B→A retrieval (binding-site causal · mean% ± sd over {seeds} seeds)");
    for &l in &layers {
        let (m, s) = mean_std(layer_rates.get(&l).unwrap());
        println!("  {l:>4}        {m:>5.0}% ± {s:<4.0}");
    }
    println!("\n→ HIGH = patching the binding site flips the RETRIEVED value → held binding causally retrieved; ± = sd across {seeds} seeds.");
}

/// CONSUMING-CONTEXT test (user: "not a spectator in OTHER evaluations of the same circuit"). Vary the HOLD DISTANCE
/// (right-subtree depth): does patching the left value at its CLOSE position become causal as the left value must be
/// carried farther before the root consumes it? Rising close-causality with hold = the consumer/retrieval circuit
/// firing WHEN NEEDED (a held value, not a spectator). Flat-null = the close summary is never the carrier.
pub fn run_hold_sweep(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --hold-sweep needs a tokenizer next to {stem}"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(40);
    let hmax: usize = flag(args, "--hmax").and_then(|s| s.parse().ok()).unwrap_or(4);
    let layer: usize = flag(args, "--layer").and_then(|s| s.parse().ok()).unwrap_or(18);
    let mut rng: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
    const PRIME: &str = "(+ 2 3) = 5\n(* 2 4) = 8\n(- 9 4) = 5\n(+ (* 2 3) 1) = 7\n";
    let gen_ans = |ids: &[i64], patch: Option<(usize, &Vec<usize>, &Vec<Vec<f32>>)>| -> Option<i64> {
        let mut g = ids.to_vec();
        let mut cont = String::new();
        for _ in 0..4 {
            let t = match patch { Some((l, ps, ds)) => lm.predict_patched(&g, l, ps, ds)?, None => lm.predict(&g) };
            let s = tg.decode(&[t]);
            if s.contains('\n') { break; }
            cont.push_str(&s);
            g.push(t);
        }
        cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
    };
    println!("# hold-sweep — does the held left value become CAUSAL at its close as hold distance grows? ({stem})");
    println!("# patch the LEFT-CLOSE residual at layer {layer} into B; right subtree = depth-`hold` + chain");
    println!("  hold   right-depth   close-only causal   baseline(B-self)");
    eprintln!("[fieldrun] hold-sweep · holds 1..{hmax} · {n} pairs each · patch layer {layer} · {stem}");
    for hold in 1..=hmax {
        let (mut total, mut base_ok, mut hit) = (0usize, 0usize, 0usize);
        for _ in 0..n {
            let (ea, eb, ansa, ansb) = match gen_hold_pair(hold, &mut rng) { Some(x) => x, None => continue };
            let aids = tg.encode(&format!("{PRIME}{ea} ="), false);
            let bids = tg.encode(&format!("{PRIME}{eb} ="), false);
            let atoms = atomize_ids(tg, &bids);
            let tree = match recursion_dl::parse_target(&atoms) { Some(t) => t, None => continue };
            let mut ops_v: Vec<&recursion_dl::Node> = Vec::new();
            collect_ops(&tree, &mut ops_v);
            if ops_v.len() < 2 { continue; }
            let left_close = match ops_v[1] { recursion_dl::Node::Op(.., c) => *c, _ => continue };
            let ares = match lm.residuals_at(&aids, &[left_close]) { Some(r) => r, None => { eprintln!("[fieldrun] arch has no residuals_at"); return; } };
            if layer >= ares[0].len() || ares[0][layer].is_empty() { continue; }
            let pos = vec![left_close];
            let donor = vec![ares[0][layer].clone()];
            total += 1;
            if gen_ans(&bids, None) == Some(ansb) { base_ok += 1; }
            if gen_ans(&bids, Some((layer, &pos, &donor))) == Some(ansa) { hit += 1; }
        }
        let pc = |x: usize| 100.0 * x as f64 / total.max(1) as f64;
        println!("  {hold:>4}   {hold:>11}   {:>14.0}%   {:>14.0}%", pc(hit), pc(base_ok));
    }
    println!("\n→ RISING close-causality with hold = the held value's CONSUMER circuit fires when the value must be carried");
    println!("  far (held-for-retrieval, not a spectator). FLAT-null across holds = the close summary is never the carrier.");
}

/// B2 CAUSAL test — interchange intervention. For matched pairs (A,B) differing only in the LEFT value, capture A's
/// residual at the left-child position and patch it into B's forward at each candidate layer; if B's OUTPUT flips to
/// A's answer, that (layer, position) causally CARRIES the value. Reports causal hit-rate vs layer (should peak where
/// the probe read the value). This is what upgrades "decodable" → "the model computes WITH it".
pub fn run_value_patch(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --value-patch needs a tokenizer next to {stem}"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(40);
    let base_seed: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15);
    let seeds: usize = flag(args, "--seeds").and_then(|s| s.parse().ok()).unwrap_or(1);
    let layers: Vec<usize> = match flag(args, "--layers") {
        Some(s) => s.split(',').filter_map(|x| x.parse().ok()).collect(),
        None => vec![4, 8, 12, 16, 18, 20, 24],
    };
    const PRIME: &str = "(+ 2 3) = 5\n(* 2 4) = 8\n(- 9 4) = 5\n(+ (* 2 3) 1) = 7\n";
    let span_mode = crate::has_flag(args, "--patch-span");
    let gen_ans = |ids: &[i64], patch: Option<(usize, &Vec<usize>, &Vec<Vec<f32>>)>| -> Option<i64> {
        let mut g = ids.to_vec();
        let mut cont = String::new();
        for _ in 0..4 {
            let t = match patch { Some((l, ps, ds)) => lm.predict_patched(&g, l, ps, ds)?, None => lm.predict(&g) };
            let s = tg.decode(&[t]);
            if s.contains('\n') { break; }
            cont.push_str(&s);
            g.push(t);
        }
        cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
    };
    let mut layer_rates: std::collections::BTreeMap<usize, Vec<f64>> = layers.iter().map(|&l| (l, Vec::new())).collect();
    let mut base_rates: Vec<f64> = Vec::new();
    eprintln!("[fieldrun] value-patch · {n} pairs × {seeds} seeds · layers {layers:?} · patch={} · {stem}", if span_mode { "LEFT-SPAN" } else { "left-close" });
    for si in 0..seeds {
    let mut rng: u64 = (base_seed ^ (si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    let (mut total, mut base_ok) = (0usize, 0usize);
    let mut hits: std::collections::BTreeMap<usize, usize> = layers.iter().map(|&l| (l, 0)).collect();
    for _ in 0..n {
        let (ea, eb, ansa, ansb, _la, _lb) = match gen_patch_pair(&mut rng) { Some(x) => x, None => continue };
        let aids = tg.encode(&format!("{PRIME}{ea} ="), false);
        let bids = tg.encode(&format!("{PRIME}{eb} ="), false);
        let atoms = atomize_ids(tg, &bids);
        let tree = match recursion_dl::parse_target(&atoms) { Some(t) => t, None => continue };
        let mut ops_v: Vec<&recursion_dl::Node> = Vec::new();
        collect_ops(&tree, &mut ops_v);
        if ops_v.len() < 3 { continue; }
        let (lopen, lclose) = match ops_v[1] { recursion_dl::Node::Op(_, _, _, o, c) => (*o, *c), _ => continue };
        // positions to patch: the whole left subtree span (operator+operands+close) or just the close
        let positions: Vec<usize> = if span_mode { (lopen..=lclose).collect() } else { vec![lclose] };
        let ares = match lm.residuals_at(&aids, &positions) { Some(r) => r, None => { eprintln!("[fieldrun] arch has no residuals_at"); return; } };
        let b_base = gen_ans(&bids, None);
        total += 1;
        if b_base == Some(ansb) { base_ok += 1; }
        for &l in &layers {
            let donors: Vec<Vec<f32>> = ares.iter().map(|per_layer| per_layer.get(l).cloned().unwrap_or_default()).collect();
            if donors.iter().any(|d| d.is_empty()) { continue; }
            if gen_ans(&bids, Some((l, &positions, &donors))) == Some(ansa) {
                *hits.get_mut(&l).unwrap() += 1;
            }
        }
    }
    let pc = |x: usize| 100.0 * x as f64 / total.max(1) as f64;
    for &l in &layers { layer_rates.get_mut(&l).unwrap().push(pc(*hits.get(&l).unwrap())); }
    base_rates.push(pc(base_ok));
    }
    let (bm, bs) = mean_std(&base_rates);
    println!("# value-patch — CAUSAL interchange ({stem}) · n={n}/seed × {seeds} seeds · patch={}", if span_mode { "left-span" } else { "left-close" });
    println!("# baseline: B computes its OWN answer {bm:.0}% ± {bs:.0} (sd over seeds)");
    println!("  layer   B→A-answer (causal carry · mean% ± sd over {seeds} seeds)");
    for &l in &layers {
        let (m, s) = mean_std(layer_rates.get(&l).unwrap());
        println!("  {l:>4}        {m:>5.0}% ± {s:<4.0}");
    }
    println!("\n→ HIGH at a layer = patching there makes B output A's answer (causal carry); ± = sd across {seeds} seeds (n={n} pairs each).");
}

/// B2 — supervised value-probe DUMP. For many depth-2 exprs, capture the per-layer residual at each subtree node's
/// close position together with that subtree's TRUE value, and write a binary file for offline linear-probe training
/// (Python). Tests whether an intermediate value is LINEARLY decodable from the residual where the lens basis can't
/// read it. Binary layout (LE): magic "FRVP", u32 n_samples, u32 n_layers, u32 d, then per sample:
/// i32 value, u8 role(0=root,1=left,2=right), u8 model_correct, then n_layers*d f32 residuals.
pub fn run_value_probe_dump(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
    let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --value-probe-dump needs a tokenizer next to {stem}"); return; } };
    let out_path = match flag(args, "--value-probe-dump") { Some(p) => p, None => { eprintln!("[fieldrun] --value-probe-dump needs a path"); return; } };
    let n: usize = flag(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(300);
    let mut maxv: i64 = flag(args, "--maxv").and_then(|s| s.parse().ok()).unwrap_or(9);
    let mut rng: u64 = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
    let family = flag(args, "--family").unwrap_or("arith");
    let (ops, prime): (&[char], &str) = match family {
        "minmax" => (&['<', '>'], "(> 4 9) = 9\n(< 4 9) = 4\n(> 8 2) = 8\n(< 8 2) = 2\n(> (< 4 9) 6) = 6\n"),
        "bool" => (&['&', '|'], "(& 1 0) = 0\n(| 1 0) = 1\n(& 1 1) = 1\n(| 0 0) = 0\n(| (& 1 0) 1) = 1\n"),
        _ => (&['+', '*', '-'], "(+ 2 3) = 5\n(* 2 4) = 8\n(- 9 4) = 5\n(+ (* 2 3) 1) = 7\n"),
    };
    if family == "bool" { maxv = 1; }

    let mut samples: Vec<(i32, u8, u8, Vec<f32>)> = Vec::new();
    let (mut n_layers, mut d) = (0usize, 0usize);
    eprintln!("[fieldrun] value-probe dump · {n} depth-2 {family} exprs → {out_path}");
    for _ in 0..n {
        let expr = match gen_depth2(ops, maxv, &mut rng) { Some(e) => e, None => continue };
        let ids = tg.encode(&format!("{prime}{expr} ="), false);
        let mut gids = ids.clone();
        let mut cont = String::new();
        for _ in 0..4 { let t = lm.predict(&gids); let s = tg.decode(&[t]); if s.contains('\n') { break; } cont.push_str(&s); gids.push(t); }
        let model_ans: Option<i64> = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
        let mut atoms: Vec<(String, usize)> = Vec::new();
        for (ti, &id) in ids.iter().enumerate() {
            let mut num = String::new();
            for ch in tg.decode(&[id]).chars() {
                if ch.is_ascii_digit() { num.push(ch); }
                else { if !num.is_empty() { atoms.push((std::mem::take(&mut num), ti)); }
                       if matches!(ch, '(' | ')' | '+' | '-' | '*' | '/' | '&' | '|' | '^' | '<' | '>' | '%') { atoms.push((ch.to_string(), ti)); } }
            }
            if !num.is_empty() { atoms.push((num, ti)); }
        }
        let tree = match recursion_dl::parse_target(&atoms) { Some(t) => t, None => continue };
        let truev = true_eval(&tree);
        let correct = (model_ans.is_some() && model_ans == truev) as u8;
        let mut ops_v: Vec<&recursion_dl::Node> = Vec::new();
        collect_ops(&tree, &mut ops_v);
        if ops_v.len() < 3 { continue; }
        let closes: Vec<usize> = ops_v.iter().take(3).filter_map(|nd| if let recursion_dl::Node::Op(.., c) = nd { Some(*c) } else { None }).collect();
        let resids = match lm.residuals_at(&ids, &closes) { Some(r) => r, None => { eprintln!("[fieldrun] arch has no residuals_at (rope family only)"); return; } };
        for (i, node) in ops_v.iter().take(3).enumerate() {
            let v = match true_eval(node) { Some(v) => v, None => continue };
            let per_layer = &resids[i];
            if n_layers == 0 { n_layers = per_layer.len(); d = per_layer.iter().map(|l| l.len()).max().unwrap_or(0); }
            if d == 0 { continue; }
            let mut flat = Vec::with_capacity(n_layers * d);
            for l in per_layer { if l.len() == d { flat.extend_from_slice(l); } else { flat.resize(flat.len() + d, 0.0); } }
            samples.push((v as i32, i as u8, correct, flat));
        }
    }

    use std::io::Write;
    let f = match std::fs::File::create(out_path) { Ok(f) => f, Err(e) => { eprintln!("[fieldrun] cannot write {out_path}: {e}"); return; } };
    let mut w = std::io::BufWriter::new(f);
    let _ = w.write_all(b"FRVP");
    let _ = w.write_all(&(samples.len() as u32).to_le_bytes());
    let _ = w.write_all(&(n_layers as u32).to_le_bytes());
    let _ = w.write_all(&(d as u32).to_le_bytes());
    for (v, role, correct, flat) in &samples {
        let _ = w.write_all(&v.to_le_bytes());
        let _ = w.write_all(&[*role, *correct]);
        for x in flat { let _ = w.write_all(&x.to_le_bytes()); }
    }
    let _ = w.flush();
    eprintln!("[fieldrun] wrote {} samples × {n_layers} layers × {d} dims → {out_path}", samples.len());
}

// ===== extracted --recursion-explain sub-mode handlers =====

/// Extracted from main.rs `--recursion-explain` dispatch (see module docs).
pub fn run_measure(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
        let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --measure needs a tokenizer next to {stem}"); return; } };
        let n_per: usize = flag(&args, "--n").and_then(|s| s.parse().ok()).unwrap_or(8);
        let dmax: usize = flag(&args, "--dmax").and_then(|s| s.parse().ok()).unwrap_or(6);
        let mut maxv: i64 = flag(&args, "--maxv").and_then(|s| s.parse().ok()).unwrap_or(9);
        let mut rng: u64 = flag(&args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
        // operator FAMILY — gauge which recursive functions the model can do (arith is native; others test ICL+compute).
        let family = flag(&args, "--family").unwrap_or("arith");
        let (ops, mprime): (&[char], &str) = match family {
            "minmax" => (&['<', '>'], "(> 4 9) = 9\n(< 4 9) = 4\n(> 8 2) = 8\n(< 8 2) = 2\n(> (< 4 9) 6) = 6\n(< (> 1 3) 2) = 2\n"),
            "bool" => (&['&', '|'], "(& 1 0) = 0\n(| 1 0) = 1\n(& 1 1) = 1\n(| 0 0) = 0\n(| (& 1 0) 1) = 1\n(& (| 0 1) 0) = 0\n"),
            "mod" => (&['%', '+'], "(% 7 3) = 1\n(% 8 4) = 0\n(% 9 2) = 1\n(% (+ 5 4) 3) = 0\n(+ (% 7 3) 2) = 3\n"),
            "addonly" => (&['+'], "(+ 1 2) = 3\n(+ 3 4) = 7\n(+ (+ 1 2) 3) = 6\n(+ 2 (+ 1 1)) = 4\n"),
            _ => (&['+', '-', '*'], "(+ 1 2) = 3\n(* 2 3) = 6\n(- 9 4) = 5\n(* 2 (+ 1 2)) = 6\n(- 8 (+ 1 2)) = 5\n"),
        };
        if family == "bool" { maxv = 1; }
        let prime_lits: Vec<i64> = mprime.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse::<i64>().ok()).collect();
        #[derive(Clone, Default)]
        struct Agg { n: usize, correct: usize, faithful: usize, clean: usize, cut: usize, semi: usize, sumd: i64, err: usize, err_cut: usize }
        let mut per: Vec<Agg> = vec![Agg::default(); dmax + 1];
        eprintln!("[fieldrun] datalog measure · family={family} · {n_per} exprs/depth × depths 1..{dmax} · maxv {maxv} · {stem}");
        for depth in 1..=dmax {
            for _ in 0..n_per {
                let (expr, truev) = gen_family(depth, ops, maxv, &mut rng);
                let mut ids = tg.encode(&format!("{mprime}{expr} ="), false);
                let mut cont = String::new();
                for _ in 0..4 {
                    let t = lm.predict(&ids);
                    let s = tg.decode(&[t]);
                    if s.contains('\n') { break; }
                    cont.push_str(&s);
                    ids.push(t);
                }
                let num: String = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
                let model_answer = match num.parse::<i64>() { Ok(v) => v, Err(_) => continue };
                let tree = match recursion_dl::parse_str(&expr) { Some(t) => t, None => continue };
                let mut lits = prime_lits.clone();
                lits.extend(expr.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse::<i64>().ok()));
                let (abd, maxd, _) = recursion_dl::analyze(&tree, model_answer, &lits);
                let a = &mut per[depth];
                a.n += 1;
                let ok = model_answer == truev;
                if ok { a.correct += 1; } else { a.err += 1; }
                match abd {
                    Some(ab) => {
                        a.faithful += 1;
                        a.sumd += ab.depth;
                        if ab.cuts.is_empty() && ab.depth as usize == maxd { a.clean += 1; }
                        else {
                            a.cut += 1;
                            if !ok && ab.cuts.iter().any(|&(_, _, ctx)| ctx) { a.err_cut += 1; }
                        }
                    }
                    None => a.semi += 1,
                }
            }
        }
        let pct = |x: usize, n: usize| if n > 0 { 100.0 * x as f64 / n as f64 } else { 0.0 };
        println!("# datalog measure — depth-bounded abductive faithfulness ({stem})");
        println!("depth   n  model-acc  faithful  meanD  clean  cut  semiring");
        let mut tot = Agg::default();
        let mut dstar = 0usize;
        for depth in 1..=dmax {
            let a = per[depth].clone();
            if a.n == 0 { continue; }
            if pct(a.correct, a.n) >= 50.0 { dstar = dstar.max(depth); }
            println!("{:>5} {:>3} {:>8.0}% {:>8.0}% {:>6.1} {:>6} {:>4} {:>8}",
                     depth, a.n, pct(a.correct, a.n), pct(a.faithful, a.n),
                     if a.faithful > 0 { a.sumd as f64 / a.faithful as f64 } else { 0.0 }, a.clean, a.cut, a.semi);
            tot.n += a.n; tot.correct += a.correct; tot.faithful += a.faithful;
            tot.clean += a.clean; tot.cut += a.cut; tot.semi += a.semi; tot.err += a.err; tot.err_cut += a.err_cut;
        }
        println!("\n→ model accuracy {:.0}% · FAITHFULNESS (abduction reproduces the model) {:.0}% of {} queries",
                 pct(tot.correct, tot.n), pct(tot.faithful, tot.n), tot.n);
        println!("→ split: {:.0}% clean recursion · {:.0}% broken-cut (early-cut retrieval) · {:.0}% semiring-needed (no depth-cut found)",
                 pct(tot.clean, tot.n), pct(tot.cut, tot.n), pct(tot.semi, tot.n));
        println!("→ effective recursion depth D* = {dstar} (deepest where model-acc ≥ 50%)  [cross-check vs the recursion-depth probe]");
        println!("→ P1 (errs as depth exceeds D*): see the model-acc cliff in the table above");
        println!("→ P2 (a wrong answer == a context-literal cut): {:.0}% of {} errors explained by a retrieved cut", pct(tot.err_cut, tot.err), tot.err);
        return;
}


/// Extracted from main.rs `--recursion-explain` dispatch (see module docs).
pub fn run_discover(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
        let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --discover needs a tokenizer next to {stem}"); return; } };
        type Op = (char, &'static str, fn(i64, i64) -> Option<i64>);
        let basis: [Op; 7] = [
            ('+', "add", |a, b| Some(a + b)),
            ('-', "sub", |a, b| (a >= b).then_some(a - b)),
            ('*', "mul", |a, b| Some(a * b)),
            ('/', "div", |a, b| (b != 0 && a % b == 0).then(|| a / b)),
            ('<', "min", |a, b| Some(a.min(b))),
            ('>', "max", |a, b| Some(a.max(b))),
            ('%', "mod", |a, b| (b != 0).then(|| a % b)),
        ];
        let sym = flag(&args, "--sym").and_then(|s| s.chars().next()).unwrap_or('@');
        let teach_c = flag(&args, "--teach").and_then(|s| s.chars().next()).unwrap_or('+');
        let teach: Op = basis.iter().copied().find(|b| b.0 == teach_c).unwrap_or(basis[0]);
        let maxv: i64 = flag(&args, "--maxv").and_then(|s| s.parse().ok()).unwrap_or(9);
        let probe_n: usize = flag(&args, "--probe-n").and_then(|s| s.parse().ok()).unwrap_or(60);
        let verify_n: usize = flag(&args, "--verify-n").and_then(|s| s.parse().ok()).unwrap_or(8);
        let mut rng: u64 = flag(&args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x2545_F491_4F6C_DD1D) | 1;

        // the model's answer to a prompt, as a leading integer (greedy, a few tokens — identical to --measure)
        let ask = |q: &str| -> Option<i64> {
            let mut ids = tg.encode(q, false);
            let mut cont = String::new();
            for _ in 0..4 {
                let t = lm.predict(&ids);
                let s = tg.decode(&[t]);
                if s.contains('\n') { break; }
                cont.push_str(&s);
                ids.push(t);
            }
            let num: String = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
            num.parse::<i64>().ok()
        };

        // TEACH: build a few-shot prime defining `sym` by EXAMPLE (the model in-context-learns it; the
        // induction below never sees `teach`, only the model's answers).
        let mut prime = String::new();
        { let (tf, mut k, mut tries) = (teach.2, 0, 0);
          while k < 8 && tries < 2000 { tries += 1;
              let a = (xorshift(&mut rng) % (maxv as u64 + 1)) as i64;
              let b = (xorshift(&mut rng) % (maxv as u64 + 1)) as i64;
              if let Some(v) = tf(a, b) { if (0..=maxv).contains(&v) {
                  prime.push_str(&format!("({sym} {a} {b}) = {v}\n")); k += 1; } } } }

        eprintln!("[fieldrun] discover · symbol '{sym}' taught as <{}> · probe {probe_n} flat + verify {verify_n}/depth · {stem}", teach.1);

        // 1. PROBE: each flat (sym a b) answer is a direct observation of apply(a,b) — no logit-lens, no priors.
        let mut triples: Vec<(i64, i64, i64)> = Vec::new();
        for _ in 0..probe_n {
            let a = (xorshift(&mut rng) % (maxv as u64 + 1)) as i64;
            let b = (xorshift(&mut rng) % (maxv as u64 + 1)) as i64;
            if let Some(v) = ask(&format!("{prime}({sym} {a} {b}) =")) { triples.push((a, b, v)); }
        }

        // 2. INDUCE: rank basis operators by how much of the OBSERVED table each reproduces (where defined).
        let mut ranked: Vec<(char, &str, usize, usize)> = basis.iter().map(|&(c, name, f)| {
            let (mut m, mut d) = (0usize, 0usize);
            for &(a, b, obs) in &triples { if let Some(v) = f(a, b) { d += 1; if v == obs { m += 1; } } }
            (c, name, m, d)
        }).collect();
        let frac = |m: usize, d: usize| if d > 0 { m as f64 / d as f64 } else { 0.0 };
        ranked.sort_by(|x, y| frac(y.2, y.3).partial_cmp(&frac(x.2, x.3)).unwrap());
        let (disc_c, disc_name, dm, dd) = ranked[0];
        let inducible = frac(dm, dd) >= 0.95 && dd >= triples.len() / 2;

        println!("# discover — induce a recursive function from behavior alone ({stem})");
        println!("symbol '{sym}'  ·  {} flat observations  ·  candidate operators ranked by table match:", triples.len());
        for &(c, name, m, d) in ranked.iter() {
            println!("    {c} {:<4} {:>5.0}%  ({m}/{d} defined)", name, 100.0 * frac(m, d));
        }

        // 3. VERIFY: does the DISCOVERED recursive Datalog reproduce the model on held-out NESTED expressions?
        let (mut nver, mut faith, mut clean) = (0usize, 0usize, 0usize);
        let mut sample: Option<(String, i64, Vec<i64>)> = None;
        if inducible {
            for depth in 2..=4 {
                for _ in 0..verify_n {
                    if let Some((expr, _)) = gen_op(depth, sym, teach.2, maxv, &mut rng) {
                        if let Some(ma) = ask(&format!("{prime}{expr} =")) {
                            let rexpr = expr.replace(sym, &disc_c.to_string()); // use the DISCOVERED rule
                            if let Some(tree) = recursion_dl::parse_str(&rexpr) {
                                let lits: Vec<i64> = expr.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse().ok()).collect();
                                let (abd, maxd, _) = recursion_dl::analyze(&tree, ma, &lits);
                                nver += 1;
                                if let Some(ab) = abd { faith += 1;
                                    if ab.cuts.is_empty() && ab.depth as usize == maxd { clean += 1; } }
                                if sample.is_none() { sample = Some((rexpr, ma, lits)); }
                            }
                        }
                    }
                }
            }
        }

        let graded = if disc_c == teach.0 { "✓ matches the taught operator" } else { "✗ DISCOVERED A DIFFERENT OP than taught" };
        println!();
        if inducible {
            println!("→ DISCOVERED: '{sym}' ≡ <{disc_name}> at {:.0}% table match  [{graded}]", 100.0 * frac(dm, dd));
            println!("→ converted to a RECURSIVE RULE: eval(N,V,B) :- node(N,\"{disc_c}\",A,C), eval(A,X,B-1), eval(C,Y,B-1), {}.",
                     recursion_dl::op_rhs(disc_c).unwrap_or("..."));
            println!("→ held-out NESTED faithfulness {:.0}% of {nver}  ·  clean (rule alone, no cut) {:.0}%  ·  rest = depth-bounded cut",
                     100.0 * frac(faith, nver), 100.0 * frac(clean, nver));
        } else {
            println!("→ NO closed-form operator fits (best <{disc_name}> only {:.0}%) — keep the OBSERVED table as EDB facts:", 100.0 * frac(dm, dd));
            println!("→   .decl apply(op:symbol, x:number, y:number, z:number)   // the residue stays legible-but-tabular, not a rule");
            println!("→ this is the 'here-be-dragons' boundary: structure detected, semantics not yet a rule (graceful degradation).");
        }
        if let Some((rexpr, ma, lits)) = sample {
            if let Some(tree) = recursion_dl::parse_str(&rexpr) {
                println!("\n# example DISCOVERED-operator Datalog (one nested query, faithful to the model):");
                print!("{}", recursion_dl::emit(&tree, Some(ma), &lits));
            }
        }
        return;
}


/// Extracted from main.rs `--recursion-explain` dispatch (see module docs).
pub fn run_induce(args: &[String], lm: &dyn crate::model::Model, tg: &Option<crate::api::TextGen>, stem: &str) {
        let tg = match tg { Some(t) => t, None => { eprintln!("[fieldrun] --induce needs a tokenizer next to {stem}"); return; } };
        let n: usize = flag(&args, "--n").and_then(|s| s.parse().ok()).unwrap_or(40);
        let mut maxv: i64 = flag(&args, "--maxv").and_then(|s| s.parse().ok()).unwrap_or(9);
        let mut rng: u64 = flag(&args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
        // operator FAMILY: arith produces NEW magnitude values (suspected illegible); minmax/bool produce
        // SELECTED/COPIED token values (suspected legible). Same machinery, different value semantics.
        let family = flag(&args, "--family").unwrap_or("arith");
        let (ops, prime): (&[char], &str) = match family {
            "minmax" => (&['<', '>'], "(> 4 9) = 9\n(< 4 9) = 4\n(> 8 2) = 8\n(< 8 2) = 2\n(> (< 4 9) 6) = 6\n"),
            "bool" => (&['&', '|'], "(& 1 0) = 0\n(| 1 0) = 1\n(& 1 1) = 1\n(| 0 0) = 0\n(| (& 1 0) 1) = 1\n"),
            _ => (&['+', '*', '-'], "(+ 2 3) = 5\n(* 2 4) = 8\n(- 9 4) = 5\n(+ (* 2 3) 1) = 7\n"),
        };
        if family == "bool" { maxv = 1; }
        #[derive(Default, Clone, Copy)]
        struct Cnt { n: usize, textbook: usize, operand: usize, answer: usize, other: usize, none: usize }
        // agg[correct?0:1][position] — split the profile by whether the model's final answer was textbook-correct.
        // The "many partial algorithms" hypothesis predicts these two slices differ: a legible component when it
        // succeeds, a different (illegible / non-textbook) one when it fails.
        let mut agg = [[Cnt::default(); 3]; 2];
        let (mut correct, mut total) = (0usize, 0usize);
        eprintln!("[fieldrun] induce sweep · {n} depth-2 {family} exprs · descriptive value-flow · {stem}");
        for _ in 0..n {
            let expr = match gen_depth2(ops, maxv, &mut rng) { Some(e) => e, None => continue };
            let ids = tg.encode(&format!("{prime}{expr} ="), false);
            let mut gids = ids.clone();
            let mut cont = String::new();
            for _ in 0..4 { let t = lm.predict(&gids); let s = tg.decode(&[t]); if s.contains('\n') { break; } cont.push_str(&s); gids.push(t); }
            let model_ans: Option<i64> = cont.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok();
            let mut atoms: Vec<(String, usize)> = Vec::new();
            for (ti, &id) in ids.iter().enumerate() {
                let mut num = String::new();
                for ch in tg.decode(&[id]).chars() {
                    if ch.is_ascii_digit() { num.push(ch); }
                    else { if !num.is_empty() { atoms.push((std::mem::take(&mut num), ti)); }
                           if matches!(ch, '(' | ')' | '+' | '-' | '*' | '/' | '&' | '|' | '^' | '<' | '>' | '%') { atoms.push((ch.to_string(), ti)); } }
                }
                if !num.is_empty() { atoms.push((num, ti)); }
            }
            let tree = match recursion_dl::parse_target(&atoms) { Some(t) => t, None => continue };
            let mut ops_v: Vec<&recursion_dl::Node> = Vec::new();
            collect_ops(&tree, &mut ops_v);
            if ops_v.len() < 3 { continue; }
            // CHEAP value reads: only at the 3 node closes + the 2 tokens before each (value may settle pre-merge).
            let closes: Vec<usize> = ops_v.iter().take(3).filter_map(|n| if let recursion_dl::Node::Op(.., c) = n { Some(*c) } else { None }).collect();
            let mut cand: Vec<usize> = Vec::new();
            for &c in &closes { for off in 0..=2usize { cand.push(c.wrapping_sub(off)); } }
            cand.sort_unstable(); cand.dedup();
            let lens = match lm.recursion_lens_at(&ids, &cand) { Some(l) => l, None => continue };
            let posmap: std::collections::HashMap<usize, &Vec<(usize, i64)>> = cand.iter().cloned().zip(lens.iter()).collect();
            // filter out next-token-PREDICTION reads (a lens token that equals the ACTUAL next token at that
            // position is the model predicting the upcoming literal, NOT a held subtree value) — removes the
            // dominant logit-lens confound so any genuine value-stack signal can surface.
            let read_near = |pos: usize| -> Option<i64> {
                let mut best: Option<(i64, usize)> = None;
                for off in 0..=2usize {
                    let p = pos.wrapping_sub(off);
                    if let Some(reads) = posmap.get(&p) {
                        let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
                        for &(_, tok) in reads.iter() {
                            if ids.get(p + 1) == Some(&tok) { continue; } // next-token prediction, not a held value
                            if let Ok(v) = tg.decode(&[tok]).trim().parse::<i64>() { *counts.entry(v).or_insert(0) += 1; }
                        }
                        if let Some((v, c)) = counts.into_iter().max_by_key(|&(_, c)| c) {
                            if best.map(|(_, bc)| c > bc).unwrap_or(true) { best = Some((v, c)); }
                        }
                    }
                }
                best.map(|(v, _)| v)
            };
            let truev = true_eval(&tree);
            total += 1;
            let is_correct = model_ans.is_some() && model_ans == truev;
            if is_correct { correct += 1; }
            let bucket = if is_correct { 0 } else { 1 };
            for (i, node) in ops_v.iter().take(3).enumerate() {
                if let recursion_dl::Node::Op(.., close) = node {
                    let t = true_eval(node);
                    let mut leaves = Vec::new();
                    collect_leaves(node, &mut leaves);
                    let c = &mut agg[bucket][i];
                    c.n += 1;
                    match read_near(*close) {
                        None => c.none += 1,
                        Some(r) if Some(r) == t => c.textbook += 1,
                        Some(r) if leaves.contains(&r) => c.operand += 1,
                        Some(r) if Some(r) == model_ans => c.answer += 1,
                        Some(_) => c.other += 1,
                    }
                }
            }
        }
        println!("# induce — DESCRIPTIVE value-flow profile ({stem}) · family={family}");
        println!("# {total} depth-2 exprs · model accuracy {:.0}% · split by whether the model's OUTPUT was correct", 100.0 * correct as f64 / total.max(1) as f64);
        for (bucket, blabel) in [(0usize, "model CORRECT"), (1usize, "model WRONG")] {
            println!("\n  [{blabel}]");
            println!("  position    n   textbook  operand-copy  =answer  other  none");
            for (i, name) in ["root ", "left ", "right"].iter().enumerate() {
                let c = &agg[bucket][i];
                let p = |x: usize| 100.0 * x as f64 / c.n.max(1) as f64;
                println!("  {name}     {:>3}   {:>6.0}%   {:>9.0}%  {:>6.0}%  {:>4.0}% {:>4.0}%",
                         c.n, p(c.textbook), p(c.operand), p(c.answer), p(c.other), p(c.none));
            }
        }
        println!("\n→ if the CORRECT and WRONG slices have DIFFERENT profiles, the model is running a FAMILY of partial");
        println!("  algorithms (one legible component when it succeeds, others when it fails) — each an ensemble member.");
        println!("  Whatever doesn't read as a clean rule → the Datalog KERNEL backstop (faithful to OUTPUT regardless).");
        return;
}
