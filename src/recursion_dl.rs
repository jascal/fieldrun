//! recursion → recursive Datalog. Parse the arithmetic s-expression recovered from the recursion-explain token stream
//! into a tree, then emit a RUNNABLE, RECURSIVE, Soufflé-compatible Datalog program: `eval/2` is a least-fixpoint over
//! the parse tree — Datalog's recursion IS the model's recursive descent. Annotated with `model_value` (what the
//! network held at each node, from the logit-lens value stack) so the symbolic recursion can be checked against the
//! trace (`faithful` / `divergent`). This is the symbolic counterpart of the deep trace — the recursion as a program.

/// A parsed arithmetic expression. `open`/`close` are the source TOKEN indices of an Op's brackets (for looking up the
/// model's value readout over that span).
pub enum Node {
    Leaf(i64),
    Op(char, Box<Node>, Box<Node>, usize, usize),
}

/// Parse ONE s-expression from `atoms[*pos..]` ((atom, token-index) pairs). Binary; n-ary `(op a b c)` is left-folded
/// to `(op (op a b) c)`. Returns None at a malformed/exhausted stream.
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
        *pos += 1; // consume ")"
        if kids.len() < 2 {
            return kids.into_iter().next(); // unary / degenerate → pass the single child through
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
        *pos += 1; // skip a stray atom ('=', a digit-less operator, …)
        parse(atoms, pos)
    }
}

/// Parse every top-level `( … )` expression and return the LAST one (the target, after the few-shot examples).
pub fn parse_target(atoms: &[(String, usize)]) -> Option<Node> {
    let mut pos = 0;
    let mut last = None;
    while pos < atoms.len() {
        if atoms[pos].0 == "(" {
            let mut p = pos;
            match parse(atoms, &mut p) {
                Some(n @ Node::Op(..)) => {
                    last = Some(n);
                    pos = p;
                }
                _ => pos += 1,
            }
        } else {
            pos += 1;
        }
    }
    last
}

/// Ground-truth recursive evaluation (the answer the symbolic program derives).
pub fn eval(n: &Node) -> Option<i64> {
    match n {
        Node::Leaf(v) => Some(*v),
        Node::Op(op, a, b, ..) => {
            let (x, y) = (eval(a)?, eval(b)?);
            Some(match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                '/' => {
                    if y == 0 {
                        return None;
                    }
                    x / y
                }
                _ => return None,
            })
        }
    }
}

fn walk(n: &Node, c: &mut usize, facts: &mut String) -> String {
    let id = format!("n{}", *c);
    *c += 1;
    match n {
        Node::Leaf(v) => facts.push_str(&format!("leaf(\"{id}\", {v}).\n")),
        Node::Op(op, a, b, ..) => {
            let aid = walk(a, c, facts);
            let bid = walk(b, c, facts);
            facts.push_str(&format!("node(\"{id}\", \"{op}\", \"{aid}\", \"{bid}\").\n"));
        }
    }
    id
}

/// Emit the recursive Datalog program for `root`. `model_answer` is the model's ACTUAL top-1 after the expression
/// (`lm.predict`) — the faithfulness anchor: `reproduces` is non-empty iff the recursive program == the model.
pub fn emit(root: &Node, model_answer: Option<i64>) -> String {
    let mut o = String::new();
    o.push_str("// recursion → recursive Datalog  (fieldrun --recursion-explain --datalog · Soufflé)\n");
    o.push_str("// The recursive structure the model evaluates, recovered as a RUNNABLE recursive program: eval/2 is a\n");
    o.push_str("// least-fixpoint over the parse tree — Datalog's recursion IS the model's recursive descent.\n");
    o.push_str("// FAITHFULNESS anchor: model_answer/1 is the model's actual top-1 (lm.predict); `reproduces` is\n");
    o.push_str("// non-empty iff eval(root) == the model's prediction — i.e. the recursive Datalog is top-1 faithful here.\n");
    o.push_str("// Run:  souffle -D- this.dl\n\n");
    o.push_str(".decl leaf(n:symbol, v:number)\n");
    o.push_str(".decl node(n:symbol, op:symbol, a:symbol, b:symbol)\n");
    o.push_str(".decl eval(n:symbol, v:number)\n");
    o.push_str(".decl root(n:symbol)\n");
    o.push_str(".decl model_answer(v:number)\n");
    o.push_str(".decl reproduces(v:number)\n");
    o.push_str(".decl diverges(symbolic:number, model:number)\n\n");

    let (mut facts, mut c) = (String::new(), 0usize);
    let root_id = walk(root, &mut c, &mut facts);
    o.push_str("// ---- parse tree (recovered from the recursion-explain bracket folds) ----\n");
    o.push_str(&facts);
    o.push_str(&format!("root(\"{root_id}\").\n"));
    match model_answer {
        Some(a) => o.push_str(&format!("model_answer({a}).        // the model's ACTUAL top-1 after the expression\n")),
        None => o.push_str("// model_answer: the model's top-1 did not decode to an integer (multi-token / non-numeric)\n"),
    }
    o.push_str("\n// ---- the RECURSIVE evaluator (least-fixpoint = the model's recursive descent) ----\n");
    o.push_str("eval(N,V) :- leaf(N,V).\n");
    o.push_str("eval(N,V) :- node(N,\"+\",A,B), eval(A,X), eval(B,Y), V = X + Y.\n");
    o.push_str("eval(N,V) :- node(N,\"-\",A,B), eval(A,X), eval(B,Y), V = X - Y.\n");
    o.push_str("eval(N,V) :- node(N,\"*\",A,B), eval(A,X), eval(B,Y), V = X * Y.\n");
    o.push_str("eval(N,V) :- node(N,\"/\",A,B), eval(A,X), eval(B,Y), Y != 0, V = X / Y.\n\n");
    o.push_str("// ---- top-1 faithfulness check against the model ----\n");
    o.push_str("reproduces(V) :- root(R), eval(R,V), model_answer(V).               // recursive Datalog == model ✓\n");
    o.push_str("diverges(S,M) :- root(R), eval(R,S), model_answer(M), S != M.        // …or where it doesn't\n\n");
    o.push_str(".output eval\n.output reproduces\n.output diverges\n");
    let sym = eval(root).map(|v| v.to_string()).unwrap_or_else(|| "?".into());
    let verdict = match model_answer {
        Some(a) => if eval(root) == Some(a) { format!("FAITHFUL (model also predicts {a})") }
                   else { format!("DIVERGES (model predicts {a}, symbolic is {sym} — the model's eval is wrong here)") },
        None => "model answer not an integer".into(),
    };
    o.push_str(&format!("\n// symbolic answer eval(\"{root_id}\") = {sym};  top-1 faithfulness: {verdict}\n"));
    o
}
