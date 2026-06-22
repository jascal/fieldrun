//! LO3a — the **context-free whole-model emit**. Where `logic.rs` exports ONE next-token decision as
//! partial-evaluation facts (`contrib(block, token, w)` — numbers already resolved for a specific
//! context), this emits the *computation itself*: the entire rope forward pass as a single Soufflé
//! Datalog program whose ONLY input is `token(pos, id)`. Swap the token facts and Soufflé recomputes
//! the next token from scratch — it generalizes to contexts the exporter never saw.
//!
//! Weights are FACTS; the forward pass is RULES. Soufflé has only `+ - * / ^` and `sum`/`max`
//! aggregates (no `exp`/`sqrt`/`sin`/`cos`), but that is sufficient:
//!   * `sqrt(x)  = x ^ 0.5`            (RMSNorm)
//!   * `exp(x)   = E ^ x`             (softmax, SiLU)   E = 2.718281828459045
//!   * RoPE `sin`/`cos` depend only on POSITION (never token content) ⇒ precomputed model-constant facts.
//! So no FFI / user-defined functors: it is plain, standard Datalog (the "verify in a neutral engine"
//! property `logic.rs` already relies on, extended to the whole model).
//!
//! Tractable for SMALL rope bundles (the demonstration case). For a full-scale model the embed/unembed
//! fact count (vocab × d) is the dense-Gram wall (LOGIC_EXPORT LE-T4): the program exists and is
//! correct, it just is not *compact* — exactly the open frontier the proposal names.

use crate::bundle::Bundle;
use std::fmt::Write;

const E: &str = "2.718281828459045"; // e, so exp(x) = E^x

/// f32 → a Soufflé-safe positional float literal. Rust's `Display` is shortest-round-trip and never
/// uses exponent notation (Soufflé rejects `1e-5`); we only need to guarantee a decimal point so the
/// literal types as `float`, not `number`.
fn ff(x: f32) -> String {
    let mut s = format!("{x}");
    if !s.contains('.') {
        s.push_str(".0");
    }
    s
}

/// Emit the whole-model forward pass for a `rope` bundle as one Datalog program.
/// `maxpos` = how many RoPE position rows to precompute (the max context length the program supports).
pub fn emit_whole(b: &Bundle, maxpos: usize, shortlist_k: Option<usize>) -> Result<String, String> {
    if b.arch != "rope" {
        return Err(format!(
            "logic-whole: arch {:?} unsupported — the whole-model emit targets the rope family (Llama/Qwen2.5/Qwen3/Mistral)",
            b.arch
        ));
    }
    let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
    if c.len() < 8 {
        return Err("logic-whole: rope config must be [n_layer,H,nkv,hd,d,ffn,vocab,tied]".into());
    }
    let (n_layer, h, nkv, hd, d, ffn, vocab, tied) = (
        c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize,
        c[4] as usize, c[5] as usize, c[6] as usize, c[7] != 0,
    );
    let theta = b.config_f[0] as f32;
    let eps = b.config_f[1] as f32;
    let half = hd / 2;
    let rep = h / nkv.max(1);
    if b.has("l0.self_attn.q_norm") {
        // Qwen3-dense per-head QK-norm: a small extra RMSNorm step. Expressible, but not yet verified
        // end-to-end in Soufflé, so refuse rather than emit an unchecked rule. Llama/Qwen2.5/Mistral are fine.
        return Err("logic-whole: this bundle ships qk_norm (Qwen3) — not yet supported by the whole-model emit".into());
    }
    let qk_norm = false;
    if !tied && !b.has("lm_head") {
        return Err("logic-whole: untied model but no lm_head array".into());
    }

    // ---- LE-T4: PO-T3-certified unembed shortlist (option 2) ----
    // Keep only the top-K output tokens by ‖U_v‖ (the unembed row norm — the only tokens that can have a large logit),
    // emit the unembed for just those, and add a Soufflé-checkable CERTIFICATE: the shortlist argmax provably equals the
    // full-vocab argmax whenever its logit S exceeds ‖x‖·max‖U_elided‖ (no elided token's logit ⟨x,U_v⟩ ≤ ‖x‖‖U_v‖ can
    // reach it). Where the certificate fires the dense vocab×d unembed shrinks to shortlist×d; the thin tail is uncertified.
    let unembed_name = if tied { "embed" } else { "lm_head" };
    let (shortlist, umax2_elided): (Option<Vec<usize>>, f32) = match shortlist_k {
        Some(k) if k > 0 && k < vocab => {
            let (ush, ud) = b.f32_array(unembed_name); // [vocab, d]
            let dc = ush[1];
            let mut norms: Vec<(usize, f32)> = (0..vocab)
                .map(|v| (v, (0..dc).map(|j| { let x = ud[v * dc + j]; x * x }).sum::<f32>()))
                .collect();
            norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // desc by ‖U_v‖²
            let keep: Vec<usize> = norms[..k].iter().map(|&(v, _)| v).collect();
            (Some(keep), norms[k].1) // umax²_elided = the (k+1)-th largest row-norm²
        }
        _ => (None, 0.0),
    };

    // RoPE inverse frequencies, computed in f32 exactly as src/rope.rs does.
    let inv: Vec<f32> = (0..half).map(|j| 1.0f32 / theta.powf(2.0 * j as f32 / hd as f32)).collect();

    let mut o = String::with_capacity(1 << 20);
    macro_rules! w { ($($a:tt)*) => {{ let _ = writeln!(o, $($a)*); }} }

    // ---- header ----
    w!("// ============================================================");
    w!("// fieldrun LOGIC EXPORT — LO3a: CONTEXT-FREE WHOLE-MODEL forward pass as ONE Datalog program.");
    w!("// Input:  token(pos,id) facts (an ARBITRARY context — provide via `-F <dir>/token.facts`).");
    w!("// Output: decide(v) = argmax next-token id; logit(v,s) = the full scoreboard.");
    w!("// Weights are FACTS; the forward pass (RMSNorm, RoPE attn, SwiGLU MLP, unembed, argmax) is RULES.");
    w!("// NOTHING is specialised to a context (unlike export --logic / stitch): swap token.facts and");
    w!("// Soufflé recomputes from scratch — it answers contexts the exporter never saw. This is LO3a.");
    w!("// exp(x)=E^x, sqrt(x)=x^0.5, RoPE sin/cos = precomputed per-position facts ⇒ plain Datalog, no FFI.");
    w!("// config: n_layer={n_layer} H={h} nkv={nkv} hd={hd} d={d} ffn={ffn} vocab={vocab} tied={tied} qk_norm={qk_norm}");
    w!("// floats: theta={theta} eps={eps}   RoPE tables: positions 0..{}", maxpos - 1);
    w!("// Run: souffle <this>.dl -F <ctxdir> -D -     (ctxdir/token.facts holds `pos<TAB>id` rows)");
    w!("// ============================================================");
    w!();

    // ---- input ----
    w!(".decl token(pos:number, id:number)");
    w!(".input token");
    w!();

    // ---- structural index relations (context-free) ----
    w!(".decl dim_d(d:number)");
    w!(".decl kvout(o:number)");
    w!(".decl ffnout(f:number)");
    w!(".decl vocab(v:number)");
    w!(".decl cidx(c:number)");
    w!(".decl headq(h:number)");
    w!(".decl head_kv(h:number, kv:number)");
    for i in 0..d { w!("dim_d({i})."); }
    for i in 0..nkv * hd { w!("kvout({i})."); }
    for i in 0..ffn { w!("ffnout({i})."); }
    match &shortlist {                                   // output tokens: the PO-T3 shortlist, or all vocab
        Some(keep) => for &v in keep { w!("vocab({v})."); },
        None => for i in 0..vocab { w!("vocab({i})."); },
    }
    for i in 0..hd { w!("cidx({i})."); }
    for i in 0..h { w!("headq({i})."); }
    for i in 0..h { w!("head_kv({i}, {}).", i / rep); }
    w!();

    // ---- RoPE pairing (which two dims rotate together) + precomputed cos/sin ----
    w!(".decl qrope(o:number, opart:number, j:number, sign:float)");
    w!(".decl krope(o:number, opart:number, j:number, sign:float)");
    let rope_pairs = |rel: &str, width: usize, o: &mut String| {
        for head in 0..width / hd {
            let base = head * hd;
            for j in 0..half {
                // first half:  new[base+j]      = old[base+j]*c      - old[base+j+half]*s
                let _ = writeln!(o, "{rel}({}, {}, {j}, -1.0).", base + j, base + j + half);
                // second half: new[base+j+half] = old[base+j+half]*c + old[base+j]*s
                let _ = writeln!(o, "{rel}({}, {}, {j}, 1.0).", base + j + half, base + j);
            }
        }
    };
    rope_pairs("qrope", h * hd, &mut o);
    rope_pairs("krope", nkv * hd, &mut o);
    w!();
    w!(".decl rope_cos(pos:number, j:number, c:float)");
    w!(".decl rope_sin(pos:number, j:number, s:float)");
    for pos in 0..maxpos {
        for j in 0..half {
            let ang = pos as f32 * inv[j];
            w!("rope_cos({pos}, {j}, {}).", ff(ang.cos()));
            w!("rope_sin({pos}, {j}, {}).", ff(ang.sin()));
        }
    }
    w!();

    // ---- weight facts ----
    // emit a 2D weight stored [in, out] (row-major) as rel(in, out, val)
    let emit_mat = |rel: &str, name: &str, rows: Option<&[usize]>, o: &mut String| -> Result<(), String> {
        let (shape, data) = b.f32_array(name);
        if shape.len() != 2 { return Err(format!("logic-whole: {name} is not 2D")); }
        let (ni, no) = (shape[0], shape[1]);
        let _ = writeln!(o, ".decl {rel}(i:number, o:number, v:float)");
        let emit_row = |i: usize, o: &mut String| { for j in 0..no { let _ = writeln!(o, "{rel}({i}, {j}, {}).", ff(data[i * no + j])); } };
        match rows {                                    // None = all rows; Some = just those (the unembed shortlist)
            Some(rs) => for &i in rs { emit_row(i, o); },
            None => for i in 0..ni { emit_row(i, o); },
        }
        Ok(())
    };
    let emit_vec = |rel: &str, name: &str, o: &mut String| {
        let v = b.arr1(name);
        let _ = writeln!(o, ".decl {rel}(d:number, v:float)");
        for (i, &val) in v.iter().enumerate() {
            let _ = writeln!(o, "{rel}({i}, {}).", ff(val));
        }
    };
    // optional bias vector (q/k/v proj on Qwen2.5); returns whether it was present
    let emit_bias = |rel: &str, name: &str, o: &mut String| -> bool {
        let bn = format!("{name}.bias");
        if b.has(&bn) { emit_vec(rel, &bn, o); true } else { false }
    };

    emit_mat("embed_w", "embed", None, &mut o)?;        // embed: ALL rows (any input token can appear in the context)
    // lm_head (untied unembed): just the shortlist rows when shortlisting — this is the LE-T4 size win (vocab×d → K×d).
    if !tied { emit_mat("lmhead_w", "lm_head", shortlist.as_deref(), &mut o)?; }
    let mut has_qb = vec![false; n_layer];
    let mut has_kb = vec![false; n_layer];
    let mut has_vb = vec![false; n_layer];
    for l in 0..n_layer {
        let p = format!("l{l}.");
        emit_vec(&format!("inln{l}"), &format!("{p}in_ln"), &mut o);
        emit_mat(&format!("qw{l}"), &format!("{p}self_attn.q_proj"), None, &mut o)?;
        emit_mat(&format!("kw{l}"), &format!("{p}self_attn.k_proj"), None, &mut o)?;
        emit_mat(&format!("vw{l}"), &format!("{p}self_attn.v_proj"), None, &mut o)?;
        has_qb[l] = emit_bias(&format!("qb{l}"), &format!("{p}self_attn.q_proj"), &mut o);
        has_kb[l] = emit_bias(&format!("kb{l}"), &format!("{p}self_attn.k_proj"), &mut o);
        has_vb[l] = emit_bias(&format!("vb{l}"), &format!("{p}self_attn.v_proj"), &mut o);
        emit_mat(&format!("ow{l}"), &format!("{p}self_attn.o_proj"), None, &mut o)?;
        emit_vec(&format!("postln{l}"), &format!("{p}post_ln"), &mut o);
        emit_mat(&format!("gatew{l}"), &format!("{p}mlp.gate_proj"), None, &mut o)?;
        emit_mat(&format!("upw{l}"), &format!("{p}mlp.up_proj"), None, &mut o)?;
        emit_mat(&format!("downw{l}"), &format!("{p}mlp.down_proj"), None, &mut o)?;
    }
    emit_vec("normw", "norm", &mut o);
    w!();

    // ---- forward-pass rules (layers unrolled — fixed depth, still context-free) ----
    let (dv, epsv, invsqhd) = (ff(d as f32), ff(eps), ff(1.0 / (hd as f32).sqrt()));
    w!(".decl x0(pos:number, d:number, v:float)");
    w!("x0(P, D, V) :- token(P, Id), embed_w(Id, D, V).");
    w!();

    for l in 0..n_layer {
        let (xin, xmid, xout) = (format!("x{l}"), format!("xmid{l}"), format!("x{}", l + 1));
        w!("// ---------- layer {l} ----------");
        // pre-attn RMSNorm
        w!(".decl ssin{l}(pos:number, s:float)");
        w!("ssin{l}(P, S) :- token(P,_), S = sum (V*V) : {{ {xin}(P,_,V) }}.");
        w!(".decl a{l}(pos:number, d:number, v:float)");
        w!("a{l}(P, D, V * (((SS/{dv})+{epsv})^(-0.5)) * G) :- {xin}(P,D,V), ssin{l}(P,SS), inln{l}(D,G).");
        // q/k/v projections (+ optional bias)
        let qadd = if has_qb[l] { ", qb{l}(O,B)".replace("{l}", &l.to_string()) } else { String::new() };
        let qsum = if has_qb[l] { "+B" } else { "" };
        w!(".decl q{l}(pos:number, o:number, v:float)");
        w!("q{l}(P,O,S{qsum}) :- token(P,_), dim_d(O){qadd}, S = sum (AV*WV) : {{ a{l}(P,I,AV), qw{l}(I,O,WV) }}.");
        let kadd = if has_kb[l] { ", kb{l}(O,B)".replace("{l}", &l.to_string()) } else { String::new() };
        let ksum = if has_kb[l] { "+B" } else { "" };
        w!(".decl k{l}(pos:number, o:number, v:float)");
        w!("k{l}(P,O,S{ksum}) :- token(P,_), kvout(O){kadd}, S = sum (AV*WV) : {{ a{l}(P,I,AV), kw{l}(I,O,WV) }}.");
        let vadd = if has_vb[l] { ", vb{l}(O,B)".replace("{l}", &l.to_string()) } else { String::new() };
        let vsum = if has_vb[l] { "+B" } else { "" };
        w!(".decl v{l}(pos:number, o:number, v:float)");
        w!("v{l}(P,O,S{vsum}) :- token(P,_), kvout(O){vadd}, S = sum (AV*WV) : {{ a{l}(P,I,AV), vw{l}(I,O,WV) }}.");
        // RoPE (applied to the q/k projection output, per head)
        w!(".decl qr{l}(pos:number, o:number, v:float)");
        w!("qr{l}(P,O,NV) :- q{l}(P,O,V), qrope(O,OP,J,SG), q{l}(P,OP,VP), rope_cos(P,J,C), rope_sin(P,J,SN), NV = V*C + SG*VP*SN.");
        w!(".decl kr{l}(pos:number, o:number, v:float)");
        w!("kr{l}(P,O,NV) :- k{l}(P,O,V), krope(O,OP,J,SG), k{l}(P,OP,VP), rope_cos(P,J,C), rope_sin(P,J,SN), NV = V*C + SG*VP*SN.");
        // attention scores (causal J<=I), scaled by 1/sqrt(hd)
        w!(".decl score{l}(h:number, i:number, j:number, s:float)");
        w!("score{l}(HH,I,J, RAW*{invsqhd}) :- headq(HH), head_kv(HH,KV), token(I,_), token(J,_), J<=I, \
            RAW = sum (QV*KV2) : {{ cidx(C), qr{l}(I,OQ,QV), OQ=HH*{hd}+C, kr{l}(J,OK,KV2), OK=KV*{hd}+C }}.");
        w!(".decl smax{l}(h:number, i:number, m:float)");
        w!("smax{l}(HH,I,M) :- score{l}(HH,I,_,_), M = max SC : {{ score{l}(HH,I,_,SC) }}.");
        w!(".decl sexp{l}(h:number, i:number, j:number, e:float)");
        w!("sexp{l}(HH,I,J,E) :- score{l}(HH,I,J,SC), smax{l}(HH,I,M), E = {E}^(SC-M).");
        w!(".decl sden{l}(h:number, i:number, z:float)");
        w!("sden{l}(HH,I,Z) :- smax{l}(HH,I,_), Z = sum EE : {{ sexp{l}(HH,I,_,EE) }}.");
        w!(".decl prob{l}(h:number, i:number, j:number, p:float)");
        w!("prob{l}(HH,I,J,P) :- sexp{l}(HH,I,J,E), sden{l}(HH,I,Z), P = E/Z.");
        // attn_out[i, h*hd+c] = Σ_j prob * v[j, kv*hd+c]
        w!(".decl attno{l}(pos:number, o:number, v:float)");
        w!("attno{l}(I,O,S) :- headq(HH), head_kv(HH,KV), cidx(C), O=HH*{hd}+C, token(I,_), \
            S = sum (P*VV) : {{ token(J,_), prob{l}(HH,I,J,P), v{l}(J,OV,VV), OV=KV*{hd}+C }}.");
        // o_proj + residual
        w!(".decl oproj{l}(pos:number, d:number, v:float)");
        w!("oproj{l}(P,D,S) :- token(P,_), dim_d(D), S = sum (AV*WV) : {{ attno{l}(P,I,AV), ow{l}(I,D,WV) }}.");
        w!(".decl {xmid}(pos:number, d:number, v:float)");
        w!("{xmid}(P,D, XV+OV) :- {xin}(P,D,XV), oproj{l}(P,D,OV).");
        // post-attn RMSNorm
        w!(".decl ssm{l}(pos:number, s:float)");
        w!("ssm{l}(P,S) :- token(P,_), S = sum (V*V) : {{ {xmid}(P,_,V) }}.");
        w!(".decl a2_{l}(pos:number, d:number, v:float)");
        w!("a2_{l}(P,D, V*(((SS/{dv})+{epsv})^(-0.5))*G) :- {xmid}(P,D,V), ssm{l}(P,SS), postln{l}(D,G).");
        // SwiGLU MLP
        w!(".decl gate{l}(pos:number, f:number, v:float)");
        w!("gate{l}(P,F,S) :- token(P,_), ffnout(F), S = sum (AV*WV) : {{ a2_{l}(P,I,AV), gatew{l}(I,F,WV) }}.");
        w!(".decl up{l}(pos:number, f:number, v:float)");
        w!("up{l}(P,F,S) :- token(P,_), ffnout(F), S = sum (AV*WV) : {{ a2_{l}(P,I,AV), upw{l}(I,F,WV) }}.");
        w!(".decl hid{l}(pos:number, f:number, v:float)");
        w!("hid{l}(P,F, (G/(1.0+{E}^(0.0-G)))*U) :- gate{l}(P,F,G), up{l}(P,F,U).");
        // down_proj + residual
        w!(".decl down{l}(pos:number, d:number, v:float)");
        w!("down{l}(P,D,S) :- token(P,_), dim_d(D), S = sum (HV*WV) : {{ hid{l}(P,F,HV), downw{l}(F,D,WV) }}.");
        w!(".decl {xout}(pos:number, d:number, v:float)");
        w!("{xout}(P,D, XV+DV) :- {xmid}(P,D,XV), down{l}(P,D,DV).");
        w!();
    }

    // ---- final RMSNorm + unembed (last position only) + argmax ----
    let xn = format!("x{n_layer}");
    let unembed_rel = if tied { "embed_w" } else { "lmhead_w" };
    w!("// ---------- final norm + unembed ({}) + argmax ----------", if tied { "tied" } else { "lm_head" });
    w!(".decl ssf(pos:number, s:float)");
    w!("ssf(P,S) :- token(P,_), S = sum (V*V) : {{ {xn}(P,_,V) }}.");
    w!(".decl xf(pos:number, d:number, v:float)");
    w!("xf(P,D, V*(((SS/{dv})+{epsv})^(-0.5))*G) :- {xn}(P,D,V), ssf(P,SS), normw(D,G).");
    w!(".decl lastpos(p:number)");
    w!("lastpos(P) :- P = max Q : {{ token(Q,_) }}.");
    w!(".decl logit(v:number, s:float)");
    w!("logit(V,S) :- vocab(V), lastpos(LP), S = sum (XV*EV) : {{ xf(LP,D,XV), {unembed_rel}(V,D,EV) }}.");
    w!(".decl decide(v:number)");
    w!("decide(V) :- logit(V,S), S = max S2 : {{ logit(_,S2) }}.");
    w!(".output decide");
    w!(".output logit");
    if shortlist.is_some() {
        // PO-T3 / LE-T4 certificate: the shortlist argmax == the full-vocab argmax when the winner's logit S exceeds
        // ‖x‖·max‖U_elided‖ — because every elided token's logit ⟨x,U_v⟩ ≤ ‖x‖·‖U_v‖ ≤ ‖x‖·max‖U_elided‖ < S, so no
        // dropped token can beat it. `certified()` true ⇒ decide is exact; false ⇒ thin-margin, fall back to full vocab.
        w!("// ---------- LE-T4 shortlist certificate (umax²_elided = {}) ----------", ff(umax2_elided));
        w!(".decl xfn(s:float)");
        w!("xfn(N) :- lastpos(LP), N = sum (V*V) : {{ xf(LP,_,V) }}.   // ‖x‖² at the predicting position");
        w!(".decl certified()");
        w!("certified() :- decide(V), logit(V,S), S>0, xfn(XN), S*S > XN*{}.   // S > ‖x‖·max‖U_elided‖", ff(umax2_elided));
        w!(".output certified");
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::ff;
    #[test]
    fn ff_is_souffle_safe() {
        // Soufflé floats: must carry a decimal point (else they type as `number`) and must NEVER use
        // exponent notation (Soufflé rejects `1e-5`). Rust Display gives positional shortest-round-trip.
        for x in [0.0f32, 1.0, -1.0, 5.0, 0.5, -0.0015, 2.0093488e-5, -8.22e-2, 1e9, 1e-9, 13.085766] {
            let s = ff(x);
            assert!(s.contains('.'), "{x} -> {s} has no decimal point");
            assert!(!s.contains('e') && !s.contains('E'), "{x} -> {s} uses exponent notation");
            // round-trips back to the same f32
            assert_eq!(s.parse::<f32>().unwrap(), x, "{x} -> {s} did not round-trip");
        }
    }
}
