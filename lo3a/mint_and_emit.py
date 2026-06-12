#!/usr/bin/env python3
"""LO3a prototype: mint a TINY real rope (Llama/Qwen2.5-style) fieldrun bundle, compute a numpy
reference forward that mirrors src/rope.rs exactly (float32), and emit the CONTEXT-FREE whole-model
forward pass as a single Soufflé Datalog program.

The emitted .dl takes an arbitrary token context as `.input token(pos,id)` and *computes* the next
token (argmax) from scratch — weights are facts, the forward pass (RMSNorm, RoPE attention, SwiGLU
MLP, unembed, argmax) is rules. Unlike the stitched trace, NOTHING about a specific context is baked
in: change token.facts and Soufflé recomputes. RoPE sin/cos are precomputed per-position model
constants (depend only on position, never on token content).
"""
import json, struct, os, sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
# variant knobs: BIAS=1 adds q/k/v proj biases (Qwen2.5); UNTIE=1 gives a separate lm_head (untied unembed)
BIAS  = os.environ.get("BIAS", "0") == "1"
UNTIE = os.environ.get("UNTIE", "0") == "1"
VAR   = ("_bias" if BIAS else "") + ("_untied" if UNTIE else "")
STEM = os.path.join(OUT, "tiny"+VAR, "tiny"+VAR)
os.makedirs(os.path.dirname(STEM), exist_ok=True)

# ---- tiny config (every dim small so Soufflé is instant, but ALL machinery exercised) ----
N_LAYER = 2
H       = 4      # query heads
NKV     = 2      # kv heads (GQA, rep = H/NKV = 2)
HD      = 8      # head dim (even -> RoPE half = 4)
D       = H * HD # 32  (hidden size = n_heads * head_dim, as in Llama)
FFN     = 64
VOCAB   = 48
THETA   = 10000.0
EPS     = 1e-6
TIED    = 0 if UNTIE else 1
HALF    = HD // 2
REP     = H // NKV
MAXPOS  = 16     # RoPE tables emitted for positions 0..MAXPOS-1

rng = np.random.default_rng(7)
def randn(*shape, scale): return (rng.standard_normal(shape) * scale).astype(np.float32)
def gain(d):              return (1.0 + rng.standard_normal(d) * 0.05).astype(np.float32)

# weights — orientation matches bundle.rs: projections stored [in, out] (mm: out = a @ W[in,out]);
# embed stored [vocab, d] (row per token, tied -> also the unembed).
W = {}
W["embed"] = randn(VOCAB, D, scale=0.8)
if UNTIE: W["lm_head"] = randn(VOCAB, D, scale=0.8)
for l in range(N_LAYER):
    p = f"l{l}."
    W[p+"in_ln"]            = gain(D)
    W[p+"self_attn.q_proj"] = randn(D, H*HD,  scale=1.0/np.sqrt(D))
    W[p+"self_attn.k_proj"] = randn(D, NKV*HD, scale=1.0/np.sqrt(D))
    W[p+"self_attn.v_proj"] = randn(D, NKV*HD, scale=1.0/np.sqrt(D))
    if BIAS:
        W[p+"self_attn.q_proj.bias"] = randn(H*HD,  scale=0.3)
        W[p+"self_attn.k_proj.bias"] = randn(NKV*HD, scale=0.3)
        W[p+"self_attn.v_proj.bias"] = randn(NKV*HD, scale=0.3)
    W[p+"self_attn.o_proj"] = randn(H*HD, D,  scale=1.0/np.sqrt(H*HD))
    W[p+"post_ln"]          = gain(D)
    W[p+"mlp.gate_proj"]    = randn(D, FFN,   scale=1.0/np.sqrt(D))
    W[p+"mlp.up_proj"]      = randn(D, FFN,   scale=1.0/np.sqrt(D))
    W[p+"mlp.down_proj"]    = randn(FFN, D,   scale=1.0/np.sqrt(FFN))
W["norm"] = gain(D)

# ---- write the fieldrun bundle (.json manifest + .bin blob), all f32 ----
def write_bundle():
    order = ["embed"] + (["lm_head"] if UNTIE else [])
    for l in range(N_LAYER):
        p = f"l{l}."
        order += [p+"in_ln", p+"self_attn.q_proj", p+"self_attn.k_proj", p+"self_attn.v_proj"]
        if BIAS:
            order += [p+"self_attn.q_proj.bias", p+"self_attn.k_proj.bias", p+"self_attn.v_proj.bias"]
        order += [p+"self_attn.o_proj", p+"post_ln", p+"mlp.gate_proj", p+"mlp.up_proj", p+"mlp.down_proj"]
    order += ["norm"]
    blob = bytearray(); arrays = []; off = 0
    for name in order:
        a = np.ascontiguousarray(W[name], dtype="<f4")
        b = a.tobytes(); blob += b
        arrays.append({"name": name, "dtype": "f32", "shape": list(a.shape), "offset": off, "bytes": len(b)})
        off += len(b)
    manifest = {"format": "fieldrun-bundle", "version": 1, "arch": "rope",
                "config": [N_LAYER, H, NKV, HD, D, FFN, VOCAB, TIED],
                "config_f": [THETA, EPS], "arrays": arrays}
    with open(STEM+".fieldrun.json", "w") as f: json.dump(manifest, f, indent=2)
    with open(STEM+".fieldrun.bin", "wb") as f: f.write(blob)
    print(f"[mint] wrote {STEM}.fieldrun.{{json,bin}}  ({off} bytes blob, {len(arrays)} arrays)")

# ---- numpy reference forward, float32, mirroring src/rope.rs::hidden + unembed ----
INV = (1.0 / (THETA ** (2.0 * np.arange(HALF, dtype=np.float32) / HD))).astype(np.float32)  # [half]

def rmsnorm(x, w):                       # x:[seq,d]
    ms = (x.astype(np.float32)**2).mean(axis=1, keepdims=True)
    inv = (1.0/np.sqrt(ms + np.float32(EPS))).astype(np.float32)
    return (x * inv * w).astype(np.float32)

def silu(x): return (x / (1.0 + np.exp(-x))).astype(np.float32)

def rope(x, n_heads):                    # x:[seq, n_heads*hd], pos0=0
    x = x.copy()
    seq = x.shape[0]
    for i in range(seq):
        for head in range(n_heads):
            base = head*HD
            for j in range(HALF):
                ang = np.float32(i) * INV[j]
                c, s = np.float32(np.cos(ang)), np.float32(np.sin(ang))
                a, b = x[i, base+j], x[i, base+j+HALF]
                x[i, base+j]      = a*c - b*s
                x[i, base+j+HALF] = b*c + a*s
    return x.astype(np.float32)

def forward_logits(ids):
    ids = list(ids); seq = len(ids)
    x = W["embed"][ids].astype(np.float32)            # [seq, d]
    for l in range(N_LAYER):
        p = f"l{l}."
        a = rmsnorm(x, W[p+"in_ln"])
        q = (a @ W[p+"self_attn.q_proj"]).astype(np.float32)   # [seq, H*hd]
        k = (a @ W[p+"self_attn.k_proj"]).astype(np.float32)   # [seq, nkv*hd]
        v = (a @ W[p+"self_attn.v_proj"]).astype(np.float32)
        if BIAS:
            q = (q + W[p+"self_attn.q_proj.bias"]).astype(np.float32)
            k = (k + W[p+"self_attn.k_proj.bias"]).astype(np.float32)
            v = (v + W[p+"self_attn.v_proj.bias"]).astype(np.float32)
        q = rope(q, H); k = rope(k, NKV)
        attn_out = np.zeros((seq, H*HD), dtype=np.float32)
        for head in range(H):
            kv = head // REP
            qh = q[:, head*HD:(head+1)*HD]
            kh = k[:, kv*HD:(kv+1)*HD]
            vh = v[:, kv*HD:(kv+1)*HD]
            scores = (qh @ kh.T).astype(np.float32) / np.float32(np.sqrt(HD))
            for i in range(seq):
                for j in range(i+1, seq): scores[i, j] = np.float32(-1e30)
            m = scores.max(axis=1, keepdims=True)
            e = np.exp(scores - m).astype(np.float32)
            probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
            attn_out[:, head*HD:(head+1)*HD] = (probs @ vh).astype(np.float32)
        x = (x + attn_out @ W[p+"self_attn.o_proj"]).astype(np.float32)
        a2 = rmsnorm(x, W[p+"post_ln"])
        g = (a2 @ W[p+"mlp.gate_proj"]).astype(np.float32)
        u = (a2 @ W[p+"mlp.up_proj"]).astype(np.float32)
        hid = (silu(g) * u).astype(np.float32)
        x = (x + hid @ W[p+"mlp.down_proj"]).astype(np.float32)
    xf = rmsnorm(x, W["norm"])
    unembed = W["lm_head"] if UNTIE else W["embed"]
    logits = (xf[-1] @ unembed.T).astype(np.float32)           # last position
    return logits

def predict(ids): return int(np.argmax(forward_logits(ids)))

# ---- Datalog emit ----
E = "2.718281828459045"  # e, for exp(x) = E^x
def ff(x):               # f32-roundtripping POSITIONAL float literal (Soufflé has no scientific notation)
    s = np.format_float_positional(np.float32(x), unique=True, trim='-')
    if s.startswith('.'):  s = '0' + s
    if s.startswith('-.'): s = '-0' + s[1:]
    if '.' not in s:       s += '.0'
    if s.endswith('.'):    s += '0'
    return s

def emit_dl(path):
    L = []
    def w(s=""): L.append(s)
    w("// ============================================================")
    w("// fieldrun LOGIC EXPORT — LO3a: CONTEXT-FREE WHOLE-MODEL forward pass as ONE Datalog program.")
    w("// Input: token(pos,id) facts (an arbitrary context). Output: decide(v) = argmax next token.")
    w("// Weights are FACTS; the forward pass (RMSNorm, RoPE attn, SwiGLU MLP, unembed, argmax) is RULES.")
    w("// Nothing here is specialised to a context — swap token.facts and Soufflé recomputes from scratch.")
    w(f"// config: n_layer={N_LAYER} H={H} nkv={NKV} hd={HD} d={D} ffn={FFN} vocab={VOCAB} theta={THETA} eps={EPS} tied={TIED}")
    w("// Run: souffle whole.dl -F <ctxdir> -D -        (ctxdir/token.facts = the context)")
    w("// ============================================================")
    w()
    # ---- input ----
    w(".decl token(pos:number, id:number)")
    w(".input token")
    w()
    # ---- index/range relations (model structure, context-free) ----
    w(".decl dim_d(d:number)")
    w(".decl kvout(o:number)")
    w(".decl ffnout(f:number)")
    w(".decl vocab(v:number)")
    w(".decl cidx(c:number)")
    w(".decl headq(h:number)")
    w(".decl head_kv(h:number, kv:number)")
    for d in range(D):    w(f"dim_d({d}).")
    for o in range(NKV*HD): w(f"kvout({o}).")
    for f in range(FFN):  w(f"ffnout({f}).")
    for vv in range(VOCAB): w(f"vocab({vv}).")
    for c in range(HD):   w(f"cidx({c}).")
    for h in range(H):    w(f"headq({h}).")
    for h in range(H):    w(f"head_kv({h}, {h//REP}).")
    w()
    # ---- RoPE pairing + precomputed cos/sin (depend only on position) ----
    w(".decl qrope(o:number, opart:number, j:number, sign:float)")
    w(".decl krope(o:number, opart:number, j:number, sign:float)")
    def rope_pairs(width, rel):
        nh = width // HD
        for head in range(nh):
            base = head*HD
            for j in range(HALF):
                # first half: new[base+j] = old[base+j]*c - old[base+j+half]*s
                w(f"{rel}({base+j}, {base+j+HALF}, {j}, -1.0).")
                # second half: new[base+j+half] = old[base+j+half]*c + old[base+j]*s
                w(f"{rel}({base+j+HALF}, {base+j}, {j}, 1.0).")
    rope_pairs(H*HD, "qrope")
    rope_pairs(NKV*HD, "krope")
    w()
    w(".decl rope_cos(pos:number, j:number, c:float)")
    w(".decl rope_sin(pos:number, j:number, s:float)")
    for pos in range(MAXPOS):
        for j in range(HALF):
            ang = np.float32(pos) * INV[j]
            w(f"rope_cos({pos}, {j}, {ff(np.cos(ang))}).")
            w(f"rope_sin({pos}, {j}, {ff(np.sin(ang))}).")
    w()
    # ---- weight facts ----
    def emit_mat(rel, mat):               # mat[in,out] -> rel(in,out,val)
        w(f".decl {rel}(i:number, o:number, v:float)")
        I, O = mat.shape
        for i in range(I):
            for o in range(O):
                w(f"{rel}({i}, {o}, {ff(mat[i,o])}).")
    def emit_vec(rel, vec):               # vec[d] -> rel(d,val)
        w(f".decl {rel}(d:number, v:float)")
        for i, val in enumerate(vec): w(f"{rel}({i}, {ff(val)}).")
    emit_mat("embed_w", W["embed"])       # [vocab, d]
    for l in range(N_LAYER):
        p = f"l{l}."
        emit_vec(f"inln{l}",  W[p+"in_ln"])
        emit_mat(f"qw{l}",    W[p+"self_attn.q_proj"])
        emit_mat(f"kw{l}",    W[p+"self_attn.k_proj"])
        emit_mat(f"vw{l}",    W[p+"self_attn.v_proj"])
        emit_mat(f"ow{l}",    W[p+"self_attn.o_proj"])
        emit_vec(f"postln{l}",W[p+"post_ln"])
        emit_mat(f"gatew{l}", W[p+"mlp.gate_proj"])
        emit_mat(f"upw{l}",   W[p+"mlp.up_proj"])
        emit_mat(f"downw{l}", W[p+"mlp.down_proj"])
    emit_vec("normw", W["norm"])
    w()
    # ---- forward-pass rules (layers unrolled) ----
    DV, EPSV, SQHD, INVSQHD = ff(D), ff(EPS), ff(np.sqrt(HD)), ff(1.0/np.sqrt(HD))
    w(".decl x0(pos:number, d:number, v:float)")
    w("x0(P, D, V) :- token(P, Id), embed_w(Id, D, V).")
    w()
    for l in range(N_LAYER):
        xin, xmid, xout = f"x{l}", f"xmid{l}", f"x{l+1}"
        w(f"// ---------- layer {l} ----------")
        # pre-attn RMSNorm
        w(f".decl ssin{l}(pos:number, s:float)")
        w(f"ssin{l}(P, S) :- token(P,_), S = sum (V*V) : {{ {xin}(P,_,V) }}.")
        w(f".decl a{l}(pos:number, d:number, v:float)")
        w(f"a{l}(P, D, V * (((SS/{DV})+{EPSV})^(-0.5)) * G) :- {xin}(P,D,V), ssin{l}(P,SS), inln{l}(D,G).")
        # q/k/v projections
        w(f".decl q{l}(pos:number, o:number, v:float)")
        w(f"q{l}(P,O,S) :- token(P,_), dim_d(O), S = sum (AV*WV) : {{ a{l}(P,I,AV), qw{l}(I,O,WV) }}.")
        w(f".decl k{l}(pos:number, o:number, v:float)")
        w(f"k{l}(P,O,S) :- token(P,_), kvout(O), S = sum (AV*WV) : {{ a{l}(P,I,AV), kw{l}(I,O,WV) }}.")
        w(f".decl v{l}(pos:number, o:number, v:float)")
        w(f"v{l}(P,O,S) :- token(P,_), kvout(O), S = sum (AV*WV) : {{ a{l}(P,I,AV), vw{l}(I,O,WV) }}.")
        # RoPE
        w(f".decl qr{l}(pos:number, o:number, v:float)")
        w(f"qr{l}(P,O,NV) :- q{l}(P,O,V), qrope(O,OP,J,SG), q{l}(P,OP,VP), rope_cos(P,J,C), rope_sin(P,J,SN), NV = V*C + SG*VP*SN.")
        w(f".decl kr{l}(pos:number, o:number, v:float)")
        w(f"kr{l}(P,O,NV) :- k{l}(P,O,V), krope(O,OP,J,SG), k{l}(P,OP,VP), rope_cos(P,J,C), rope_sin(P,J,SN), NV = V*C + SG*VP*SN.")
        # attention scores (causal: only J<=I), scaled
        w(f".decl score{l}(h:number, i:number, j:number, s:float)")
        w(f"score{l}(HH,I,J, RAW*{INVSQHD}) :- headq(HH), head_kv(HH,KV), token(I,_), token(J,_), J<=I, "
          f"RAW = sum (QV*KV2) : {{ cidx(C), qr{l}(I,OQ,QV), OQ=HH*{HD}+C, kr{l}(J,OK,KV2), OK=KV*{HD}+C }}.")
        w(f".decl smax{l}(h:number, i:number, m:float)")
        w(f"smax{l}(HH,I,M) :- score{l}(HH,I,_,_), M = max SC : {{ score{l}(HH,I,_,SC) }}.")
        w(f".decl sexp{l}(h:number, i:number, j:number, e:float)")
        w(f"sexp{l}(HH,I,J,E) :- score{l}(HH,I,J,SC), smax{l}(HH,I,M), E = {E}^(SC-M).")
        w(f".decl sden{l}(h:number, i:number, z:float)")
        w(f"sden{l}(HH,I,Z) :- smax{l}(HH,I,_), Z = sum EE : {{ sexp{l}(HH,I,_,EE) }}.")
        w(f".decl prob{l}(h:number, i:number, j:number, p:float)")
        w(f"prob{l}(HH,I,J,P) :- sexp{l}(HH,I,J,E), sden{l}(HH,I,Z), P = E/Z.")
        # attn_out[i, h*hd+c] = sum_j prob * v[j, kv*hd+c]
        w(f".decl attno{l}(pos:number, o:number, v:float)")
        w(f"attno{l}(I,O,S) :- headq(HH), head_kv(HH,KV), cidx(C), O=HH*{HD}+C, token(I,_), "
          f"S = sum (P*VV) : {{ token(J,_), prob{l}(HH,I,J,P), v{l}(J,OV,VV), OV=KV*{HD}+C }}.")
        # o_proj + residual
        w(f".decl oproj{l}(pos:number, d:number, v:float)")
        w(f"oproj{l}(P,D,S) :- token(P,_), dim_d(D), S = sum (AV*WV) : {{ attno{l}(P,I,AV), ow{l}(I,D,WV) }}.")
        w(f".decl {xmid}(pos:number, d:number, v:float)")
        w(f"{xmid}(P,D, XV+OV) :- {xin}(P,D,XV), oproj{l}(P,D,OV).")
        # post-attn RMSNorm
        w(f".decl ssm{l}(pos:number, s:float)")
        w(f"ssm{l}(P,S) :- token(P,_), S = sum (V*V) : {{ {xmid}(P,_,V) }}.")
        w(f".decl a2_{l}(pos:number, d:number, v:float)")
        w(f"a2_{l}(P,D, V*(((SS/{DV})+{EPSV})^(-0.5))*G) :- {xmid}(P,D,V), ssm{l}(P,SS), postln{l}(D,G).")
        # SwiGLU MLP
        w(f".decl gate{l}(pos:number, f:number, v:float)")
        w(f"gate{l}(P,F,S) :- token(P,_), ffnout(F), S = sum (AV*WV) : {{ a2_{l}(P,I,AV), gatew{l}(I,F,WV) }}.")
        w(f".decl up{l}(pos:number, f:number, v:float)")
        w(f"up{l}(P,F,S) :- token(P,_), ffnout(F), S = sum (AV*WV) : {{ a2_{l}(P,I,AV), upw{l}(I,F,WV) }}.")
        w(f".decl hid{l}(pos:number, f:number, v:float)")
        w(f"hid{l}(P,F, (G/(1.0+{E}^(0.0-G)))*U) :- gate{l}(P,F,G), up{l}(P,F,U).")
        # down_proj + residual
        w(f".decl down{l}(pos:number, d:number, v:float)")
        w(f"down{l}(P,D,S) :- token(P,_), dim_d(D), S = sum (HV*WV) : {{ hid{l}(P,F,HV), downw{l}(F,D,WV) }}.")
        w(f".decl {xout}(pos:number, d:number, v:float)")
        w(f"{xout}(P,D, XV+DV) :- {xmid}(P,D,XV), down{l}(P,D,DV).")
        w()
    # ---- final RMSNorm + unembed (last position) + argmax ----
    xN = f"x{N_LAYER}"
    w(f"// ---------- final norm + unembed (tied) + argmax ----------")
    w(".decl ssf(pos:number, s:float)")
    w(f"ssf(P,S) :- token(P,_), S = sum (V*V) : {{ {xN}(P,_,V) }}.")
    w(".decl xf(pos:number, d:number, v:float)")
    w(f"xf(P,D, V*(((SS/{DV})+{EPSV})^(-0.5))*G) :- {xN}(P,D,V), ssf(P,SS), normw(D,G).")
    w(".decl lastpos(p:number)")
    w("lastpos(P) :- P = max Q : { token(Q,_) }.")
    w(".decl logit(v:number, s:float)")
    w("logit(V,S) :- vocab(V), lastpos(LP), S = sum (XV*EV) : { xf(LP,D,XV), embed_w(V,D,EV) }.")
    w(".decl decide(v:number)")
    w("decide(V) :- logit(V,S), S = max S2 : { logit(_,S2) }.")
    w(".output decide")
    w(".output logit")
    with open(path, "w") as f: f.write("\n".join(L) + "\n")
    print(f"[emit] wrote {path}  ({len(L)} lines)")

if __name__ == "__main__":
    write_bundle()
    if not BIAS and not UNTIE:   # the Python reference emitter only covers the base path; fieldrun emits the rest
        emit_dl(os.path.join(OUT, "whole.dl"))
    # sanity: a couple of predictions + logit margins
    for ctx in ([3, 14, 7, 2, 29], [40, 1, 1, 9]):
        lg = forward_logits(ctx); top = np.argsort(lg)[::-1][:3]
        print(f"[ref] variant={VAR or 'base'} ctx={ctx} -> predict {int(top[0])}  top3 {[ (int(t), round(float(lg[t]),4)) for t in top]}")
