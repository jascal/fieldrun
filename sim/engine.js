// The Threx engine — a from-scratch forward pass of the tiny RoPE (Llama-style)
// model, plus the phrasebook (n-gram + recency + induction) and the route
// classifier. Vanilla JS, no deps; runs in Node (for validation) and the browser
// (the live runnable on the site). Mirrors sim/train.py exactly, which is in turn
// validated top-1 == the fieldrun binary, so every number here is the real one.
//
// It returns not just the logits but the full trace the visualization steps
// through: embeddings, per-head attention matrices, each head's and neuron's
// write into the residual stream, the final logits/softmax, and the DLA "chord"
// (each source's contribution c_j to the chosen token's logit, summing to it).
(function (root) {
'use strict';

// ---- small linear-algebra helpers (plain arrays) --------------------------
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const add = (a, b) => a.map((x, i) => x + b[i]);
const scale = (a, s) => a.map(x => x * s);
const matvec = (W, x) => W.map(row => dot(row, x));          // W:(out,in) · x:(in) -> (out)
const silu = x => x / (1 + Math.exp(-x));
function softmax(z) {
  const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
function rmsnorm(x, w, eps) {
  let ms = 0; for (const v of x) ms += v * v; ms /= x.length;
  const r = 1 / Math.sqrt(ms + eps);
  return x.map((v, i) => v * r * w[i]);
}
function argmax(z) { let bi = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[bi]) bi = i; return bi; }

// rotary: split-half (HF rotate_half). cos/sin are length head_dim at a position.
function ropeApply(v, cos, sin) {
  const hd = v.length, h = hd / 2, out = new Array(hd);
  for (let i = 0; i < h; i++) {        // rotate_half(x) = [-x2, x1]
    out[i] = v[i] * cos[i] - v[i + h] * sin[i];
    out[i + h] = v[i + h] * cos[i + h] + v[i] * sin[i + h];
  }
  return out;
}
function ropeTables(cfg, L) {
  const hd = cfg.head_dim, cos = [], sin = [];
  for (let p = 0; p < L; p++) {
    const c = new Array(hd), s = new Array(hd);
    for (let i = 0; i < hd / 2; i++) {
      const freq = p / Math.pow(cfg.theta, (2 * i) / hd);
      c[i] = c[i + hd / 2] = Math.cos(freq);
      s[i] = s[i + hd / 2] = Math.sin(freq);
    }
    cos.push(c); sin.push(s);
  }
  return { cos, sin };
}

// ---- the forward pass, instrumented ---------------------------------------
// Returns a rich trace for the LAST position (the prediction point):
//   embed, layers[ { heads:[{attn:[L], readPos, writeVec, ... }], mlp:{neurons..} } ],
//   resid (final residual at last pos), logits, probs, argmax,
//   sources: [{kind, label, layer, idx, write:[d], dla}]  (the DLA chord)
function forward(model, ids) {
  const cfg = model.config, d = cfg.d, nh = cfg.n_head, hd = cfg.head_dim, L = ids.length;
  const { cos, sin } = ropeTables(cfg, L);
  // residual stream x[pos][dim]; track per-source writes at the LAST position
  let x = ids.map(t => model.embed[t].slice());
  const sources = [{ kind: 'embed', label: 'token embedding', layer: -1, idx: -1,
                     write: model.embed[ids[L - 1]].slice() }];
  const layerTrace = [];

  for (let l = 0; l < cfg.n_layer; l++) {
    const Lw = model.layers[l];
    const hn = x.map(v => rmsnorm(v, Lw.in_ln, cfg.eps));
    // project q,k,v for every position, split into heads, apply rope
    const Q = [], K = [], V = [];
    for (let p = 0; p < L; p++) {
      const q = matvec(Lw.q, hn[p]), k = matvec(Lw.k, hn[p]), v = matvec(Lw.v, hn[p]);
      const qh = [], kh = [], vh = [];
      for (let h = 0; h < nh; h++) {
        qh.push(ropeApply(q.slice(h * hd, h * hd + hd), cos[p], sin[p]));
        kh.push(ropeApply(k.slice(h * hd, h * hd + hd), cos[p], sin[p]));
        vh.push(v.slice(h * hd, h * hd + hd));
      }
      Q.push(qh); K.push(kh); V.push(vh);
    }
    // attention at EVERY position (needed for the residual), trace the last pos
    const attnOut = x.map(() => new Array(d).fill(0));
    const headInfo = [];
    for (let p = 0; p < L; p++) {
      const concat = new Array(nh * hd).fill(0);
      for (let h = 0; h < nh; h++) {
        const sc = [];
        for (let j = 0; j <= p; j++) sc.push(dot(Q[p][h], K[j][h]) / Math.sqrt(hd));
        const a = softmax(sc);
        const ov = new Array(hd).fill(0);
        for (let j = 0; j <= p; j++) for (let t = 0; t < hd; t++) ov[t] += a[j] * V[j][h][t];
        for (let t = 0; t < hd; t++) concat[h * hd + t] = ov[t];
        if (p === L - 1) {
          // this head's write into the residual = W_o restricted to its slice
          const wv = new Array(d).fill(0);
          for (let o = 0; o < d; o++) { let s = 0; for (let t = 0; t < hd; t++) s += Lw.o[o][h * hd + t] * ov[t]; wv[o] = s; }
          let rp = 0; for (let j = 1; j <= p; j++) if (a[j] > a[rp]) rp = j;   // strongest attended pos
          headInfo.push({ head: h, attn: a.slice(), readPos: rp, readTok: ids[rp], write: wv, ov });
        }
      }
      attnOut[p] = matvec(Lw.o, concat);
    }
    x = x.map((v, p) => add(v, attnOut[p]));
    for (const hi of headInfo)
      sources.push({ kind: 'head', label: `L${l} head ${hi.head}`, layer: l, idx: hi.head,
                     write: hi.write, attn: hi.attn, readPos: hi.readPos, readTok: hi.readTok });

    // MLP (SwiGLU), per-position; trace per-neuron writes at the last pos
    const hn2 = x.map(v => rmsnorm(v, Lw.post_ln, cfg.eps));
    const neurons = [];
    for (let p = 0; p < L; p++) {
      const g = matvec(Lw.gate, hn2[p]), u = matvec(Lw.up, hn2[p]);
      const act = g.map((gi, n) => silu(gi) * u[n]);
      const mlpOut = matvec(Lw.down, act);
      x[p] = add(x[p], mlpOut);
      if (p === L - 1) {
        for (let n = 0; n < cfg.ffn; n++) {
          const wv = Lw.down.map(row => row[n] * act[n]);   // neuron n's residual write
          neurons.push({ neuron: n, act: act[n], write: wv });
        }
      }
    }
    for (const ne of neurons)
      sources.push({ kind: 'neuron', label: `L${l} neuron ${ne.neuron}`, layer: l, idx: ne.neuron,
                     write: ne.write, act: ne.act });
    layerTrace.push({ heads: headInfo, neurons });
  }

  // final norm + tied unembed
  const last = x[L - 1];
  let ms = 0; for (const v of last) ms += v * v; ms = Math.sqrt(ms / d + cfg.eps);
  const Utilde = model.embed.map(row => row.map((v, i) => v * model.norm[i])); // U_v ⊙ norm_weight
  const logits = Utilde.map(u => dot(u, last) / ms);
  const probs = softmax(logits);
  const pred = argmax(logits);

  // DLA: c_j^v = <Utilde_v, d_j> / ms ; sum over sources == logit_v (exact)
  for (const s of sources) s.dla = dot(Utilde[pred], s.write) / ms;
  const sortedChord = sources.map(s => ({ ...s, write: undefined })).sort((a, b) => b.dla - a.dla);
  const sumc = sources.reduce((a, s) => a + s.dla, 0);            // == logits[pred]
  const sq = sources.reduce((a, s) => a + s.dla * s.dla, 0);
  const PR = (sumc * sumc) / sq;                                   // participation ratio
  // margin geometry: runner-up and normalized margin (distance to the facet)
  const order = logits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const runner = order[1][1];
  let nrm = 0; for (let i = 0; i < d; i++) { const dd = Utilde[pred][i] - Utilde[runner][i]; nrm += dd * dd; }
  nrm = Math.sqrt(nrm);
  const margin = logits[pred] - logits[runner];
  const normMargin = margin / (nrm / ms);

  return { logits, probs, pred, runner, margin, normMargin, PR, chord: sortedChord,
           sources, layers: layerTrace, residual: last, recoveredLogit: sumc,
           ms, Utilde };   // ms = final-norm scale; Utilde = U_v ⊙ norm_weight (for projecting writes)
}

// ---- the phrasebook: n-gram store + recency + in-context induction --------
// Mirrors fieldrun's CandCfg/candidates + Store::predict so the route a prompt
// gets in the browser is the one the fieldrun binary assigns.
const CFG = { recent: 64, induction: 4, quad: 8, tri: 8, bi: 8, uni: 128 };

function ngramTop(store, ctx) {
  const n = ctx.length, key = (a) => a.join(',');
  if (n >= 3 && store.quad[key(ctx.slice(n - 3))]) return { tok: store.quad[key(ctx.slice(n - 3))][0], idiom: 'quad' };
  if (n >= 2 && store.tri[key(ctx.slice(n - 2))]) return { tok: store.tri[key(ctx.slice(n - 2))][0], idiom: 'tri' };
  if (n >= 1 && store.bi[String(ctx[n - 1])]) return { tok: store.bi[String(ctx[n - 1])][0], idiom: 'bi' };
  if (store.uni.length) return { tok: store.uni[0], idiom: 'uni' };
  return { tok: -1, idiom: 'none' };
}
function inductionCands(ctx, k) {
  // tokens following earlier recurrences of the suffix (tail length >=2, longest first)
  const out = [], n = ctx.length;
  for (let span = Math.min(4, n - 1); span >= 2 && out.length < k; span--) {
    const tail = ctx.slice(n - span);
    for (let i = 0; i + span < n - span + 1; i++) {
      let m = true; for (let t = 0; t < span; t++) if (ctx[i + t] !== tail[t]) { m = false; break; }
      if (m && i + span < n) { const f = ctx[i + span]; if (!out.includes(f)) out.push(f); }
    }
  }
  return out.slice(0, k);
}
function candidates(store, ctx) {
  const set = new Set(), n = ctx.length, add = t => { if (t >= 0) set.add(t); };
  const sources = {};
  // recency: last-N distinct context tokens
  const rec = []; for (let i = n - 1; i >= 0 && rec.length < CFG.recent; i--) if (!rec.includes(ctx[i])) rec.push(ctx[i]);
  rec.forEach(add); sources.recent = rec.slice();
  // induction
  const ind = inductionCands(ctx, CFG.induction); ind.forEach(add); sources.induction = ind;
  // n-grams
  const key = a => a.join(',');
  sources.quad = (n >= 3 && store.quad[key(ctx.slice(n - 3))]) ? store.quad[key(ctx.slice(n - 3))].slice(0, CFG.quad) : [];
  sources.tri = (n >= 2 && store.tri[key(ctx.slice(n - 2))]) ? store.tri[key(ctx.slice(n - 2))].slice(0, CFG.tri) : [];
  sources.bi = (n >= 1 && store.bi[String(ctx[n - 1])]) ? store.bi[String(ctx[n - 1])].slice(0, CFG.bi) : [];
  sources.uni = store.uni.slice(0, CFG.uni);
  for (const kk of ['quad', 'tri', 'bi', 'uni']) sources[kk].forEach(add);
  return { set, sources };
}
function route(store, ctx, argmaxTok) {
  const kb = ngramTop(store, ctx);
  const { set, sources } = candidates(store, ctx);
  const covered = set.has(argmaxTok);
  let r = kb.tok === argmaxTok ? 'RETRIEVED' : covered ? 'SELECTED' : 'COMPOSED';
  return { route: r, kb: kb.tok, idiom: kb.idiom, covered, candidates: set, sources };
}

// ---- model loader (weights.json shape) ------------------------------------
function loadModel(w) { return { config: w.config, embed: w.embed, norm: w.norm, layers: w.layers }; }

const API = { loadModel, forward, route, candidates, ngramTop, softmax, argmax };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
root.ThrexEngine = API;
})(typeof window !== 'undefined' ? window : globalThis);
