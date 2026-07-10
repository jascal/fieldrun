//! Jacobian-lens (J-lens) — an offline-fit, EMPIRICAL mid-stack read-out aid. Where the plain logit-lens reads an
//! intermediate residual by unembedding it directly (assuming `J_l = I`, the identity downstream map), the J-lens first
//! routes it through `J_l = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_l,t ]` — the layer's AVERAGED causal Jacobian to the
//! final residual — then unembeds. So a layer-`l` activation is scored by what the network is *disposed to make it emit*,
//! not by the identity-path guess the workspace note reports is noisy at mid layers (Anthropic, "Verbalizable
//! Representations as Global Workspace", transformer-circuits.pub/2026/workspace). `read(h_l) = softmax(W_U · norm(J_l h_l))`.
//!
//! This is a PROBE, tagged `empirical`: `J_l` is a first-order, context-averaged approximation. It never touches the
//! forward path or the faithfulness gate — it only re-reads captured residuals. `fit()` estimates `{J_l}` with a
//! finite-difference JVP (fieldrun owns the forward, so it can restart it from an arbitrary layer with a perturbed
//! residual — `Model::jlens_forward_from`); the estimator is the unbiased Hutchinson outer-product
//! `E_g[(J g) g^T] = J` for `g ~ N(0, I)`, central-differenced. `run_eval` compares the J-lens recursion trace against
//! the logit-lens one (resolve-layer + across-depth argmax stability) so the improvement is measured, not assumed.

use ndarray::{s, Array1, Array2};

use crate::model::Model;

const MAGIC: u32 = 0x314E4C4A; // "JLN1" LE — the {J_l} dump header

// ── reproducible Gaussian PRNG (splitmix64 + Box–Muller) — dependency-free so the estimator stays deterministic ──────
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1) from the top 24 bits.
    fn next_unit(&mut self) -> f32 {
        let m = (self.next_u64() >> 40) as f32; // 24 bits
        (m + 0.5) * (1.0 / 16_777_216.0) // shifted off the endpoints so ln() below is finite
    }
    /// One standard-normal draw (Box–Muller, cosine branch).
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Estimate `J_l = E_t[ mean_{t'≥t} ∂h_final,t' / ∂x0,t ]` at ONE layer from its captured post-block residual `x0`
/// (seq × d). `forward_from(x)` runs blocks `l+1..n_layer` on a (perturbed) `x` and returns the pre-final-norm residual
/// (seq × d). The estimator is the unbiased Hutchinson outer-product: for `g ~ N(0, I_d)`, a central-difference JVP gives
/// `d ≈ J g` (averaged over the downstream targets `t'≥t`), and `E_g[d g^T] = J`. Returns the `d × d` matrix with the
/// convention `J[j,i] = ∂out_j/∂in_i`, i.e. it is applied downstream as `J · resid_row`.
pub fn estimate_jacobian(
    x0: &Array2<f32>,
    src_positions: &[usize],
    forward_from: &dyn Fn(&Array2<f32>) -> Array2<f32>,
    probes: usize,
    step_rel: f32,
    rng: &mut Rng,
) -> Array2<f32> {
    let (seq, d) = (x0.nrows(), x0.ncols());
    let mut j = Array2::<f32>::zeros((d, d));
    let mut count = 0usize;
    for &t in src_positions {
        if t >= seq {
            continue;
        }
        // finite-difference step scaled to this position's residual: eps · ‖x0[t]‖ / √d ≈ step_rel · RMS(x0[t]).
        let norm_t = x0.row(t).iter().map(|v| v * v).sum::<f32>().sqrt();
        let eps = (step_rel * norm_t / (d as f32).sqrt()).max(1e-6);
        let n_tprime = (seq - t) as f32; // targets t' ∈ [t, seq)
        for _ in 0..probes {
            let g: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
            let mut xp = x0.clone();
            let mut xm = x0.clone();
            for i in 0..d {
                let dv = eps * g[i];
                xp[[t, i]] += dv;
                xm[[t, i]] -= dv;
            }
            let hp = forward_from(&xp);
            let hm = forward_from(&xm);
            // directional derivative d ≈ J g, averaged over the downstream targets t' ≥ t
            let mut dvec = vec![0f32; d];
            for tp in t..seq {
                for (jd, dval) in dvec.iter_mut().enumerate() {
                    *dval += (hp[[tp, jd]] - hm[[tp, jd]]) / (2.0 * eps);
                }
            }
            let inv = 1.0 / n_tprime;
            for v in dvec.iter_mut() {
                *v *= inv;
            }
            // accumulate the outer product d ⊗ g : J[j,i] += d[j]·g[i]
            for (jj, &dj) in dvec.iter().enumerate() {
                let jrow = jj * d;
                let jslice = j.as_slice_mut().unwrap();
                for ii in 0..d {
                    jslice[jrow + ii] += dj * g[ii];
                }
            }
            count += 1;
        }
    }
    if count > 0 {
        let inv = 1.0 / count as f32;
        j.mapv_inplace(|v| v * inv);
    }
    j
}

/// Evenly-spaced source positions in `[1, seq)` (skip the BOS at 0), capped at `max`. These are the `t` the fit perturbs.
fn sample_src(seq: usize, max: usize) -> Vec<usize> {
    if seq < 2 {
        return vec![];
    }
    let avail = seq - 1; // positions 1..seq
    let n = max.max(1).min(avail);
    if n == avail {
        return (1..seq).collect();
    }
    // spread n picks across 1..seq
    (0..n).map(|i| 1 + (i * (avail - 1)) / (n - 1).max(1)).collect()
}

pub struct FitCfg {
    pub probes: usize,
    pub step_rel: f32,
    pub max_seq: usize,             // truncate each prompt to this many tokens (0 = no cap)
    pub max_src: usize,             // source positions sampled per prompt
    pub seed: u64,
    pub ckpt: Option<(String, usize)>, // (path, every_n_prompts): write the running estimate so a crash leaves a usable dump
}

/// Turn the running accumulators into the finished `{J_l}` : average each fitted layer by its prompt count; every
/// other layer (including the last) is the identity, so it reads as the plain logit-lens.
fn finalize(acc: &[Option<Array2<f32>>], nfit: &[usize], d: usize) -> Vec<Array2<f32>> {
    (0..acc.len())
        .map(|l| match &acc[l] {
            Some(a) if nfit[l] > 0 => a.mapv(|v| v / nfit[l] as f32),
            _ => Array2::<f32>::eye(d),
        })
        .collect()
}

/// Fit `{J_l}` over a corpus for the requested `layers` (others returned as identity ⇒ they read as the plain
/// logit-lens). Returns exactly `n_layer` matrices. `progress(prompt_idx, layer)` is called after each layer of each
/// prompt. The last layer is always identity (its downstream map to the final residual is the identity by definition).
pub fn fit(
    model: &dyn Model,
    prompts: &[Vec<i64>],
    layers: &[usize],
    cfg: &FitCfg,
    mut progress: impl FnMut(usize, usize),
) -> Option<Vec<Array2<f32>>> {
    let mut acc: Vec<Option<Array2<f32>>> = Vec::new();
    let mut nfit: Vec<usize> = Vec::new();
    let mut n_layer = 0usize;
    let mut d = 0usize;
    let mut rng = Rng::new(cfg.seed);
    for (pi, ids0) in prompts.iter().enumerate() {
        let ids: Vec<i64> = if cfg.max_seq > 0 && ids0.len() > cfg.max_seq {
            ids0[..cfg.max_seq].to_vec()
        } else {
            ids0.clone()
        };
        if ids.len() < 3 {
            continue;
        }
        let resids = model.jlens_capture(&ids)?;
        if n_layer == 0 {
            n_layer = resids.len();
            d = resids.first().map(|r| r.ncols()).unwrap_or(0);
            acc = (0..n_layer).map(|_| None).collect();
            nfit = vec![0; n_layer];
        }
        let src = sample_src(ids.len(), cfg.max_src);
        for &l in layers {
            if l + 1 >= n_layer {
                continue; // last layer (and out-of-range) → identity, no fit
            }
            let x0 = &resids[l];
            let ff = |x: &Array2<f32>| model.jlens_forward_from(l, x).expect("arch wires jlens_forward_from");
            let jl = estimate_jacobian(x0, &src, &ff as &dyn Fn(&Array2<f32>) -> Array2<f32>, cfg.probes, cfg.step_rel, &mut rng);
            acc[l] = Some(match acc[l].take() {
                Some(a) => a + jl,
                None => jl,
            });
            nfit[l] += 1;
            progress(pi, l);
        }
        // periodic checkpoint: write the running estimate so a crash mid-run leaves a usable (fewer-prompt) dump
        if let Some((path, every)) = &cfg.ckpt {
            if *every > 0 && (pi + 1) % *every == 0 {
                let _ = save(path, &finalize(&acc, &nfit, d), d);
            }
        }
    }
    if n_layer == 0 || d == 0 {
        return None;
    }
    Some(finalize(&acc, &nfit, d))
}

/// Serialize `{J_l}` : `MAGIC(u32) | n_layer(u32) | d(u32) | n_layer × (d·d f32 LE, row-major)`.
pub fn save(path: &str, jmats: &[Array2<f32>], d: usize) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(12 + jmats.len() * d * d * 4);
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&(jmats.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    for m in jmats {
        for &v in m.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &buf)
}

/// Inverse of `save`. Returns `(d, {J_l})`.
pub fn load(path: &str) -> std::io::Result<(usize, Vec<Array2<f32>>)> {
    let bytes = std::fs::read(path)?;
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 12 {
        return Err(err("jlens file too short"));
    }
    let rd = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    if rd(0) != MAGIC {
        return Err(err("bad jlens magic"));
    }
    let nl = rd(4) as usize;
    let d = rd(8) as usize;
    let need = 12 + nl * d * d * 4;
    if bytes.len() != need {
        return Err(err("jlens size mismatch vs header"));
    }
    let mut mats = Vec::with_capacity(nl);
    let mut o = 12;
    for _ in 0..nl {
        let mut vals = Vec::with_capacity(d * d);
        for _ in 0..(d * d) {
            vals.push(f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]));
            o += 4;
        }
        mats.push(Array2::from_shape_vec((d, d), vals).map_err(|_| err("jlens reshape"))?);
    }
    Ok((d, mats))
}

/// Blend a fitted matrix toward the identity: `J' = (1−λ)·I + λ·J`. `λ = 1` leaves it unchanged; `λ = 0` collapses it
/// to the identity (i.e. back to the plain logit-lens). This is the shrinkage knob: an under-fit, noise-dominated `J_l`
/// (per-entry noise `σ` inflates every one of the `d²` entries) degrades GRACEFULLY toward the logit-lens as `λ → 0`,
/// instead of scrambling the read-out. Applied at eval time, so `λ` sweeps need no refit.
pub fn shrink_toward_identity(m: &Array2<f32>, lambda: f32) -> Array2<f32> {
    let mut out = m.mapv(|v| v * lambda);
    for i in 0..out.nrows().min(out.ncols()) {
        out[[i, i]] += 1.0 - lambda;
    }
    out
}

// ── low-rank denoising of J ──────────────────────────────────────────────────────────────────────────────────────────
// The paper reports the verbalizable J-space is only ~5-10% of activation variance ⇒ the useful part of `J_l - I` is
// LOW RANK. The Hutchinson estimate is full-rank: signal in a few top singular directions + ~σ√d noise smeared across
// all d. SVD-truncating `J_l - I` to rank k keeps the signal subspace and discards the noise tail — a free denoise of an
// ALREADY-fit `J` (no extra probes). No LAPACK in fieldrun, so this is a dependency-free randomized SVD.

/// Modified Gram–Schmidt: orthonormalize the columns of `y` (d×r) in place-ish, returning a d×r matrix with orthonormal
/// columns (degenerate columns zeroed).
fn mgs(y: &Array2<f32>) -> Array2<f32> {
    let (d, r) = (y.nrows(), y.ncols());
    // Keep only numerically-independent columns. The threshold is RELATIVE (residual vs the column's own norm) — an
    // absolute cutoff misfires after subspace iteration inflates the scale, normalizing round-off into spurious unit
    // vectors and breaking orthonormality. Dependent columns are dropped, not zeroed, so Q stays orthonormal.
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for j in 0..r {
        let mut col: Vec<f32> = (0..d).map(|row| y[[row, j]]).collect();
        let n0 = col.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n0 < 1e-30 {
            continue;
        }
        for q in &cols {
            let dot: f32 = col.iter().zip(q).map(|(a, b)| a * b).sum();
            for (c, &qq) in col.iter_mut().zip(q) {
                *c -= dot * qq;
            }
        }
        let n1 = col.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n1 < 1e-6 * n0 {
            continue; // dependent on the columns already kept → drop it
        }
        for c in col.iter_mut() {
            *c /= n1;
        }
        cols.push(col);
    }
    let mut q = Array2::<f32>::zeros((d, cols.len()));
    for (jc, col) in cols.iter().enumerate() {
        for (row, &v) in col.iter().enumerate() {
            q[[row, jc]] = v;
        }
    }
    q
}

/// Eigendecomposition of a small SYMMETRIC matrix by cyclic Jacobi rotations. Returns (eigenvalues descending,
/// eigenvectors as columns in the same order). Used only on the tiny r×r `BBᵀ` in the randomized SVD.
fn sym_eig_jacobi(a: &Array2<f32>) -> (Vec<f32>, Array2<f32>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<f32>::eye(n);
    for _ in 0..100 {
        let mut off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[[p, q]] * m[[p, q]];
            }
        }
        if off <= 1e-18 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-20 {
                    continue;
                }
                let theta = (m[[q, q]] - m[[p, p]]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..n {
                    let (mip, miq) = (m[[i, p]], m[[i, q]]);
                    m[[i, p]] = c * mip - s * miq;
                    m[[i, q]] = s * mip + c * miq;
                }
                for i in 0..n {
                    let (mpi, mqi) = (m[[p, i]], m[[q, i]]);
                    m[[p, i]] = c * mpi - s * mqi;
                    m[[q, i]] = s * mpi + c * mqi;
                }
                for i in 0..n {
                    let (vip, viq) = (v[[i, p]], v[[i, q]]);
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }
    let diag: Vec<f32> = (0..n).map(|i| m[[i, i]]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| diag[b].total_cmp(&diag[a]));
    let evals = idx.iter().map(|&i| diag[i]).collect();
    let mut evecs = Array2::<f32>::zeros((n, n));
    for (col, &i) in idx.iter().enumerate() {
        for row in 0..n {
            evecs[[row, col]] = v[[row, i]];
        }
    }
    (evals, evecs)
}

/// Best rank-`k` approximation of `a` (d×d) by randomized SVD: sketch the range with `k+oversample` random probes and
/// `power` subspace iterations, orthonormalize (Φ = top-k left singular vectors), then project `A_k = Φ (Φᵀ A)`.
/// `k == 0` or `k >= d` ⇒ no truncation (returns a clone).
/// The top-`k` left-singular subspace of `a` (d×keff, orthonormal columns; keff = min(k, numerical rank)) by randomized
/// range-finding. For a symmetric PSD `a` these are its top-k eigenvectors. Empty (d×0) if the sketch is rank-0.
fn topk_subspace(a: &Array2<f32>, k: usize, rng: &mut Rng, power: usize) -> Array2<f32> {
    let d = a.nrows();
    let r = (k + 8).min(d); // oversampling for a stable range estimate
    let mut omega = Array2::<f32>::zeros((d, r));
    for x in omega.iter_mut() {
        *x = rng.gauss();
    }
    let mut y = a.dot(&omega); // d×r range sketch
    for _ in 0..power {
        let aty = a.t().dot(&y); // d×r  (Aᵀ Y)
        y = a.dot(&aty); //          A (Aᵀ Y) — sharpen toward the dominant subspace
    }
    let q = mgs(&y); // d×q' orthonormal (q' = numerical rank of the sketch, ≤ r)
    let qn = q.ncols();
    if qn == 0 {
        return Array2::<f32>::zeros((d, 0));
    }
    let keff = k.min(qn); // if the sketch is lower-rank than k, return that lower rank
    let b = q.t().dot(a); // q'×d
    let s = b.dot(&b.t()); // q'×q' = B Bᵀ (symmetric PSD)
    let (_ev, w) = sym_eig_jacobi(&s); // eigenvectors, descending
    let wk = w.slice(s![.., 0..keff]).to_owned(); // q'×keff
    q.dot(&wk) // d×keff — the top left singular vectors
}

fn truncate_rank(a: &Array2<f32>, k: usize, rng: &mut Rng, power: usize) -> Array2<f32> {
    let d = a.nrows();
    if k == 0 || k >= d {
        return a.clone();
    }
    let phi = topk_subspace(a, k, rng, power); // d×keff top-k left singular vectors
    if phi.ncols() == 0 {
        return Array2::<f32>::zeros((d, a.ncols()));
    }
    phi.dot(&phi.t().dot(a)) // A_k = Φ Φᵀ A (rank keff)
}

/// Logit-weighted low-rank denoise (G′): keep the part of `J − I` that most affects the OUTPUT LOGITS, not the part with
/// the largest raw magnitude. The read is `Wᵤ·norm(J·r)`, so the output space is weighted by the unembed Gram `M = WᵤᵀWᵤ`.
/// We keep `L = J−I`'s action on the top-`k` INPUT directions whose L-image carries the most logit energy — the top-k
/// eigenvectors `Φ` of `A = Lᵀ M L` — giving `L_k = L Φ Φᵀ`, then `J_k = I + L_k`. This matches the paper's J-space
/// (defined by the `Wᵤ·J` directions), unlike a plain SVD of `J−I` which keeps the biggest *residual* reshaping.
/// `m` is the (approximate) `d×d` unembed Gram. `k == 0` or `k >= d` ⇒ unchanged.
pub fn rank_reduce_logit(j: &Array2<f32>, k: usize, m: &Array2<f32>, seed: u64) -> Array2<f32> {
    let d = j.nrows();
    if k == 0 || k >= d {
        return j.clone();
    }
    let mut l = j.clone();
    for i in 0..d {
        l[[i, i]] -= 1.0; // L = J − I
    }
    let a = l.t().dot(&m.dot(&l)); // Lᵀ M L (d×d, symmetric PSD) — logit-energy metric on the input directions
    let mut rng = Rng::new(seed);
    let phi = topk_subspace(&a, k, &mut rng, 2); // top-k logit-relevant input directions
    if phi.ncols() == 0 {
        return j.clone();
    }
    let lk = l.dot(&phi).dot(&phi.t()); // L_k = (L Φ) Φᵀ — keep L on the relevant input subspace
    let mut jk = lk;
    for i in 0..d {
        jk[[i, i]] += 1.0; // J_k = I + L_k
    }
    jk
}

/// Low-rank-denoise a fitted Jacobian: keep only the rank-`k` part of `J - I` (the paper's low-dim J-space), then add
/// `I` back. `k == 0` or `k >= d` ⇒ unchanged. Seeded internally for reproducibility.
pub fn rank_reduce(j: &Array2<f32>, k: usize, seed: u64) -> Array2<f32> {
    let d = j.nrows();
    if k == 0 || k >= d {
        return j.clone();
    }
    let mut l = j.clone();
    for i in 0..d {
        l[[i, i]] -= 1.0; // L = J - I
    }
    let mut rng = Rng::new(seed);
    let mut jk = truncate_rank(&l, k, &mut rng, 2);
    for i in 0..d {
        jk[[i, i]] += 1.0; // J' = I + L_k
    }
    jk
}

/// Frobenius distance of `J_l` from the identity — a cheap "how far from the logit-lens is this layer's map" readout.
pub fn dist_from_identity(m: &Array2<f32>) -> f32 {
    let d = m.nrows();
    let mut s = 0f32;
    for i in 0..d {
        for k in 0..m.ncols() {
            let e = m[[i, k]] - if i == k { 1.0 } else { 0.0 };
            s += e * e;
        }
    }
    s.sqrt()
}

// ── .npz export: put {J_l} on the numpy channel pil consumes (fieldrun_io.py) without adding a zip/numpy dep ──────────
// A `.npz` is just a ZIP (STORED, no compression — what `np.savez` emits) of `.npy` members. We hand-roll both so the
// only dependency stays `serde`; the round-trip is verified against real numpy. See `run_export`.

/// CRC-32/IEEE (the checksum every ZIP entry carries). Bit-at-a-time — this runs a handful of times per export, so a
/// lookup table isn't worth the static. Verified against the standard vector crc32("123456789") = 0xCBF43926.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// One NumPy `.npy` v1.0 buffer: magic + version + a 64-byte-aligned header dict + the raw C-order data. `descr` is a
/// NumPy dtype string (e.g. `<f4`, `<i4`); `data` must already be little-endian in row-major order.
fn npy_bytes(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header0 = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // total preamble (10) + header string + trailing '\n' must be a multiple of 64
    let need = 11 + header0.len();
    let pad = (64 - (need % 64)) % 64;
    let mut header = header0;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Pack members into a STORED (uncompressed) ZIP — byte-compatible with `np.load` on a `numpy.savez` archive.
fn npz_write(path: &str, members: &[(&str, Vec<u8>)]) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();
    for (name, data) in members {
        let off = buf.len() as u32;
        let crc = crc32(data);
        let sz = data.len() as u32;
        let nl = name.len() as u16;
        // local file header
        buf.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // method = stored
        buf.extend_from_slice(&0u16.to_le_bytes()); // mod time
        buf.extend_from_slice(&0u16.to_le_bytes()); // mod date
        buf.extend_from_slice(&crc.to_le_bytes());
        buf.extend_from_slice(&sz.to_le_bytes()); // compressed size
        buf.extend_from_slice(&sz.to_le_bytes()); // uncompressed size
        buf.extend_from_slice(&nl.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // extra len
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(data);
        // central directory record
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&0u16.to_le_bytes()); // method
        central.extend_from_slice(&0u16.to_le_bytes()); // time
        central.extend_from_slice(&0u16.to_le_bytes()); // date
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&sz.to_le_bytes());
        central.extend_from_slice(&sz.to_le_bytes());
        central.extend_from_slice(&nl.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra
        central.extend_from_slice(&0u16.to_le_bytes()); // comment
        central.extend_from_slice(&0u16.to_le_bytes()); // disk start
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        central.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        central.extend_from_slice(&off.to_le_bytes()); // local header offset
        central.extend_from_slice(name.as_bytes());
    }
    let cd_off = buf.len() as u32;
    let cd_size = central.len() as u32;
    buf.extend_from_slice(&central);
    // end of central directory
    buf.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // this disk
    buf.extend_from_slice(&0u16.to_le_bytes()); // disk with cd
    buf.extend_from_slice(&(members.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(members.len() as u16).to_le_bytes());
    buf.extend_from_slice(&cd_size.to_le_bytes());
    buf.extend_from_slice(&cd_off.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // comment len
    std::fs::write(path, &buf)
}

/// Write `{J_l}` to an `.npz` with `J` = float32 [n_layer, d, d] (row-major) and `fitted` = int32 [n_layer] (1 where a
/// non-identity Jacobian was fit). Returns the fitted-layer indices.
pub fn export_npz(path: &str, jmats: &[Array2<f32>], d: usize) -> std::io::Result<Vec<usize>> {
    let n = jmats.len();
    let fit_flag: Vec<bool> = jmats.iter().map(|m| dist_from_identity(m) > 1e-6).collect();
    let mut jdata: Vec<u8> = Vec::with_capacity(n * d * d * 4);
    for m in jmats {
        for &v in m.iter() {
            jdata.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut fdata: Vec<u8> = Vec::with_capacity(n * 4);
    for &f in &fit_flag {
        fdata.extend_from_slice(&(f as i32).to_le_bytes());
    }
    npz_write(path, &[("J.npy", npy_bytes("<f4", &[n, d, d], &jdata)), ("fitted.npy", npy_bytes("<i4", &[n], &fdata))])?;
    Ok((0..n).filter(|&l| fit_flag[l]).collect())
}

/// `--jlens-export <out.npz>`: transcode the internal JLN1 `{J_l}` dump onto the numpy channel pil's `fieldrun_io.py`
/// consumes — a stored-zip `.npz` (`J`/`fitted` arrays) plus a `.meta.json` sidecar. Pure file transcode: no model or
/// tokenizer, so it runs in the lean build too. `inp` = the JLN1 file, `out` = the `.npz` path.
pub fn run_export(inp: &str, out: &str) {
    let (d, jmats) = match load(inp) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[jlens] --jlens-export: load {inp}: {e} — run --jlens-fit first (or pass --jlens-in <file>)");
            return;
        }
    };
    let fitted = match export_npz(out, &jmats, d) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[jlens] --jlens-export: write {out}: {e}");
            return;
        }
    };
    let meta_path = out.strip_suffix(".npz").map(|s| format!("{s}.meta.json")).unwrap_or_else(|| format!("{out}.meta.json"));
    let meta = serde_json::json!({
        "format": "fieldrun-jlens-v1",
        "source": inp,
        "n_layer": jmats.len(),
        "d": d,
        "dtype": "float32",
        "arrays": {
            "J": "[n_layer, d, d] averaged causal Jacobian  E_{t, t'>=t, prompt}[ d h_final,t' / d h_l,t ]",
            "fitted": "[n_layer] int32: 1 if this layer was fit, 0 = identity (reads as the plain logit-lens)"
        },
        "apply": "route a layer-l residual r through  J[l] @ r   (numpy: r @ J[l].T), then the model's final norm + unembed",
        "capture_point": "h_l = the POST-block residual of layer l (after the attn+MLP residual add, PRE final-norm); h_final = the post-last-block residual (pre final-norm), so J[n_layer-1] = I",
        "fitted_layers": fitted,
        "note": "EMPIRICAL readout: a first-order, context-averaged approximation of the downstream map. NOT faithfulness-gated; off the forward path."
    });
    let _ = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default());
    eprintln!(
        "[jlens] --jlens-export: wrote {out}  (J [{n},{d},{d}] f32 + fitted[{n}]; {} fitted layers) + {meta_path}",
        fitted.len(),
        n = jmats.len()
    );
}

/// The model OUTPUT tensors pil's J-lens sweep needs alongside `{J_l}`: the unembedding `U` (V,d) whose rows score
/// tokens, and the final-norm gain `gamma` (d,). `norm_type` tells pil whether its folded-basis conjugation
/// `diag(gamma) J diag(1/gamma)` is EXACT ("rmsnorm") or approximate ("layernorm": omits the mean-centering rank-1
/// term + the ln_f bias). Model-constant, EMPIRICAL, off the forward path / faithfulness gate.
pub struct UnembedExport {
    pub u: Array2<f32>,          // (V, d) unembedding rows; U[id] scores token id
    pub gamma: Vec<f32>,         // (d,) final-norm gain
    pub norm_type: &'static str, // "rmsnorm" | "layernorm"
    pub tied: bool,
}

/// `--tensors-export <out.npz>`: write the model's unembedding `U` (V,d) and final-norm gain `gamma` (d,) onto the numpy
/// channel — the two model constants pil's `experiments/jlens_correction_sweep.py` needs (`--U` / `--gamma`) alongside a
/// `--jlens-export` `{J_l}`. Unlike `--jlens-export` this needs the LOADED model (it reads the weights), not a
/// transcode. Writes a stored-zip `.npz` (`U`/`gamma` arrays) + a `.meta.json` sidecar (`norm_type`, `tied`, apply).
pub fn run_tensors_export(model: &dyn Model, out: &str) {
    let ex = match model.export_unembed() {
        Some(x) => x,
        None => {
            eprintln!("[jlens] --tensors-export: this arch does not expose the unembedding (rope / neox only)");
            return;
        }
    };
    let (v, d) = (ex.u.nrows(), ex.u.ncols());
    let mut udata: Vec<u8> = Vec::with_capacity(v * d * 4);
    for &val in ex.u.iter() {
        udata.extend_from_slice(&val.to_le_bytes());
    }
    let mut gdata: Vec<u8> = Vec::with_capacity(d * 4);
    for &val in &ex.gamma {
        gdata.extend_from_slice(&val.to_le_bytes());
    }
    let members = [
        ("U.npy", npy_bytes("<f4", &[v, d], &udata)),
        ("gamma.npy", npy_bytes("<f4", &[d], &gdata)),
    ];
    if let Err(e) = npz_write(out, &members) {
        eprintln!("[jlens] --tensors-export: write {out}: {e}");
        return;
    }
    let meta_path = out.strip_suffix(".npz").map(|s| format!("{s}.meta.json")).unwrap_or_else(|| format!("{out}.meta.json"));
    let meta = serde_json::json!({
        "format": "fieldrun-tensors-v1",
        "vocab": v,
        "d": d,
        "dtype": "float32",
        "tied": ex.tied,
        "norm_type": ex.norm_type,
        "arrays": {
            "U": "[vocab, d] unembedding rows; U[id] scores token id (rows indexed by token id)",
            "gamma": "[d] final-norm gain"
        },
        "apply": "pil jcorrect_sources(gamma=gamma) forms the folded-basis operator diag(gamma) J diag(1/gamma)",
        "gamma_exact": ex.norm_type == "rmsnorm",
        "note": "gamma-conjugation is EXACT for rmsnorm; for layernorm it omits the mean-centering rank-1 term and the ln_f bias (approximate). Feed to experiments/jlens_correction_sweep.py --U/--gamma."
    });
    let _ = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default());
    eprintln!(
        "[jlens] --tensors-export: wrote {out}  (U [{v},{d}] f32 + gamma[{d}]; norm={norm}, tied={tied}) + {meta_path}",
        norm = ex.norm_type,
        tied = ex.tied
    );
}

// ── CLI drivers (need the tokenizer → api build, like the rest of the --recursion-explain probe surface) ─────────────
#[cfg(feature = "api")]
mod cli {
    use super::*;
    use crate::api::TextGen;
    use crate::{flag, has_flag};

    /// A handful of generic prompts so `--jlens-fit` runs out of the box. A real fit should pass a ~1k-prompt corpus via
    /// `--jlens-corpus` (one prompt per line) — the whole point of `J_l` is the average over MANY contexts.
    const DEFAULT_CORPUS: &[&str] = &[
        "The capital of France is Paris, a city on the river Seine.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "She opened the old wooden door and stepped into the quiet room.",
        "In mathematics, a prime number has exactly two distinct divisors.",
        "The stock market fell sharply after the central bank raised rates.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into sugar.",
        "He packed his bags, locked the house, and drove to the airport.",
        "The theory predicts that the two particles remain correlated at a distance.",
    ];

    fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
        flag(args, name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    fn parse_f32(args: &[String], name: &str, default: f32) -> f32 {
        flag(args, name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    fn corpus(args: &[String], tg: &TextGen, ids: &[i64]) -> Vec<Vec<i64>> {
        if let Some(path) = flag(args, "--jlens-corpus") {
            match std::fs::read_to_string(path) {
                Ok(txt) => {
                    let ps: Vec<Vec<i64>> = txt
                        .lines()
                        .map(|l| l.trim())
                        .filter(|l| !l.is_empty())
                        .map(|l| tg.encode(l, false))
                        .filter(|v| v.len() >= 3)
                        .collect();
                    if !ps.is_empty() {
                        return ps;
                    }
                    eprintln!("[jlens] --jlens-corpus {path}: no usable lines, falling back to the default corpus");
                }
                Err(e) => eprintln!("[jlens] --jlens-corpus {path}: {e}; falling back to the default corpus"),
            }
        }
        // else: the loaded --text/--ids context if it's long enough, otherwise the built-in default corpus.
        if ids.len() >= 3 {
            return vec![ids.to_vec()];
        }
        DEFAULT_CORPUS.iter().map(|s| tg.encode(s, false)).collect()
    }

    fn parse_layers(args: &[String], n_layer: usize) -> Vec<usize> {
        match flag(args, "--jlens-layers") {
            None | Some("all") => (0..n_layer.saturating_sub(1)).collect(), // every layer but the last (= identity)
            Some(spec) => spec.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).filter(|&l| l + 1 < n_layer).collect(),
        }
    }

    /// `--recursion-explain --jlens-fit`: estimate `{J_l}` over a corpus and write the dump. Knobs: `--jlens-out`,
    /// `--jlens-corpus`, `--jlens-probes`, `--jlens-step`, `--jlens-max-seq`, `--jlens-max-src`, `--jlens-seed`,
    /// `--jlens-layers a,b,c|all`.
    pub fn run_fit(args: &[String], model: &dyn Model, tg: &Option<TextGen>, stem: &str, ids: &[i64]) {
        let Some(tg) = tg.as_ref() else {
            eprintln!("[jlens] --jlens-fit needs a tokenizer (TextGen); none loaded for this bundle");
            return;
        };
        let out = flag(args, "--jlens-out").map(|s| s.to_string()).unwrap_or_else(|| format!("{stem}.jlens"));
        let prompts = corpus(args, tg, ids);
        // shapes from one capture
        let probe0 = prompts.iter().find(|p| p.len() >= 3);
        let Some(probe0) = probe0 else {
            eprintln!("[jlens] no prompt with ≥3 tokens; nothing to fit");
            return;
        };
        let Some(resids) = model.jlens_capture(probe0) else {
            eprintln!("[jlens] this arch does not implement jlens_capture / jlens_forward_from (rope family only, for now)");
            return;
        };
        let (n_layer, d) = (resids.len(), resids[0].ncols());
        let layers = parse_layers(args, n_layer);
        let ckpt_every = parse_usize(args, "--jlens-ckpt-every", 10);
        let cfg = FitCfg {
            probes: parse_usize(args, "--jlens-probes", 64),
            step_rel: parse_f32(args, "--jlens-step", 0.05),
            max_seq: parse_usize(args, "--jlens-max-seq", 48),
            max_src: parse_usize(args, "--jlens-max-src", 4),
            seed: parse_usize(args, "--jlens-seed", 1) as u64,
            ckpt: if ckpt_every > 0 { Some((out.clone(), ckpt_every)) } else { None },
        };
        eprintln!(
            "[jlens] fit: {} prompts · d={d} · {n_layer} layers (fitting {} of them) · probes={} step={} max_seq={} max_src={} seed={} · checkpoint every {ckpt_every} prompts → {out}",
            prompts.len(), layers.len(), cfg.probes, cfg.step_rel, cfg.max_seq, cfg.max_src, cfg.seed
        );
        let total = prompts.len();
        let jmats = fit(model, &prompts, &layers, &cfg, |pi, l| {
            if l == *layers.last().unwrap_or(&0) {
                eprint!("\r[jlens]   prompt {}/{total}   ", pi + 1);
            }
        });
        let Some(jmats) = jmats else {
            eprintln!("\n[jlens] fit produced no matrices (arch unsupported or corpus empty)");
            return;
        };
        eprintln!();
        match save(&out, &jmats, d) {
            Ok(()) => {
                eprintln!("[jlens] wrote {out}  ({} matrices × {d}×{d} f32 = {:.1} MB)", jmats.len(), (jmats.len() * d * d * 4) as f64 / 1e6);
                // report each fitted layer's departure from the identity (logit-lens): larger ⇒ the downstream map
                // reshapes this layer's read-out more, i.e. where the J-lens most differs from the logit-lens.
                eprintln!("[jlens] ‖J_l − I‖_F by layer (fitted layers; * = larger reshaping):");
                let dists: Vec<(usize, f32)> = layers.iter().map(|&l| (l, dist_from_identity(&jmats[l]))).collect();
                let maxd = dists.iter().map(|&(_, x)| x).fold(0f32, f32::max).max(1e-6);
                for (l, dv) in dists {
                    let bar = "*".repeat(((dv / maxd) * 30.0) as usize);
                    eprintln!("  L{l:>2}  {dv:>8.3}  {bar}");
                }
            }
            Err(e) => eprintln!("[jlens] write {out}: {e}"),
        }
    }

    fn flips(r: &crate::model::RecPos) -> usize {
        r.lens_full.windows(2).filter(|w| w[0].1 != w[1].1).count()
    }
    fn stab(r: &crate::model::RecPos) -> f32 {
        // fraction of layers from resolve..end that still read final_top1 (commitment after the first resolve)
        let from = r.resolve_layer.saturating_sub(1);
        let tail = &r.lens_full[from.min(r.lens_full.len())..];
        if tail.is_empty() {
            return 1.0;
        }
        tail.iter().filter(|(_, t)| *t == r.final_top1).count() as f32 / tail.len() as f32
    }

    struct Agg {
        rl: f32,
        rj: f32,
        fl: f32,
        fj: f32,
        sl: f32,
        sj: f32,
        earlier: usize,
        later: usize,
        same: usize,
    }
    fn aggregate(trace_l: &[crate::model::RecPos], trace_j: &[crate::model::RecPos]) -> Agg {
        let (mut rl, mut rj, mut fl, mut fj, mut sl, mut sj) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
        let (mut earlier, mut later, mut same) = (0usize, 0usize, 0usize);
        for (a, b) in trace_l.iter().zip(trace_j.iter()) {
            rl += a.resolve_layer as f32;
            rj += b.resolve_layer as f32;
            fl += flips(a) as f32;
            fj += flips(b) as f32;
            sl += stab(a);
            sj += stab(b);
            match b.resolve_layer.cmp(&a.resolve_layer) {
                std::cmp::Ordering::Less => earlier += 1,
                std::cmp::Ordering::Greater => later += 1,
                std::cmp::Ordering::Equal => same += 1,
            }
        }
        let n = trace_l.len().max(1) as f32;
        Agg { rl: rl / n, rj: rj / n, fl: fl / n, fj: fj / n, sl: sl / n, sj: sj / n, earlier, later, same }
    }

    /// Approximate the unembed Gram `M = WᵤᵀWᵤ` (d×d) by sampling `s` random unembed rows and forming `Uₛᵀ Uₛ` (one
    /// matmul). Unscaled — only its eigen-directions feed the logit-weighted truncation. `None` if the arch lacks
    /// `unembed_row`/`logits`. The Gram is model-only (independent of the context), so `s` rows is plenty for the metric.
    fn sample_unembed_gram(model: &dyn Model, ids: &[i64], s: usize) -> Option<Array2<f32>> {
        let vocab = model.logits(ids)?.len();
        let s = s.min(vocab);
        let mut rng = Rng::new(7);
        let (mut rows, mut d, mut n) = (Vec::<f32>::new(), 0usize, 0usize);
        for _ in 0..s {
            let v = (rng.next_u64() % vocab as u64) as usize;
            if let Some(u) = model.unembed_row(v) {
                if d == 0 {
                    d = u.len();
                }
                if u.len() == d {
                    rows.extend_from_slice(&u);
                    n += 1;
                }
            }
        }
        if d == 0 || n == 0 {
            return None;
        }
        let us = Array2::from_shape_vec((n, d), rows).ok()?;
        Some(us.t().dot(&us)) // d×d
    }

    /// `--recursion-explain --jlens-eval`: read the context through BOTH lenses and report where the J-lens resolves
    /// the final token earlier and/or reads more stably across depth than the plain logit-lens. Knobs: `--jlens-in`,
    /// `--jlens-shrink λ|λ1,λ2,…` (shrinkage toward the logit-lens; default `1.0` = raw `J`, sweep to find the best λ).
    pub fn run_eval(args: &[String], model: &dyn Model, tg: &Option<TextGen>, stem: &str, ids: &[i64]) {
        let Some(tg) = tg.as_ref() else {
            eprintln!("[jlens] --jlens-eval needs a tokenizer (TextGen); none loaded for this bundle");
            return;
        };
        let inp = flag(args, "--jlens-in").map(|s| s.to_string()).unwrap_or_else(|| format!("{stem}.jlens"));
        let (d, jmats) = match load(&inp) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("[jlens] load {inp}: {e} — run `--jlens-fit` first");
                return;
            }
        };
        if ids.len() < 3 {
            eprintln!("[jlens] need a context of ≥3 tokens (pass --text or --ids)");
            return;
        }
        let Some(trace_l) = model.recursion_trace_lens(ids, None) else {
            eprintln!("[jlens] this arch has no recursion_trace_lens (rope family only, for now)");
            return;
        };
        let lambdas: Vec<f32> = flag(args, "--jlens-shrink")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![1.0]);
        // Low-rank denoise each J BEFORE shrinking (applied once, reused across the λ sweep). Two modes:
        //   --jlens-logit-rank k : keep the k directions of J−I most relevant to the OUTPUT logits (weighted by the
        //                          unembed Gram M = WᵤᵀWᵤ, sampled) — the paper's J-space (Wᵤ·J).  [G′, preferred]
        //   --jlens-rank k       : keep the top-k of J−I in the raw metric (plain SVD denoise).
        // 0 (default) = no truncation.
        let logit_rank: usize = flag(args, "--jlens-logit-rank").and_then(|s| s.parse().ok()).unwrap_or(0);
        let rank: usize = flag(args, "--jlens-rank").and_then(|s| s.parse().ok()).unwrap_or(0);
        let jmats: Vec<Array2<f32>> = if logit_rank > 0 {
            match sample_unembed_gram(model, ids, 4096) {
                Some(m) => {
                    eprintln!("[jlens] logit-weighted low-rank: keeping rank {logit_rank} (Wᵤ-relevant) of each J−I");
                    jmats.iter().map(|j| rank_reduce_logit(j, logit_rank, &m, 1)).collect()
                }
                None => {
                    eprintln!("[jlens] --jlens-logit-rank: arch lacks unembed_row/logits — using full J");
                    jmats
                }
            }
        } else if rank > 0 {
            eprintln!("[jlens] plain low-rank denoise: keeping rank {rank} of each J−I");
            jmats.iter().map(|m| rank_reduce(m, rank, 1)).collect()
        } else {
            jmats
        };
        let nl = trace_l.first().map(|r| r.n_layer).unwrap_or(0);
        let dec = |t: i64| tg.token_label(t);
        let rank_s = if logit_rank > 0 {
            format!(" · logit-rank {logit_rank}")
        } else if rank > 0 {
            format!(" · rank {rank}")
        } else {
            String::new()
        };
        println!("[jlens] eval · d={d} · {nl} layers · {} positions · J-lens vs logit-lens · shrink λ∈{lambdas:?}{rank_s}", trace_l.len());

        for &lam in &lambdas {
            let shrunk: Vec<Array2<f32>> = jmats.iter().map(|m| shrink_toward_identity(m, lam)).collect();
            let Some(trace_j) = model.recursion_trace_lens(ids, Some(shrunk.as_slice())) else {
                eprintln!("[jlens] recursion_trace_lens with J-lens returned None");
                return;
            };
            // detailed per-position table only when NOT sweeping (a single λ), so a sweep stays compact.
            if lambdas.len() == 1 {
                println!("  pos  token→next          final       resolve(logit→J)   flips(logit→J)   stab(logit→J)");
                for (rl, rj) in trace_l.iter().zip(trace_j.iter()) {
                    let tok = if rl.pos + 1 < ids.len() {
                        format!("{}→{}", dec(ids[rl.pos]).trim(), dec(ids[rl.pos + 1]).trim())
                    } else {
                        dec(ids[rl.pos])
                    };
                    let mark = match rj.resolve_layer.cmp(&rl.resolve_layer) {
                        std::cmp::Ordering::Less => "↑earlier",
                        std::cmp::Ordering::Greater => "↓later",
                        std::cmp::Ordering::Equal => "=",
                    };
                    println!(
                        "  {:>3}  {:<18}  {:<10}  {:>2}/{:<2} → {:>2}/{:<2} {:<8}  {:>2} → {:<2}       {:.2} → {:.2}",
                        rl.pos, trunc(&tok, 18), trunc(dec(rl.final_top1).trim(), 10),
                        rl.resolve_layer, nl, rj.resolve_layer, nl, mark, flips(rl), flips(rj), stab(rl), stab(rj)
                    );
                }
                println!();
            }
            let a = aggregate(&trace_l, &trace_j);
            println!(
                "  λ={lam:<4}  resolve {:.2}→{:.2} ({:+.2})   flips {:.2}→{:.2} ({:+.2})   stab {:.2}→{:.2} ({:+.2})   moved {}↑/{}↓/{}=",
                a.rl, a.rj, a.rj - a.rl, a.fl, a.fj, a.fj - a.fl, a.sl, a.sj, a.sj - a.sl, a.earlier, a.later, a.same
            );
        }
        println!("  (lower resolve = reads the answer sooner · lower flips = less mid-stack noise · higher stab = more committed)");
        println!("  (J-lens is an EMPIRICAL readout aid — first-order + context-averaged; it does not touch the forward path.)");
    }

    fn trunc(s: &str, n: usize) -> String {
        if s.chars().count() <= n {
            s.to_string()
        } else {
            format!("{}…", s.chars().take(n - 1).collect::<String>())
        }
    }

    fn argmax(v: &[f32]) -> i64 {
        v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as i64).unwrap_or(0)
    }

    struct CPrompt {
        ent: String,
        ids: Vec<i64>,
        pred: i64,
    }

    /// Build aligned interchange pairs from a template + entity pool: prompts of equal token length whose base
    /// predictions differ (so a swap has something to flip). Returns (prompts, (base_idx, source_idx) pairs, template).
    fn causal_pairs<'a>(args: &'a [String], model: &dyn Model, tg: &TextGen) -> Option<(Vec<CPrompt>, Vec<(usize, usize)>, &'a str)> {
        let template = flag(args, "--causal-template").unwrap_or("The capital of {} is");
        let default_pool = "France,China,Japan,Spain,Italy,Germany,Russia,Egypt,India,Brazil,Canada,Mexico,Greece,Turkey,Norway,Poland,Chile,Peru,Cuba,Iran,Kenya,Portugal,Austria,Ireland";
        let pool: Vec<&str> = flag(args, "--causal-entities").unwrap_or(default_pool).split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        let max_pairs: usize = flag(args, "--causal-pairs").and_then(|s| s.parse().ok()).unwrap_or(24);
        let mut ps: Vec<CPrompt> = Vec::new();
        for ent in &pool {
            let text = template.replacen("{}", ent, 1);
            let ids = tg.encode(&text, false);
            if ids.len() < 3 {
                continue;
            }
            match model.logits(&ids) {
                Some(lg) => ps.push(CPrompt { ent: ent.to_string(), ids, pred: argmax(&lg) }),
                None => {
                    eprintln!("[jlens] causal: arch has no logits hook");
                    return None;
                }
            }
        }
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        'outer: for i in 0..ps.len() {
            for j in 0..ps.len() {
                if i != j && ps[i].ids.len() == ps[j].ids.len() && ps[i].pred != ps[j].pred {
                    pairs.push((i, j));
                    if pairs.len() >= max_pairs {
                        break 'outer;
                    }
                }
            }
        }
        if pairs.is_empty() {
            eprintln!("[jlens] causal: no aligned pairs (same length, different predictions) from {} prompts", ps.len());
            return None;
        }
        Some((ps, pairs, template))
    }

    /// `--recursion-explain --jlens-causal`: CAUSAL interchange tracing — the metric-independent test of the paper's
    /// swap claim (5.3/5.4). For aligned prompt pairs (same length, differing only in a swapped concept), e.g.
    /// "The capital of France is"→Paris vs "…China is"→Beijing, patch the base run's residual at (layer, last-pos) with
    /// the SOURCE run's, and measure whether the output flips to the source's answer (France→Beijing). A hard behavioral
    /// readout — top-1 flip rate + the source-token logit shift — swept over layers to localize where the swappable
    /// concept lives. No lens, no resolve-layer. Uses `residuals_at` + `logits_patched` (rope family, for now).
    pub fn run_causal(args: &[String], model: &dyn Model, tg: &Option<TextGen>, _stem: &str) {
        let Some(tg) = tg.as_ref() else {
            eprintln!("[jlens] --jlens-causal needs a tokenizer");
            return;
        };
        let Some((ps, pairs, template)) = causal_pairs(args, model, tg) else { return };
        let nl = model.dims().map(|(n, _)| n).unwrap_or(0);
        if nl == 0 || model.residuals_at(&ps[0].ids, &[0]).is_none() {
            eprintln!("[jlens] --jlens-causal needs residuals_at + dims (rope family, for now)");
            return;
        }
        let dec = |t: i64| tg.token_label(t);
        println!("[jlens] causal interchange · template {template:?} · {} aligned pairs · {nl} layers · patch last-position residual", pairs.len());
        println!("  swap each base→source: does patching the source's layer-ℓ residual flip base→source's answer?");
        // per-layer accumulators: flip count, summed source-target logit shift, normalized shift toward source.
        let mut flip = vec![0u32; nl];
        let mut shift = vec![0f32; nl];
        let npairs = pairs.len();
        for &(bi, si) in &pairs {
            let (base, src) = (&ps[bi], &ps[si]);
            let last = base.ids.len() - 1;
            let base_lg = model.logits(&base.ids).unwrap();
            let base_tgt = base_lg[src.pred as usize]; // base logit of the SOURCE's answer (what we try to raise)
            // source residuals at the last position, all layers
            let src_res = model.residuals_at(&src.ids, &[last]).unwrap(); // [pos][layer][d]
            for l in 0..nl {
                let donor = &src_res[0][l];
                if let Some(lg) = model.logits_patched(&base.ids, l, &[last], std::slice::from_ref(donor)) {
                    if argmax(&lg) == src.pred {
                        flip[l] += 1;
                    }
                    shift[l] += lg[src.pred as usize] - base_tgt;
                }
            }
        }
        // report per-layer flip rate + mean logit shift, and the peak.
        let (mut peak_l, mut peak_r) = (0usize, -1f32);
        println!("  layer  flip→source   mean Δlogit(source)   bar");
        for l in 0..nl {
            let rate = flip[l] as f32 / npairs as f32;
            if rate > peak_r {
                peak_r = rate;
                peak_l = l;
            }
            let bar = "█".repeat((rate * 30.0).round() as usize);
            println!("  L{l:<3}   {:>4.0}%  ({:>2}/{npairs})   {:>+8.2}          {bar}", rate * 100.0, flip[l], shift[l] / npairs as f32);
        }
        // sample pairs for legibility
        println!("  ── sample swaps (base→source ⇒ answers) ──");
        for &(bi, si) in pairs.iter().take(6) {
            println!("    {:<10} ({}) → {:<10} ({})", ps[bi].ent, dec(ps[bi].pred).trim(), ps[si].ent, dec(ps[si].pred).trim());
        }
        let band: Vec<usize> = (0..nl).filter(|&l| flip[l] as f32 / npairs as f32 >= 0.5).collect();
        println!("  → peak flip-rate {:.0}% at L{peak_l}; ≥50%-flip band: {band:?}", peak_r * 100.0);
        println!("  (FAITHFUL causal test — the patch is a real forward with a swapped residual; behavioral readout, no lens/resolve-layer.)");
    }

    /// `--recursion-explain --jlens-causal-jspace`: the J-SPACE causal test (paper 5.1). At the causal layer (where a
    /// full-residual swap flips the answer, from `--jlens-causal`), patch the base with only a COMPONENT of the swap
    /// `Δ = x_source − x_base`: the J-space projection `P_J Δ` (P_J = top-k logit-relevant subspace, eigenvectors of
    /// `J_lᵀ M J_l`), its complement `(I−P_J)Δ`, or a random rank-k subspace (control). If the J-space part alone flips
    /// the answer (≈ full) while the complement doesn't — and J-space ≫ random — the swappable concept lives in the
    /// J-space. Behavioral readout: the noisy `{J_l}` is used only as a coarse SUBSPACE, so lens fragility doesn't bite.
    /// Knobs: `--jlens-in`, `--causal-layer L` (default late), `--causal-jspace k1,k2,…` (rank sweep). rope family only.
    pub fn run_causal_jspace(args: &[String], model: &dyn Model, tg: &Option<TextGen>, stem: &str) {
        let Some(tg) = tg.as_ref() else {
            eprintln!("[jlens] --jlens-causal-jspace needs a tokenizer");
            return;
        };
        let Some((ps, pairs, _t)) = causal_pairs(args, model, tg) else { return };
        let nl = model.dims().map(|(n, _)| n).unwrap_or(0);
        if nl == 0 || model.residuals_at(&ps[0].ids, &[0]).is_none() {
            eprintln!("[jlens] --jlens-causal-jspace needs residuals_at + dims (rope family, for now)");
            return;
        }
        let inp = flag(args, "--jlens-in").map(|s| s.to_string()).unwrap_or_else(|| format!("{stem}.jlens"));
        let (d, jmats) = match load(&inp) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("[jlens] load {inp}: {e} — run --jlens-fit first (or pass --jlens-in)");
                return;
            }
        };
        let Some(m) = sample_unembed_gram(model, &ps[0].ids, 4096) else {
            eprintln!("[jlens] --jlens-causal-jspace: arch lacks unembed_row for the Gram");
            return;
        };
        let layer = flag(args, "--causal-layer").and_then(|s| s.parse::<usize>().ok()).unwrap_or(nl.saturating_sub(2)).min(nl - 1);
        let ks: Vec<usize> = flag(args, "--causal-jspace")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect::<Vec<_>>())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![8, 32, 128]);

        // cache each prompt's residual at (layer, its last position) — reused across the k sweep
        let resid: Vec<Array1<f32>> = ps
            .iter()
            .map(|p| {
                let last = p.ids.len() - 1;
                Array1::from(model.residuals_at(&p.ids, &[last]).unwrap()[0][layer].clone())
            })
            .collect();
        let npairs = pairs.len() as f32;
        let flip_of = |bi: usize, si: usize, donor: &Array1<f32>| -> bool {
            let last = ps[bi].ids.len() - 1;
            let dv = donor.to_vec();
            model.logits_patched(&ps[bi].ids, layer, &[last], std::slice::from_ref(&dv)).map(|lg| argmax(&lg) == ps[si].pred).unwrap_or(false)
        };
        // full-residual swap = the B1 baseline at this layer
        let full: u32 = pairs.iter().map(|&(bi, si)| flip_of(bi, si, &resid[si]) as u32).sum();
        println!("[jlens] causal J-space test · L{layer} · {} pairs · d={d} · full-swap flip {:.0}%", pairs.len(), full as f32 / npairs * 100.0);
        // candidate subspaces to house the swappable concept: the two J-space definitions, a diff-subspace ORACLE (top
        // PCA of the actual swap directions Δ — where the concept lives BY CONSTRUCTION), and random. `cap` = the mean
        // fraction of ‖Δ‖ the subspace captures; `flip` = patching base with only that subspace's slice of Δ.
        let jl = &jmats[layer];
        let a = jl.t().dot(&m.dot(jl)); // Jᵀ M J — logit-weighted (paper's Wᵤ·J) J-space
        let mut lmi = jl.clone(); // J − I — plain J-space (downstream reshaping)
        for i in 0..d {
            lmi[[i, i]] -= 1.0;
        }
        let deltas: Vec<Array1<f32>> = pairs.iter().map(|&(bi, si)| &resid[si] - &resid[bi]).collect();
        let mut dmat = Array2::<f32>::zeros((d, pairs.len())); // Δ matrix (d × npairs) for the oracle PCA
        for (c, dl) in deltas.iter().enumerate() {
            dmat.column_mut(c).assign(dl);
        }
        let dcov = dmat.dot(&dmat.t()); // d×d Δ-covariance; its top-k eigenvectors are the principal swap directions
        let cap = |phi: &Array2<f32>| -> f32 {
            deltas.iter().map(|dl| {
                let p = phi.dot(&phi.t().dot(dl));
                let (pn, dn) = (p.dot(&p).sqrt(), dl.dot(dl).sqrt());
                if dn > 1e-9 { pn / dn } else { 0.0 }
            }).sum::<f32>() / npairs
        };
        let flip_sub = |phi: &Array2<f32>| -> u32 {
            pairs.iter().enumerate().map(|(pi, &(bi, si))| {
                let pj = phi.dot(&phi.t().dot(&deltas[pi]));
                flip_of(bi, si, &(&resid[bi] + &pj)) as u32
            }).sum()
        };
        println!("  swap ONLY a subspace's slice of Δ=x_src−x_base.  flip = does it redirect base→source; cap = ‖P Δ‖/‖Δ‖.");
        println!("  k     diff-ORACLE      J(WᵤᵀWᵤ)        J(plain J−I)    complement   random");
        println!("         flip  cap       flip  cap       flip  cap       flip         flip");
        for &k in &ks {
            let mut rng = Rng::new(1);
            let phi_w = topk_subspace(&a, k, &mut rng, 2); // logit-weighted J-space
            let phi_p = topk_subspace(&lmi, k, &mut rng, 2); // plain J-space
            let phi_o = topk_subspace(&dcov, k, &mut rng, 2); // ORACLE: top PCA of the swaps themselves
            let mut rr = Array2::<f32>::zeros((d, k.min(d)));
            for x in rr.iter_mut() {
                *x = rng.gauss();
            }
            let rrand = mgs(&rr);
            let (fw, fp, fo, frd) = (flip_sub(&phi_w), flip_sub(&phi_p), flip_sub(&phi_o), flip_sub(&rrand));
            // complement of the logit-weighted J-space (swap everything EXCEPT P_w)
            let cpl: u32 = pairs.iter().enumerate().map(|(pi, &(bi, si))| {
                let pj = phi_w.dot(&phi_w.t().dot(&deltas[pi]));
                flip_of(bi, si, &(&resid[si] - &pj)) as u32
            }).sum();
            let pct = |x: u32| x as f32 / npairs * 100.0;
            println!(
                "  {k:<4}  {:>4.0}% {:>5.2}     {:>4.0}% {:>5.2}     {:>4.0}% {:>5.2}     {:>4.0}%        {:>4.0}%",
                pct(fo), cap(&phi_o), pct(fw), cap(&phi_w), pct(fp), cap(&phi_p), pct(cpl), pct(frd)
            );
        }
        println!("  → diff-ORACLE flip high at small k ⇒ the swap IS low-dimensional. A J-space def is validated iff it");
        println!("    matches the oracle (high flip + high cap). Low cap ⇒ that basis misses the concept.  (behavioral; no lens.)");
    }

    /// `--recursion-explain --jlens-trajectory`: the FAITHFUL residual-trajectory explain for the predicting position.
    /// Per residual-stream write (embed, then each layer's attn / mlp), it reports: the write magnitude `‖d̃_b‖`, that
    /// block's EXACT direct logit contribution to the predicted token `Δ→pred`, the CUMULATIVE logit-lens read (top-k
    /// vocab tokens the residual points at after this block — tagged empirical), and a MEASURED causal flag (does
    /// zeroing this whole block flip the prediction, via `predict_ablated_blocks`). The `resolve` marker is the first
    /// block whose cumulative read locks to the final token. Everything is exact (`residual_normed_writes`) or measured
    /// (block ablation) — no named "concepts", no J-space amplification claims. Knobs: `--traj-topk`, `--traj-causal 0`,
    /// `--traj-json`. rope/neox families.
    pub fn run_trajectory(args: &[String], model: &dyn Model, tg: &Option<TextGen>, ids: &[i64]) {
        let Some(tg) = tg.as_ref() else {
            eprintln!("[jlens] --jlens-trajectory needs a tokenizer");
            return;
        };
        let topk: usize = flag(args, "--traj-topk").and_then(|s| s.parse().ok()).unwrap_or(3);
        let do_causal = flag(args, "--traj-causal").map(|s| s != "0" && s != "off").unwrap_or(true);
        trajectory(model, tg, ids, topk, do_causal, crate::has_flag(args, "--traj-json"));
    }

    /// The trajectory-explain core, shared by the CLI probe (`--jlens-trajectory`) and the chat REPL's `/trajectory`.
    /// Prints the per-write residual trajectory for the predicting position of `ids`. `do_causal` runs the 2·n_layer
    /// block-ablation forwards (the slow part — ~seconds; the lens-only pass is interactive).
    pub fn trajectory(model: &dyn Model, tg: &TextGen, ids: &[i64], topk: usize, do_causal: bool, json: bool) {
        if ids.len() < 2 {
            eprintln!("[jlens] trajectory needs a context of ≥2 tokens");
            return;
        }
        let (labels, dvec) = match model.residual_normed_writes(ids) {
            Some(x) => x,
            None => {
                eprintln!("[jlens] --jlens-trajectory: arch lacks residual_normed_writes (rope/neox for now)");
                return;
            }
        };
        let logits = match model.logits(ids) {
            Some(l) => l,
            None => {
                eprintln!("[jlens] --jlens-trajectory: no logits hook");
                return;
            }
        };
        let pred = argmax(&logits);
        let (mut ru, mut rul) = (0i64, f32::MIN);
        for (i, &v) in logits.iter().enumerate() {
            if i as i64 != pred && v > rul {
                rul = v;
                ru = i as i64;
            }
        }
        let u_pred = match model.unembed_row(pred as usize) {
            Some(u) => u,
            None => return,
        };
        // token_label already renders with quotes + id; flatten control chars so a code token can't break the table row.
        let dec = |t: i64| tg.token_label(t).replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
        let d = dvec[0].len();
        // parse "L18.attn" → (layer, is_attn); "embed" → None
        let parse = |lab: &str| -> Option<(usize, bool)> {
            let (num, kind) = lab.strip_prefix('L')?.split_once('.')?;
            Some((num.parse().ok()?, kind == "attn"))
        };
        println!(
            "[jlens] trajectory · predicts {} (logit {:.2}, margin {:+.2} vs {}) · {} writes · cum-lens=logit (empirical)",
            dec(pred), logits[pred as usize], logits[pred as usize] - rul, dec(ru), labels.len()
        );
        println!("  block         write‖    Δ→pred   cum-lens top{topk}                     {}", if do_causal { "ablate→flip?" } else { "(causal off — /trajectory causal)" });
        let mut cum = vec![0f32; d];
        let mut resolve: Option<usize> = None;
        // (label, write_l2, dla, top-k ids, flip)
        let mut rec: Vec<(String, f32, f32, Vec<i64>, Option<bool>)> = Vec::new();
        for (b, (lab, w)) in labels.iter().zip(&dvec).enumerate() {
            for (c, &wv) in w.iter().enumerate() {
                cum[c] += wv;
            }
            let wl2 = w.iter().map(|v| v * v).sum::<f32>().sqrt();
            let dla = w.iter().zip(&u_pred).map(|(a, b)| a * b).sum::<f32>();
            let lens = model.unembed_project(&cum).unwrap();
            let top = crate::explain::top_promoted(&lens, 1.0, topk);
            if top.first() == Some(&pred) && resolve.is_none() {
                resolve = Some(b);
            }
            let (caus, flip) = if do_causal {
                match parse(lab) {
                    Some((l, is_attn)) => {
                        let (al, ml): (Vec<usize>, Vec<usize>) = if is_attn { (vec![l], vec![]) } else { (vec![], vec![l]) };
                        match model.predict_ablated_blocks(ids, &[], &[], &al, &ml) {
                            Some(t) => {
                                let f = t != pred;
                                (if f { format!("FLIP→{}", dec(t)) } else { "·".into() }, Some(f))
                            }
                            None => ("—".into(), None),
                        }
                    }
                    None => ("—".into(), None), // embed: no block ablation
                }
            } else {
                (String::new(), None)
            };
            let toks = top.iter().map(|&t| dec(t)).collect::<Vec<_>>().join("  ");
            let rmark = if Some(b) == resolve { " ◄resolve" } else { "" };
            println!("  {:<12}  {:>6.2}   {:>+7.2}   {:<30}{:<9} {}", lab, wl2, dla, toks, rmark, caus);
            rec.push((lab.clone(), wl2, dla, top, flip));
        }
        // ---- summary ----
        let rb = resolve.map(|b| format!("{} (write {}/{})", rec[b].0, b + 1, rec.len())).unwrap_or_else(|| "never (last write only)".into());
        println!("  → resolves to {} at {rb}", dec(pred));
        let mut order: Vec<usize> = (0..rec.len()).collect();
        order.sort_by(|&a, &b| rec[b].2.total_cmp(&rec[a].2));
        let writers: Vec<String> = order.iter().take(4).map(|&i| format!("{} {:+.2}", rec[i].0, rec[i].2)).collect();
        println!("  → top exact writers to {}: {}", dec(pred), writers.join(", "));
        if do_causal {
            let flips: Vec<String> = rec.iter().filter(|r| r.4 == Some(true)).map(|r| r.0.clone()).collect();
            println!("  → single-block ablations that FLIP the prediction: {}", if flips.is_empty() { "(none singly — the write is distributed)".into() } else { flips.join(", ") });
        }
        println!("  (‖·‖ + Δ→pred are EXACT block contributions; cum-lens is the logit-lens readout — empirical, VOCAB tokens not named concepts; ablate→flip is MEASURED.)");
        if json {
            let blocks: Vec<serde_json::Value> = rec.iter().map(|(lab, wl2, dla, top, flip)| {
                serde_json::json!({ "block": lab, "write_l2": wl2, "dla_pred": dla,
                    "cum_lens_top": top.iter().map(|&t| dec(t)).collect::<Vec<_>>(), "ablate_flips": flip })
            }).collect();
            let out = serde_json::json!({
                "predicted": dec(pred), "predicted_id": pred, "runner_up": dec(ru), "margin": logits[pred as usize] - rul,
                "resolve_block": resolve.map(|b| rec[b].0.clone()),
                "lens": {"type": "logit-lens", "tag": "empirical", "note": "cumulative-write read; VOCAB tokens, not named/J-space concepts"},
                "blocks": blocks });
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        }
    }

    /// Dispatch: returns true if it handled a `--jlens-*` subcommand.
    pub fn dispatch(args: &[String], model: &dyn Model, tg: &Option<TextGen>, stem: &str, ids: &[i64]) -> bool {
        if has_flag(args, "--jlens-fit") {
            run_fit(args, model, tg, stem, ids);
            true
        } else if has_flag(args, "--jlens-eval") {
            run_eval(args, model, tg, stem, ids);
            true
        } else if has_flag(args, "--jlens-trajectory") {
            run_trajectory(args, model, tg, ids);
            true
        } else if has_flag(args, "--jlens-causal-jspace") {
            run_causal_jspace(args, model, tg, stem);
            true
        } else if has_flag(args, "--jlens-causal") {
            run_causal(args, model, tg, stem);
            true
        } else if let Some(out) = flag(args, "--tensors-export") {
            run_tensors_export(model, out);
            true
        } else {
            false
        }
    }
}

#[cfg(feature = "api")]
pub use cli::{dispatch, trajectory};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn gauss_is_roughly_standard_normal() {
        let mut r = Rng::new(42);
        let n = 20_000;
        let xs: Vec<f32> = (0..n).map(|_| r.gauss()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean} not ≈ 0");
        assert!((var - 1.0).abs() < 0.1, "var {var} not ≈ 1");
    }

    // A synthetic "downstream map" that is exactly linear WITH position mixing: forward_from(x) = M · x · A, where M is
    // lower-triangular (causal) seq×seq and A is d×d. Then ∂h[t']/∂x[t] = M[t',t]·Aᵀ, and averaging over t'≥t gives the
    // estimator's target J = c_t·Aᵀ with c_t = mean_{t'≥t} M[t',t]. Central-differencing a linear map is exact, so the
    // only error is the Hutchinson MC variance — this pins BOTH the outer-product recovery and the t'≥t averaging.
    #[test]
    fn estimate_recovers_known_linear_map() {
        let (seq, d) = (3usize, 4usize);
        // deterministic A and lower-triangular M
        let a = Array2::from_shape_fn((d, d), |(i, j)| 0.1 * (i as f32) - 0.2 * (j as f32) + if i == j { 1.0 } else { 0.3 });
        let m = Array2::from_shape_fn((seq, seq), |(i, j)| if j <= i { 0.5 + 0.1 * (i as f32) + 0.05 * (j as f32) } else { 0.0 });
        let a_cl = a.clone();
        let m_cl = m.clone();
        let forward_from = move |x: &Array2<f32>| m_cl.dot(x).dot(&a_cl);

        let x0 = Array2::from_shape_fn((seq, d), |(i, j)| 1.0 + 0.3 * (i as f32) - 0.1 * (j as f32));
        let t = 1usize; // single source position ⇒ a clean closed form
        let mut rng = Rng::new(7);
        let jest = estimate_jacobian(&x0, &[t], &forward_from as &dyn Fn(&Array2<f32>) -> Array2<f32>, 20_000, 1e-3, &mut rng);

        // expected: c_t · Aᵀ  with c_t = mean_{t'≥t} M[t',t]
        let c_t: f32 = (t..seq).map(|tp| m[[tp, t]]).sum::<f32>() / (seq - t) as f32;
        let mut maxerr = 0f32;
        for i in 0..d {
            for j in 0..d {
                let want = c_t * a[[j, i]]; // (Aᵀ)[i,j] = A[j,i]
                maxerr = maxerr.max((jest[[i, j]] - want).abs());
            }
        }
        assert!(maxerr < 0.08, "max |Ĵ − c·Aᵀ| = {maxerr} too large (MC/estimator error)");
    }

    #[test]
    fn save_load_round_trips() {
        let d = 3usize;
        let mats = vec![
            Array2::from_shape_fn((d, d), |(i, j)| (i * 10 + j) as f32 * 0.5),
            Array2::<f32>::eye(d),
        ];
        let path = std::env::temp_dir().join("fieldrun_jlens_roundtrip.bin");
        let ps = path.to_str().unwrap();
        save(ps, &mats, d).unwrap();
        let (dg, got) = load(ps).unwrap();
        assert_eq!(dg, d);
        assert_eq!(got.len(), mats.len());
        for (a, b) in mats.iter().zip(got.iter()) {
            assert_eq!(a, b);
        }
        let _ = std::fs::remove_file(ps);
    }


    #[test]
    fn sym_eig_jacobi_recovers_known_spectrum() {
        // A = Q diag(3,1) Qᵀ for a 45° Q ⇒ eigenvalues {3,1}, eigenvectors the rotated axes.
        let c = std::f32::consts::FRAC_1_SQRT_2;
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap(); // eig {3,1}
        let (ev, vec) = sym_eig_jacobi(&a);
        assert!((ev[0] - 3.0).abs() < 1e-4 && (ev[1] - 1.0).abs() < 1e-4);
        // leading eigenvector ~ (1,1)/√2 (up to sign)
        assert!((vec[[0, 0]].abs() - c).abs() < 1e-3 && (vec[[1, 0]].abs() - c).abs() < 1e-3);
    }

    #[test]
    fn truncate_recovers_a_low_rank_matrix() {
        // A = rank-3 signal + small full-rank noise, d=24. Truncating to rank 3 should recover the signal well and beat
        // keeping the noisy full matrix.
        let (d, k) = (24usize, 3usize);
        let mut rng = Rng::new(11);
        let b = Array2::from_shape_fn((d, k), |_| rng.gauss());
        let c = Array2::from_shape_fn((d, k), |_| rng.gauss());
        let signal = b.dot(&c.t()); // exact rank 3
        let noise = Array2::from_shape_fn((d, d), |_| 0.02 * rng.gauss());
        let a = &signal + &noise;
        let ak = truncate_rank(&a, k, &mut rng, 2);
        let err_k: f32 = (&ak - &signal).iter().map(|v| v * v).sum();
        let err_full: f32 = (&a - &signal).iter().map(|v| v * v).sum();
        assert!(err_k < err_full, "rank-k truncation ({err_k}) must be closer to the signal than the noisy full matrix ({err_full})");
        assert!(err_k < 0.6 * err_full, "truncation should remove a meaningful share of the noise energy");
    }

    #[test]
    fn logit_weighted_keeps_relevant_over_big_irrelevant() {
        // L has (a) a BIG map e3→e4 into a logit-IRRELEVANT output coord (M weight 0.001), and (b) a SMALL map e0→e0
        // into a RELEVANT coord (M weight 1). Plain rank-1 keeps the big one (σ=10); logit-weighted rank-1 must keep the
        // relevant one — because A = Lᵀ M L ranks e0 (0.5²·1) above e3 (10²·0.001).
        let d = 6usize;
        let m = Array2::from_shape_fn((d, d), |(i, j)| if i == j { if i < 3 { 1.0 } else { 0.001 } } else { 0.0 });
        let mut l = Array2::<f32>::zeros((d, d));
        l[[4, 3]] = 10.0; // big, irrelevant
        l[[0, 0]] = 0.5; // small, relevant
        let mut j = l.clone();
        for i in 0..d {
            j[[i, i]] += 1.0; // J = I + L
        }
        let jl = rank_reduce_logit(&j, 1, &m, 1);
        assert!(jl[[0, 0]] > 1.3, "logit-weighted must KEEP the relevant e0→e0 (J[0,0]≈1.5), got {}", jl[[0, 0]]);
        assert!(jl[[4, 3]].abs() < 0.5, "logit-weighted must DROP the big irrelevant e3→e4, got {}", jl[[4, 3]]);
        let jp = rank_reduce(&j, 1, 1); // plain SVD keeps the biggest-magnitude direction
        assert!(jp[[4, 3]] > 5.0, "plain rank keeps the big e3→e4, got {}", jp[[4, 3]]);
        assert!(jp[[0, 0]] < 1.3, "plain rank drops the small e0→e0, got {}", jp[[0, 0]]);
    }

    #[test]
    fn truncate_full_rank_is_a_noop() {
        let mut rng = Rng::new(3);
        let a = Array2::from_shape_fn((6, 6), |_| rng.gauss());
        assert_eq!(truncate_rank(&a, 0, &mut rng, 2), a); // k=0
        assert_eq!(truncate_rank(&a, 6, &mut rng, 2), a); // k>=d
    }

    #[test]
    fn rank_reduce_preserves_identity_and_reduces_l() {
        let d = 20usize;
        // J = I exactly ⇒ L = 0 ⇒ rank-reduce is still I.
        let jid = Array2::<f32>::eye(d);
        let r = rank_reduce(&jid, 4, 1);
        assert!((&r - &jid).iter().map(|v| v.abs()).fold(0.0, f32::max) < 1e-5);
        // J = I + rank-2 ⇒ rank_reduce(k=2) should recover it (L is exactly rank 2).
        let mut rng = Rng::new(9);
        let u = Array2::from_shape_fn((d, 2), |_| rng.gauss());
        let v = Array2::from_shape_fn((d, 2), |_| rng.gauss());
        let mut j = u.dot(&v.t());
        for i in 0..d {
            j[[i, i]] += 1.0;
        }
        let jr = rank_reduce(&j, 2, 1);
        assert!((&jr - &j).iter().map(|v| v * v).sum::<f32>() < 1e-3, "rank-2 J recovered by rank_reduce(k=2)");
    }

    #[test]
    fn crc32_matches_standard_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926); // the canonical CRC-32/IEEE check value
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn npy_header_is_64b_aligned_and_wellformed() {
        let data = vec![0u8; 4 * 6]; // 6 f32
        let npy = npy_bytes("<f4", &[2, 3], &data);
        assert_eq!(&npy[0..8], b"\x93NUMPY\x01\x00");
        let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0, "magic+version+len+header must be 64-byte aligned");
        let header = std::str::from_utf8(&npy[10..10 + hlen]).unwrap();
        assert!(header.ends_with('\n'));
        assert!(header.contains("'descr': '<f4'"));
        assert!(header.contains("'shape': (2, 3)"));
        assert_eq!(npy.len(), 10 + hlen + data.len());
    }

    #[test]
    fn npz_has_zip_signatures_and_members() {
        let path = std::env::temp_dir().join("fieldrun_jlens_npz.npz");
        let ps = path.to_str().unwrap();
        let d = 2;
        let mats = vec![Array2::<f32>::eye(d), Array2::from_shape_fn((d, d), |(i, j)| (i + j) as f32)];
        let fitted = export_npz(ps, &mats, d).unwrap();
        assert_eq!(fitted, vec![1]); // layer 0 is identity, layer 1 is fit
        let bytes = std::fs::read(ps).unwrap();
        assert_eq!(&bytes[0..4], &0x0403_4b50u32.to_le_bytes()); // first local file header
        // end-of-central-directory signature present near the tail
        assert!(bytes.windows(4).any(|w| w == 0x0605_4b50u32.to_le_bytes()));
        // both member names appear
        assert!(bytes.windows(5).any(|w| w == b"J.npy"));
        assert!(bytes.windows(10).any(|w| w == b"fitted.npy"));
        let _ = std::fs::remove_file(ps);
    }

    #[test]
    fn shrink_interpolates_between_identity_and_j() {
        let j = Array2::from_shape_fn((3, 3), |(i, k)| (i * 3 + k) as f32 + 0.5);
        // λ=1 ⇒ unchanged; λ=0 ⇒ identity; λ=0.5 ⇒ halfway with (1-λ) added on the diagonal.
        assert_eq!(shrink_toward_identity(&j, 1.0), j);
        assert_eq!(shrink_toward_identity(&j, 0.0), Array2::<f32>::eye(3));
        let half = shrink_toward_identity(&j, 0.5);
        assert!((half[[0, 1]] - 0.5 * j[[0, 1]]).abs() < 1e-6); // off-diagonal just scales by λ
        assert!((half[[0, 0]] - (0.5 * j[[0, 0]] + 0.5)).abs() < 1e-6); // diagonal gets +(1-λ)
    }

    #[test]
    fn dist_from_identity_zero_for_eye() {
        assert!(dist_from_identity(&Array2::<f32>::eye(5)) < 1e-6);
        let m = Array2::from_shape_fn((2, 2), |(i, j)| if i == j { 1.0 } else { 2.0 });
        assert!((dist_from_identity(&m) - (8.0f32).sqrt()).abs() < 1e-5); // two off-diagonal 2's ⇒ ‖·‖=√8
    }

    #[test]
    fn sample_src_spreads_and_caps() {
        assert_eq!(sample_src(1, 4), Vec::<usize>::new());
        assert_eq!(sample_src(5, 10), vec![1, 2, 3, 4]); // fewer available than asked ⇒ all of 1..seq
        let s = sample_src(20, 4);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], 1);
        assert!(s.iter().all(|&p| p >= 1 && p < 20));
        assert!(s.windows(2).all(|w| w[0] < w[1]));
    }
}
