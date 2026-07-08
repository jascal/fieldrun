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

use ndarray::Array2;

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
        let nl = trace_l.first().map(|r| r.n_layer).unwrap_or(0);
        let dec = |t: i64| tg.token_label(t);
        println!("[jlens] eval · d={d} · {nl} layers · {} positions · J-lens vs logit-lens · shrink λ∈{lambdas:?}", trace_l.len());

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

    /// Dispatch: returns true if it handled a `--jlens-*` subcommand.
    pub fn dispatch(args: &[String], model: &dyn Model, tg: &Option<TextGen>, stem: &str, ids: &[i64]) -> bool {
        if has_flag(args, "--jlens-fit") {
            run_fit(args, model, tg, stem, ids);
            true
        } else if has_flag(args, "--jlens-eval") {
            run_eval(args, model, tg, stem, ids);
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
pub use cli::dispatch;

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
