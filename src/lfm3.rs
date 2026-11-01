//! LFM3's tri-hybrid text backbone: short causal convolution plus full and sliding GQA attention.

use crate::bundle::Bundle;
use crate::model::Model;
use ndarray::{s, Array1, Array2};

pub struct Lfm3 {
    b: Bundle,
    nl: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
    ty: Vec<usize>,
    theta_full: f32,
    theta_slide: f32,
    partial: usize,
    window: usize,
    tied: bool,
}

fn rms(x: &Array2<f32>, w: Array1<f32>, eps: f32) -> Array2<f32> {
    let mut y = x.clone();
    for mut r in y.rows_mut() {
        let z = 1.0 / ((r.iter().map(|v| v * v).sum::<f32>() / r.len() as f32) + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v *= z * w[i];
        }
    }
    y
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softmax(a: &mut Array2<f32>) {
    for mut r in a.rows_mut() {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.;
        for v in r.iter_mut() {
            *v = (*v - m).exp();
            z += *v;
        }
        for v in r.iter_mut() {
            *v /= z;
        }
    }
}

impl Lfm3 {
    pub fn new(b: Bundle) -> Self {
        let c = &b.config;
        Self {
            nl: c[0] as usize,
            nh: c[1] as usize,
            nkv: c[2] as usize,
            hd: c[3] as usize,
            eps: b.config_f[0] as f32,
            theta_full: b.config_f[1] as f32,
            theta_slide: b.config_f[2] as f32,
            partial: c[8] as usize,
            window: c[9] as usize,
            tied: c[7] != 0,
            ty: c[10..].iter().map(|&x| x as usize).collect(),
            b,
        }
    }
    fn rope(&self, x: &mut Array2<f32>, heads: usize, pos0: usize, theta: f32) {
        let half = self.hd / 2;
        for (i, mut r) in x.rows_mut().into_iter().enumerate() {
            for h in 0..heads {
                for j in 0..half {
                    if j >= self.partial / 2 {
                        continue;
                    }
                    let a = (pos0 + i) as f32 / theta.powf(2.0 * j as f32 / self.hd as f32);
                    let (c, s) = (a.cos(), a.sin());
                    let k = h * self.hd;
                    let (u, v) = (r[k + j], r[k + j + half]);
                    r[k + j] = u * c - v * s;
                    r[k + j + half] = v * c + u * s;
                }
            }
        }
    }
    fn head_norm(&self, x: &mut Array2<f32>, name: &str, heads: usize) {
        let w = self.b.arr1(name);
        for mut r in x.rows_mut() {
            for h in 0..heads {
                let o = h * self.hd;
                let z = 1.
                    / (((0..self.hd).map(|i| r[o + i] * r[o + i]).sum::<f32>() / self.hd as f32)
                        + self.eps)
                        .sqrt();
                for i in 0..self.hd {
                    r[o + i] *= z * w[i];
                }
            }
        }
    }
    fn conv(&self, x: &Array2<f32>, name: &str) -> Array2<f32> {
        let w = self.b.arr2o(name);
        let (n, d) = (x.nrows(), x.ncols());
        let k = w.ncols();
        let mut y = Array2::zeros((n, d));
        for i in 0..n {
            for c in 0..d {
                for j in 0..k {
                    if i >= k - 1 - j {
                        y[[i, c]] += x[[i - (k - 1 - j), c]] * w[[c, j]];
                    }
                }
            }
        }
        y
    }
    fn short_conv(&self, x: &Array2<f32>, p: &str) -> Array2<f32> {
        let d = x.ncols();
        let bc_x = self.b.mm(x, &format!("{p}conv.in_proj"));
        let mut bx = Array2::zeros((x.nrows(), d));
        for r in 0..x.nrows() {
            for c in 0..d {
                bx[[r, c]] = bc_x[[r, c]] * bc_x[[r, 2 * d + c]];
            }
        }
        let conv = self.conv(&bx, &format!("{p}conv.conv"));
        let mut y = conv;
        for r in 0..x.nrows() {
            for c in 0..d {
                y[[r, c]] *= bc_x[[r, d + c]];
            }
        }
        self.b.mm(&y, &format!("{p}conv.out_proj"))
    }
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        let n = ids.len();
        // Unlike Llama-family models, LFM3 applies `embedding_norm` to the decoder output,
        // not the token embeddings. The official checkpoints therefore have no separate final norm.
        let mut x = self.b.rows_f32("embed", ids);
        let rep = self.nh / self.nkv;
        for l in 0..self.nl {
            let p = format!("l{l}.");
            let a = rms(&x, self.b.arr1(&format!("{p}operator_norm")), self.eps);
            let op = if self.ty[l] == 0 {
                self.short_conv(&a, &p)
            } else {
                let mut q = self.b.mm(&a, &format!("{p}self_attn.q_proj"));
                let mut k = self.b.mm(&a, &format!("{p}self_attn.k_proj"));
                let v = self.b.mm(&a, &format!("{p}self_attn.v_proj"));
                self.head_norm(&mut q, &format!("{p}self_attn.q_norm"), self.nh);
                self.head_norm(&mut k, &format!("{p}self_attn.k_norm"), self.nkv);
                let theta = if self.ty[l] == 1 {
                    self.theta_full
                } else {
                    self.theta_slide
                };
                self.rope(&mut q, self.nh, 0, theta);
                self.rope(&mut k, self.nkv, 0, theta);
                let mut out = Array2::zeros((n, self.nh * self.hd));
                for h in 0..self.nh {
                    let kv = h / rep;
                    let qh = q.slice(s![.., h * self.hd..(h + 1) * self.hd]);
                    let kh = k.slice(s![.., kv * self.hd..(kv + 1) * self.hd]);
                    let vh = v.slice(s![.., kv * self.hd..(kv + 1) * self.hd]);
                    let mut sc = qh.dot(&kh.t()) / (self.hd as f32).sqrt();
                    for i in 0..n {
                        for j in 0..n {
                            if j > i || (self.ty[l] == 2 && i - j >= self.window) {
                                sc[[i, j]] = -1e30;
                            }
                        }
                    }
                    softmax(&mut sc);
                    out.slice_mut(s![.., h * self.hd..(h + 1) * self.hd])
                        .assign(&sc.dot(&vh));
                }
                let g = self.b.mm(&a, &format!("{p}self_attn.gate_proj"));
                for (i, v) in out.iter_mut().enumerate() {
                    *v *= sigmoid(g[[i / (self.nh * self.hd), i % (self.nh * self.hd) / self.hd]]);
                }
                self.b.mm(&out, &format!("{p}self_attn.o_proj"))
            };
            let x1 = &x
                + &rms(
                    &op,
                    self.b.arr1(&format!("{p}post_operator_norm")),
                    self.eps,
                );
            let f = rms(&x1, self.b.arr1(&format!("{p}ffn_norm")), self.eps);
            let mut h = self.b.mm(&f, &format!("{p}feed_forward.w1"));
            let w3 = self.b.mm(&f, &format!("{p}feed_forward.w3"));
            for (v, u) in h.iter_mut().zip(w3.iter()) {
                *v = silu(*v) * u;
            }
            let ff = self.b.mm(&h, &format!("{p}feed_forward.w2"));
            x = &x1 + &rms(&ff, self.b.arr1(&format!("{p}post_ffn_norm")), self.eps);
        }
        rms(&x, self.b.arr1("embedding_norm"), self.eps)
    }
}
impl Model for Lfm3 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let x = self.hidden(ids);
        let un = if self.tied { "embed" } else { "lm_head" };
        self.b
            .rowdot_f32(un, &x.row(x.nrows() - 1).to_vec())
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i64
    }
}
