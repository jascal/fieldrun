//! Tier B — composition. The GPT-2 forward pass as Rust matmuls over the flat bundle, a faithful port of pylm's
//! `numpy_lm.py`: token+position embeddings, then per layer LayerNorm -> causal multi-head attention -> LayerNorm ->
//! GELU MLP, a final LayerNorm, and the tied unembed. This is the half flat retrieval cannot do (genuine dense
//! computation, the forge tax) — it runs here as plain `ndarray` matmuls, no framework. fp32 in, exact vs numpy.

use ndarray::{s, Array1, Array2};

use crate::bundle::Bundle;
use crate::model::Model;

fn default_observer_check() -> bool { true }

fn observer_identity(observed: &Array2<f32>, plain: &Array2<f32>) -> Result<f32, String> {
    if observed.shape() != plain.shape() || observed.iter().zip(plain.iter()).any(|(a,b)|
        !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits()) {
        return Err("observer changed hidden state".into());
    }
    Ok(0.0)
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct ResidualMemory {
    pub site: String,
    pub write: Vec<f32>,
}

fn replay_memory(x: &mut Array2<f32>, memory: Option<&ResidualMemory>, site: &str) -> Option<Array2<f32>> {
    let memory = memory.filter(|m| m.site == site)?;
    let last = x.nrows()-1;
    let mut added = Array2::<f32>::zeros(x.raw_dim());
    for (i, &value) in memory.write.iter().enumerate() {
        x[[last,i]] += value;
        added[[last,i]] = value;
    }
    Some(added)
}

#[derive(serde::Deserialize)]
pub struct ReadSelection {
    pub layers: Vec<usize>,
    pub heads: Vec<usize>,
    pub positions: Vec<usize>,
    pub contexts: Vec<Vec<i64>>,
    #[serde(default)]
    pub last_position: bool,
    #[serde(default)]
    pub capture_sources: bool,
    #[serde(default)]
    pub last_prediction_only: bool,
    #[serde(default)]
    pub candidate_tokens: Vec<i64>,
    #[serde(default)]
    pub score_candidates_only: bool,
    #[serde(default = "default_observer_check")]
    pub check_observer: bool,
    #[serde(default)]
    pub memory_writes: Vec<Option<ResidualMemory>>,
}

pub struct Gpt2 {
    b: Bundle,
    n_layer: usize,
    n_head: usize,
    d: usize,
    route: f32,    // Tier C: fraction of MLP neurons to compute per token (0 = off / full)
    kv_int8: bool, // store the KV cache as int8 (per-head scale) — 4x smaller cache for long context
}

fn layernorm(x: &Array2<f32>, g: Array1<f32>, b: Array1<f32>) -> Array2<f32> {
    let eps = 1e-5f32;
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let mu = row.sum() / n;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mu) * inv * g[i] + b[i];
        }
    }
    out
}

fn gelu(x: &mut Array2<f32>) {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    x.mapv_inplace(|v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()));
}

fn softmax_rows(a: &mut Array2<f32>) {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        row.mapv_inplace(|v| v / s);
    }
}

impl Gpt2 {
    pub fn new(b: Bundle, route: f32, kv_int8: bool) -> Gpt2 {
        let c = &b.config; // [n_layer, n_head, n_embd, n_positions, vocab]
        let (n_layer, n_head, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        Gpt2 { b, n_layer, n_head, d, route, kv_int8 }
    }

    fn down(&self, h: &Array2<f32>, name: &str) -> Array2<f32> {
        if self.route > 0.0 && self.route < 1.0 {
            self.b.mm_routed_down(h, name, self.route)
        } else {
            self.b.mm(h, name)
        }
    }

    /// Final hidden states (seq, d) after the last LayerNorm — the forward pass minus the unembed. Mirrors
    /// `numpy_lm.NumpyGPT2.logits` up to `x @ wte.T`. The unembed is split out so `predict` projects only the last row.
    fn hidden(&self, ids: &[i64]) -> Array2<f32> {
        self.hidden_observed(ids, None, &mut Vec::new(), &mut Vec::new())
    }

    fn hidden_observed(&self, ids: &[i64], selection: Option<&ReadSelection>, reads: &mut Vec<serde_json::Value>, sources: &mut Vec<serde_json::Value>) -> Array2<f32> {
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>());
        let mut capture_source = |id: String, write: &Array2<f32>| {
            if let Some(sel) = selection.filter(|s| s.capture_sources) {
                for &i in &sel.positions {
                    sources.push(serde_json::json!({"id": id, "position": i, "write": write.row(i).to_vec()}));
                }
            }
        };
        capture_source("embed".into(), &x);
        let memory = selection.and_then(|s| s.memory_writes.first()).and_then(|m| m.as_ref());
        if let Some(added) = replay_memory(&mut x, memory, "embed") {
            capture_source("memory@embed".into(), &added);
        }

        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            for h in 0..self.n_head {
                let q = qkv.slice(s![.., h * hd..(h + 1) * hd]);
                let k = qkv.slice(s![.., self.d + h * hd..self.d + (h + 1) * hd]);
                let v = qkv.slice(s![.., 2 * self.d + h * hd..2 * self.d + (h + 1) * hd]);
                let mut scores = q.dot(&k.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e10; // causal mask
                    }
                }
                let observing = selection.is_some_and(|s| s.layers.contains(&l) && s.heads.contains(&h));
                let raw_scores = observing.then(|| scores.clone());
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&v));
                if let Some(sel) = selection.filter(|_| observing) {
                    for &i in &sel.positions {
                        reads.push(serde_json::json!({
                            "layer": l, "head": h, "query_position": i,
                            "scale": 1.0f32 / (hd as f32).sqrt(),
                            "query": q.row(i).to_vec(),
                            "keys": k.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                            "values": v.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                            "masked_logits": raw_scores.as_ref().unwrap().row(i).to_vec(),
                            "probabilities": scores.row(i).to_vec(),
                            "head_write": attn_out.slice(s![i, h * hd..(h + 1) * hd]).to_vec(),
                        }));
                    }
                }
            }
            let attention_write = self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight"))
                + &self.b.arr1(&format!("{p}attn.c_proj.bias"));
            capture_source(format!("L{l}.attn"), &attention_write);
            x = &x + &attention_write;
            if let Some(added) = replay_memory(&mut x, memory, &format!("L{l}.attn")) {
                capture_source(format!("memory@L{l}.attn"), &added);
            }

            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut h_mlp = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut h_mlp);
            let mlp_write = self.down(&h_mlp, &format!("{p}mlp.c_proj.weight"))
                + &self.b.arr1(&format!("{p}mlp.c_proj.bias"));
            capture_source(format!("L{l}.mlp"), &mlp_write);
            x = &x + &mlp_write;
            if let Some(added) = replay_memory(&mut x, memory, &format!("L{l}.mlp")) {
                capture_source(format!("memory@L{l}.mlp"), &added);
            }
        }

        if selection.is_some() {
            for row in reads.iter_mut() {
                let i = row["query_position"].as_u64().unwrap() as usize;
                row["final_residual"] = serde_json::json!(x.row(i).to_vec());
            }
        }
        layernorm(&x, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"))
    }

    /// Observe the actual native forward; no alternative attention implementation.
    pub fn read_dump(&self, selection: &ReadSelection) -> Result<serde_json::Value, String> {
        if self.route != 0.0 || self.kv_int8 {
            return Err("read dump requires full gates and f32 KV mode".into());
        }
        if selection.layers.is_empty() || selection.heads.is_empty() || (!selection.last_position && selection.positions.is_empty())
            || selection.contexts.is_empty()
            || selection.layers.iter().any(|&l| l >= self.n_layer)
            || selection.heads.iter().any(|&h| h >= self.n_head) {
            return Err("nonempty, in-range head and position selection required".into());
        }
        if selection.candidate_tokens.iter().any(|&t| t < 0 || t >= self.b.config[4]) {
            return Err("candidate token outside vocabulary".into());
        }
        if selection.score_candidates_only && selection.candidate_tokens.is_empty() {
            return Err("candidate-only mode requires a nonempty candidate set".into());
        }
        if !selection.memory_writes.is_empty() && selection.memory_writes.len() != selection.contexts.len() {
            return Err("memory writes must align one-to-one with contexts".into());
        }
        for memory in selection.memory_writes.iter().flatten() {
            let valid_site = memory.site == "embed" || (0..self.n_layer).any(|l|
                memory.site == format!("L{l}.attn") || memory.site == format!("L{l}.mlp"));
            if !valid_site || memory.write.len() != self.d || memory.write.iter().any(|v| !v.is_finite()) {
                return Err("memory requires a valid source site and one finite residual-width vector".into());
            }
            if selection.check_observer {
                return Err("memory replay must explicitly disable the unmatched observer comparison".into());
            }
        }
        let mut cases = Vec::new();
        for (case_index, ids) in selection.contexts.iter().enumerate() {
            if ids.is_empty() || ids.len() > self.b.config[3] as usize
                || (!selection.last_position && selection.positions.iter().any(|&i| i >= ids.len()))
                || ids.iter().any(|&i| i < 0 || i >= self.b.config[4]) {
                return Err("invalid token context or query position".into());
            }
            let mut reads = Vec::new();
            let selected = ReadSelection {
                layers: selection.layers.clone(), heads: selection.heads.clone(),
                positions: if selection.last_position { vec![ids.len() - 1] } else { selection.positions.clone() },
                contexts: Vec::new(), last_position: false, capture_sources: selection.capture_sources,
                last_prediction_only: selection.last_prediction_only,
                candidate_tokens: selection.candidate_tokens.clone(),
                score_candidates_only: selection.score_candidates_only,
                check_observer: selection.check_observer,
                memory_writes: vec![selection.memory_writes.get(case_index).cloned().flatten()],
            };
            let mut sources = Vec::new();
            let started = std::time::Instant::now();
            let observed = self.hidden_observed(ids, Some(&selected), &mut reads, &mut sources);
            let prediction_positions: Vec<usize> = if selection.last_prediction_only {
                vec![ids.len() - 1]
            } else { (0..ids.len()).collect() };
            let mut native_output_row_calls = 0usize;
            let mut predictions = |x: &Array2<f32>| {
                let mut predicted = Vec::new();
                let mut candidate_scores = Vec::new();
                for &i in &prediction_positions {
                    if selection.score_candidates_only {
                        let logits = self.b.rowdot_f32_subset("wte", &x.row(i).to_vec(), &selection.candidate_tokens);
                        native_output_row_calls += selection.candidate_tokens.len();
                        let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                        predicted.push(selection.candidate_tokens[best] as usize);
                        candidate_scores.push(logits);
                    } else {
                        let logits = self.b.rowdot_f32("wte", &x.row(i).to_vec());
                        native_output_row_calls += logits.len();
                        predicted.push(logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0);
                        candidate_scores.push(selection.candidate_tokens.iter().map(|&t| logits[t as usize]).collect::<Vec<_>>());
                    }
                }
                (predicted, candidate_scores)
            };
            let (observed_predictions, candidate_scores) = predictions(&observed);
            let elapsed_us = started.elapsed().as_micros();
            // Validation is outside the measured forward/decode and never unembeds twice.
            let plain = selection.check_observer.then(|| self.hidden(ids));
            let observer_error = plain.as_ref().map(|plain| observer_identity(&observed, plain)).transpose()?;
            // Equal hidden bits imply the same deterministic scorer output.
            let plain_predictions = plain.as_ref().map(|_| observed_predictions.clone());
            cases.push(serde_json::json!({
                "ids": ids, "reads": reads,
                "sources": sources, "prediction_positions": prediction_positions,
                "predictions": observed_predictions, "unobserved_predictions": plain_predictions,
                "candidate_tokens": selection.candidate_tokens, "candidate_scores": candidate_scores,
                "scoring_scope": if selection.score_candidates_only { "candidate_subset" } else { "full_vocabulary" },
                "native_output_row_calls": native_output_row_calls,
                "elapsed_forward_decode_us": elapsed_us,
                "measurement_scope": "observed_forward_and_decode_only",
                "observer_check_performed": selection.check_observer,
                "memory_replay": selected.memory_writes.first().cloned().flatten(),
                "final_hidden": observed.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "observer_hidden_max_error": observer_error,
            }));
        }
        Ok(serde_json::json!({"schema": "pdc.read-incidence.v1", "arch": "gpt2", "cases": cases}))
    }

    fn mlp_act_last(&self, ids: &[i64], layer: usize) -> Option<Vec<f32>> {
        if layer >= self.n_layer || ids.is_empty() {
            return None;
        }
        // Same arithmetic as `hidden`, same order, stopping at `layer` — no later layer feeds back into this one.
        // GPT-2's FFN is GELU (c_fc), not SwiGLU, so the "post-activation" vector here is post-GELU.
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>());
        for l in 0..=layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            for h in 0..self.n_head {
                let q = qkv.slice(s![.., h * hd..(h + 1) * hd]);
                let k = qkv.slice(s![.., self.d + h * hd..self.d + (h + 1) * hd]);
                let v = qkv.slice(s![.., 2 * self.d + h * hd..2 * self.d + (h + 1) * hd]);
                let mut scores = q.dot(&k.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e10; // causal mask
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&v));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight"))
                + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut h_mlp = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut h_mlp);
            if l == layer {
                return Some(h_mlp.row(seq - 1).to_vec());
            }
            x = &x + &(self.down(&h_mlp, &format!("{p}mlp.c_proj.weight"))
                + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        None
    }

    /// Explain the prediction: the live attention-head circuits + top MLP features at the predicting position.
    fn explanation(&self, ids: &[i64]) -> crate::explain::Explanation {
        use crate::explain::*;
        let seq = ids.len();
        let hd = self.d / self.n_head;
        let mut x = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &(0..seq as i64).collect::<Vec<_>>());
        let mut att_last: Vec<Vec<Vec<f32>>> = Vec::new(); // per layer, per head, last-position attention row
        let mut head_act: Vec<Vec<f32>> = Vec::new(); // per layer: attn_out's last row (d) — for head direct-logit attribution
        let mut mlp_h: Vec<Vec<f32>> = Vec::new();
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            let mut attn_out = Array2::<f32>::zeros((seq, self.d));
            let mut layer_att = Vec::with_capacity(self.n_head);
            for h in 0..self.n_head {
                let q = qkv.slice(s![.., h * hd..(h + 1) * hd]);
                let k = qkv.slice(s![.., self.d + h * hd..self.d + (h + 1) * hd]);
                let v = qkv.slice(s![.., 2 * self.d + h * hd..2 * self.d + (h + 1) * hd]);
                let mut scores = q.dot(&k.t()) / (hd as f32).sqrt();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[[i, j]] = -1e10;
                    }
                }
                softmax_rows(&mut scores);
                layer_att.push(scores.row(seq - 1).to_vec());
                attn_out.slice_mut(s![.., h * hd..(h + 1) * hd]).assign(&scores.dot(&v));
            }
            att_last.push(layer_att);
            head_act.push(attn_out.row(seq - 1).to_vec());
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            mlp_h.push(hm.row(seq - 1).to_vec());
            x = &x + &(self.b.mm(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        let xf = layernorm(&x, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"));
        let x_last = x.row(seq - 1).to_vec(); // residual just before the final norm
        let xf_last = xf.row(seq - 1).to_vec(); // …and just after — together they recover the norm's frozen scale
        let ln_bias = self.b.arr1("ln_f.bias").to_vec();
        let lg = self.b.rowdot_f32("wte", &xf_last);
        let model_predicts = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let gain = self.b.arr1("ln_f.weight").to_vec(); // final LayerNorm gain — for direct-logit attribution (center=true)
        let u_pred = self.b.weight_row("wte", model_predicts as usize); // predicted token's unembed row
        assemble(
            ids,
            &att_last,
            &head_act,
            &mlp_h,
            &lg,
            model_predicts,
            &gain,
            true,
            &ln_bias,
            &x_last,
            &xf_last,
            &u_pred,
            0,
            &|_v| Vec::new(),
            |l, n| self.b.weight_row(&format!("h{l}.mlp.c_proj.weight"), n), // neuron n's write direction (any dtype)
            |l, head| head_raw_contrib(&self.b, &format!("h{l}.attn.c_proj.weight"), &head_act[l], head, hd),
            |c| self.b.rowdot_f32("wte", c),
        )
    }

    /// Run `m` new positions (rows of `emb`, already token+position embeddings for absolute positions `cur..cur+m`)
    /// through all layers, appending their K/V to the cache and attending against the whole cache. Returns the
    /// pre-final-LN hidden states (m, d). Used for both prefill (m = prompt len, cur = 0) and decode (m = 1).
    fn forward_block(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array2<f32> {
        let d = self.d;
        let hd = d / self.n_head;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            kc[l].slice_mut(s![cur..klen, ..]).assign(&qkv.slice(s![.., d..2 * d])); // append new K/V to the cache
            vc[l].slice_mut(s![cur..klen, ..]).assign(&qkv.slice(s![.., 2 * d..3 * d]));
            let q = qkv.slice(s![.., 0..d]);
            let mut attn_out = Array2::<f32>::zeros((m, d));
            for hh in 0..self.n_head {
                let qh = q.slice(s![.., hh * hd..(hh + 1) * hd]);
                let kh = kc[l].slice(s![0..klen, hh * hd..(hh + 1) * hd]); // attend over the whole cache
                let vh = vc[l].slice(s![0..klen, hh * hd..(hh + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e10; // causal: new row i is absolute position cur+i
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., hh * hd..(hh + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            x = &x + &(self.down(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        x
    }

    /// Same as `forward_block` but with an int8 KV cache (per-position-per-head scale): quantise new K/V on write,
    /// dequantise on read. 4x smaller cache for long context; the small per-head quant error keeps tokens ~identical.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_q(&self, emb: &Array2<f32>, cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>],
                       vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]) -> Array2<f32> {
        let (d, h_n) = (self.d, self.n_head);
        let hd = d / h_n;
        let m = emb.nrows();
        let klen = cur + m;
        let mut x = emb.clone();
        let q8 = |v: f32, sc: f32| (v / sc).round().clamp(-127.0, 127.0) as i8;
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = layernorm(&x, self.b.arr1(&format!("{p}ln_1.weight")), self.b.arr1(&format!("{p}ln_1.bias")));
            let qkv = self.b.mm(&a, &format!("{p}attn.c_attn.weight")) + &self.b.arr1(&format!("{p}attn.c_attn.bias"));
            for i in 0..m {
                let pos = cur + i;
                for head in 0..h_n {
                    let (kb, vb) = (d + head * hd, 2 * d + head * hd);
                    let sck = (0..hd).fold(0f32, |mx, c| mx.max(qkv[[i, kb + c]].abs())) / 127.0;
                    let scv = (0..hd).fold(0f32, |mx, c| mx.max(qkv[[i, vb + c]].abs())) / 127.0;
                    let (sck, scv) = (if sck > 0.0 { sck } else { 1.0 }, if scv > 0.0 { scv } else { 1.0 });
                    ks[l][pos * h_n + head] = sck;
                    vs[l][pos * h_n + head] = scv;
                    for c in 0..hd {
                        kc[l][pos * d + head * hd + c] = q8(qkv[[i, kb + c]], sck);
                        vc[l][pos * d + head * hd + c] = q8(qkv[[i, vb + c]], scv);
                    }
                }
            }
            let q = qkv.slice(s![.., 0..d]);
            let mut attn_out = Array2::<f32>::zeros((m, d));
            for head in 0..h_n {
                let mut kh = Array2::<f32>::zeros((klen, hd)); // dequantise the cached head
                let mut vh = Array2::<f32>::zeros((klen, hd));
                for pos in 0..klen {
                    let (sck, scv) = (ks[l][pos * h_n + head], vs[l][pos * h_n + head]);
                    for c in 0..hd {
                        kh[[pos, c]] = kc[l][pos * d + head * hd + c] as f32 * sck;
                        vh[[pos, c]] = vc[l][pos * d + head * hd + c] as f32 * scv;
                    }
                }
                let qh = q.slice(s![.., head * hd..(head + 1) * hd]);
                let mut scores = qh.dot(&kh.t()) / (hd as f32).sqrt();
                for i in 0..m {
                    for j in (cur + i + 1)..klen {
                        scores[[i, j]] = -1e10;
                    }
                }
                softmax_rows(&mut scores);
                attn_out.slice_mut(s![.., head * hd..(head + 1) * hd]).assign(&scores.dot(&vh));
            }
            x = &x + &(self.b.mm(&attn_out, &format!("{p}attn.c_proj.weight")) + &self.b.arr1(&format!("{p}attn.c_proj.bias")));
            let a2 = layernorm(&x, self.b.arr1(&format!("{p}ln_2.weight")), self.b.arr1(&format!("{p}ln_2.bias")));
            let mut hm = self.b.mm(&a2, &format!("{p}mlp.c_fc.weight")) + &self.b.arr1(&format!("{p}mlp.c_fc.bias"));
            gelu(&mut hm);
            x = &x + &(self.down(&hm, &format!("{p}mlp.c_proj.weight")) + &self.b.arr1(&format!("{p}mlp.c_proj.bias")));
        }
        x
    }

    fn generate_kv_int8(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let (d, total) = (self.d, prompt.len() + n_new);
        let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * d]).collect();
        let mut vc = kc.clone();
        let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.n_head]).collect();
        let mut vs = ks.clone();
        let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
        let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos);
        let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]);
            let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    fn head_argmax(&self, xb: &Array2<f32>) -> i64 {
        let xfn = layernorm(xb, self.b.arr1("ln_f.weight"), self.b.arr1("ln_f.bias"));
        let logits = self.b.rowdot_f32("wte", &xfn.row(xb.nrows() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }
}

impl Model for Gpt2 {
    fn predict(&self, ids: &[i64]) -> i64 {
        let xf = self.hidden(ids); // unembed only the predicting position
        let logits = self.b.rowdot_f32("wte", &xf.row(ids.len() - 1).to_vec());
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64
    }

    fn explain(&self, ids: &[i64]) -> Option<crate::explain::Explanation> {
        Some(self.explanation(ids))
    }

    /// KV-cache generation: prefill the prompt once, then each new token only forwards its own row and attends against
    /// the cached K/V — O(1) layer work per token instead of re-running the whole context. Identical greedy tokens to
    /// the naive path (the cache is exact), just without the recompute.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        if self.kv_int8 {
            return self.generate_kv_int8(prompt, n_new);
        }
        let d = self.d;
        let total = prompt.len() + n_new;
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, d))).collect();
        let mut vc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, d))).collect();
        let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
        let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos); // prefill
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        loop {
            out.push(next);
            if out.len() == n_new {
                return out;
            }
            let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]); // decode one token
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
    }

    // KV-cached streaming (early-stop at eos + per-token emit) for chat/serve; mirrors `generate`'s f32 loop.
    fn generate_stream(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool) -> Vec<i64> {
        let d = self.d;
        let total = prompt.len() + max_tokens;
        if self.kv_int8 {
            // int8 KV cache for chat/serve — 4x smaller cache (longer context in the same budget); lossy by design.
            let mut kc: Vec<Vec<i8>> = (0..self.n_layer).map(|_| vec![0i8; total * d]).collect();
            let mut vc = kc.clone();
            let mut ks: Vec<Vec<f32>> = (0..self.n_layer).map(|_| vec![0f32; total * self.n_head]).collect();
            let mut vs = ks.clone();
            let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
            let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos);
            let xb = self.forward_block_q(&emb, 0, &mut kc, &mut ks, &mut vc, &mut vs);
            let mut next = self.head_argmax(&xb);
            let mut out = Vec::new();
            let mut pos = prompt.len();
            loop {
                if eos.contains(&next) { break; }
                out.push(next);
                if !emit(next) || out.len() == max_tokens { break; }
                let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]);
                let xb = self.forward_block_q(&e, pos, &mut kc, &mut ks, &mut vc, &mut vs);
                next = self.head_argmax(&xb);
                pos += 1;
            }
            return out;
        }
        let mut kc: Vec<Array2<f32>> = (0..self.n_layer).map(|_| Array2::zeros((total, d))).collect();
        let mut vc = kc.clone();
        let ppos: Vec<i64> = (0..prompt.len() as i64).collect();
        let emb = &self.b.rows_f32("wte", prompt) + &self.b.rows_f32("wpe", &ppos);
        let xb = self.forward_block(&emb, 0, &mut kc, &mut vc);
        let mut next = self.head_argmax(&xb);
        let mut out = Vec::new();
        let mut pos = prompt.len();
        loop {
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            if !emit(next) || out.len() == max_tokens {
                break;
            }
            let e = &self.b.rows_f32("wte", &[next]) + &self.b.rows_f32("wpe", &[pos as i64]);
            let xb = self.forward_block(&e, pos, &mut kc, &mut vc);
            next = self.head_argmax(&xb);
            pos += 1;
        }
        out
    }

    fn generate_stream_prefix(&self, prompt: &[i64], max_tokens: usize, eos: &[i64], emit: &mut dyn FnMut(i64) -> bool, cache: &mut crate::model::PrefixKv) -> Vec<i64> {
        if self.kv_int8 {
            let (d, n_head, n_layer) = (self.d, self.n_head, self.n_layer);
            let alloc = |total: usize| {
                let kc: Vec<Vec<i8>> = (0..n_layer).map(|_| vec![0i8; total * d]).collect();
                let vc = kc.clone();
                let ks: Vec<Vec<f32>> = (0..n_layer).map(|_| vec![0f32; total * n_head]).collect();
                let vs = ks.clone();
                (kc, vc, ks, vs)
            };
            let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Vec<i8>], ks: &mut [Vec<f32>], vc: &mut [Vec<i8>], vs: &mut [Vec<f32>]| {
                let ppos: Vec<i64> = (cur..cur + ids.len()).map(|i| i as i64).collect();
                let emb = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &ppos);
                self.forward_block_q(&emb, cur, kc, ks, vc, vs)
            };
            return crate::model::prefix_generate_q(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb));
        }
        let (d, n_layer) = (self.d, self.n_layer);
        let alloc = |total: usize| {
            let kc: Vec<Array2<f32>> = (0..n_layer).map(|_| Array2::zeros((total, d))).collect();
            let vc = kc.clone();
            (kc, vc)
        };
        let mut fwd = |ids: &[i64], cur: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]| {
            let ppos: Vec<i64> = (cur..cur + ids.len()).map(|i| i as i64).collect();
            let emb = &self.b.rows_f32("wte", ids) + &self.b.rows_f32("wpe", &ppos);
            self.forward_block(&emb, cur, kc, vc)
        };
        crate::model::prefix_generate(prompt, max_tokens, eos, emit, cache, n_layer, &alloc, &mut fwd, &|xb, _ctx| self.head_argmax(xb))
    }
}

#[cfg(test)]
mod pdc_tests {
    use super::*;
    #[test]
    fn observer_identity_fails_closed() {
        let observed = ndarray::arr2(&[[1.0, 2.0]]);
        assert!(observer_identity(&observed, &ndarray::arr2(&[[1.0, 2.1]])).is_err());
        assert!(observer_identity(&observed, &ndarray::arr2(&[[1.0, f32::NAN]])).is_err());
    }
    #[test]
    fn observer_counts_only_the_observed_scorer() {
        let dir = std::env::temp_dir().join(format!("fr_gpt2_pdc_review_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("b").to_string_lossy().into_owned();
        let mut bin = Vec::new();
        let mut arrays = Vec::new();
        let mut add = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
            let offset = bin.len();
            for x in values { bin.extend_from_slice(&x.to_le_bytes()); }
            arrays.push(serde_json::json!({"name":name,"shape":shape,"dtype":"f32","offset":offset,"bytes":bin.len()-offset}));
        };
        add("wte", vec![3,2], vec![1.,0.,0.,1.,2.,-1.]);
        add("wpe", vec![3,2], vec![0.;6]);
        for prefix in ["ln_f", "h0.ln_1", "h0.ln_2"] {
            add(&format!("{prefix}.weight"), vec![2], vec![1.;2]);
            add(&format!("{prefix}.bias"), vec![2], vec![0.;2]);
        }
        for (name, shape, bias) in [("h0.attn.c_attn",vec![2,6],6),
            ("h0.attn.c_proj",vec![2,2],2),("h0.mlp.c_fc",vec![2,8],8),
            ("h0.mlp.c_proj",vec![8,2],2)] {
            add(&format!("{name}.weight"), shape.clone(), vec![0.;shape.iter().product()]);
            add(&format!("{name}.bias"), vec![bias], vec![0.;bias]);
        }
        std::fs::write(format!("{stem}.fieldrun.bin"), bin).unwrap();
        std::fs::write(format!("{stem}.fieldrun.json"), serde_json::to_vec(&serde_json::json!({
            "format":crate::bundle::FORMAT,"version":crate::bundle::VERSION,"arch":"gpt2",
            "config":[1,1,2,3,3],"arrays":arrays})).unwrap()).unwrap();
        let model = Gpt2::new(Bundle::load(&stem).unwrap(),0.,false);
        for subset in [false,true] {
            let mut last = None;
            for observer in [false,true] {
                let selection: ReadSelection = serde_json::from_value(serde_json::json!({
                    "layers":[0],"heads":[0],"positions":[1],"contexts":[[0,1]],
                    "capture_sources":true,"candidate_tokens":[0,1],
                    "score_candidates_only":subset,"check_observer":observer})).unwrap();
                let result = model.read_dump(&selection).unwrap();
                let case = &result["cases"][0];
                assert_eq!(case["native_output_row_calls"], if subset {4} else {6});
                if observer {
                    assert_eq!(case["observer_hidden_max_error"],0.);
                    assert_eq!(case["unobserved_predictions"],case["predictions"]);
                    assert_eq!(last.as_ref().unwrap(), &case["predictions"]);
                } else { last = Some(case["predictions"].clone()); }
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
