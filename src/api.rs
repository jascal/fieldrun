//! The fieldrun HTTP API — one loaded model served over `/predict`, `/generate`, `/explain`. A minimal blocking
//! `tiny_http` server (no async runtime), so the runtime stays a small native binary. Requests and responses are JSON
//! over token ids; a frontend (or the CLI's vocab decoder) handles tokenisation. The model is loaded once at startup.
//!
//!   POST /predict   {"ids":[...]}                 -> {"next": <id>}
//!   POST /generate  {"prompt":[...], "n":N}       -> {"tokens":[...]}
//!   POST /explain   {"ids":[...]}                  -> <Explanation JSON>  (GPT-2 only; else {"error":...})
//!   GET  /health                                  -> {"ok": true}

use serde::Deserialize;

use crate::model::Model;

#[derive(Deserialize)]
struct PredictReq {
    ids: Vec<i64>,
}

#[derive(Deserialize)]
struct GenerateReq {
    prompt: Vec<i64>,
    #[serde(default = "default_n")]
    n: usize,
}

fn default_n() -> usize {
    32
}

#[derive(Deserialize)]
struct ExplainReq {
    ids: Vec<i64>,
}

pub fn serve(lm: Box<dyn Model>, arch: &str, port: u16) {
    let server = tiny_http::Server::http(("0.0.0.0", port)).expect("bind port");
    eprintln!("[fieldrun] serving {arch} on http://0.0.0.0:{port}  (POST /predict /generate /explain · GET /health)");
    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let json = handle(&url, &body, lm.as_ref());
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
        let _ = req.respond(tiny_http::Response::from_string(json).with_header(header));
    }
}

fn handle(url: &str, body: &str, lm: &dyn Model) -> String {
    match url.split('?').next().unwrap_or(url) {
        "/health" => "{\"ok\":true}".to_string(),
        "/predict" => match serde_json::from_str::<PredictReq>(body) {
            Ok(r) if !r.ids.is_empty() => format!("{{\"next\":{}}}", lm.predict(&r.ids)),
            _ => err("bad body; expected {\"ids\":[...]}"),
        },
        "/generate" => match serde_json::from_str::<GenerateReq>(body) {
            Ok(r) if !r.prompt.is_empty() => {
                serde_json::json!({ "tokens": lm.generate(&r.prompt, r.n) }).to_string()
            }
            _ => err("bad body; expected {\"prompt\":[...],\"n\":N}"),
        },
        "/explain" => match serde_json::from_str::<ExplainReq>(body) {
            Ok(r) if !r.ids.is_empty() => match lm.explain(&r.ids) {
                Some(ex) => serde_json::to_string(&ex).unwrap(),
                None => err("explain not supported for this arch"),
            },
            _ => err("bad body; expected {\"ids\":[...]}"),
        },
        _ => err("unknown route (try POST /predict /generate /explain, GET /health)"),
    }
}

fn err(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}
