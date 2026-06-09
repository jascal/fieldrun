//! The fieldrun HTTP API. A minimal blocking `tiny_http` server (no async runtime). Two layers:
//!
//!  - **native** (always available) — token ids in/out, no tokenizer:
//!      POST /predict   {"ids":[...]}            -> {"next": <id>}
//!      POST /generate  {"prompt":[...],"n":N}   -> {"tokens":[...]}
//!      POST /explain   {"ids":[...]}            -> <Explanation JSON>   (archs with explain; else {"error":...})
//!      GET  /health                             -> {"ok": true}
//!
//!  - **OpenAI- + Anthropic-compatible** (text in/out, `--features api`, needs a `<stem>.tokenizer.json`):
//!      POST /v1/chat/completions  · POST /v1/completions · GET /v1/models   (OpenAI)
//!      POST /v1/messages                                                    (Anthropic)
//!    Greedy generation; `max_tokens` honoured; output stops at the model's EOS. Chat uses a ChatML template (a
//!    reasonable generic default — not necessarily the model's exact trained template). Not the model's real
//!    tokenizer? then text endpoints 400; the native endpoints still work. No streaming yet (responses are whole).

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

/// Text generation over a bundled tokenizer (the OpenAI/Anthropic + `--chat` layer). Only built with `--features api`.
#[cfg(feature = "api")]
pub struct TextGen {
    tok: tokenizers::Tokenizer,
    eos: Vec<i64>,
}

#[cfg(feature = "api")]
impl TextGen {
    /// Load `<stem>.tokenizer.json` (written by `convert`); `eos` from the bundle. None if there's no tokenizer.
    pub fn load(stem: &str, eos: Vec<i64>) -> Option<TextGen> {
        let path = format!("{stem}.tokenizer.json");
        tokenizers::Tokenizer::from_file(&path).ok().map(|tok| TextGen { tok, eos })
    }

    fn encode(&self, text: &str, add_special: bool) -> Vec<i64> {
        self.tok
            .encode(text, add_special)
            .map(|e| e.get_ids().iter().map(|&u| u as i64).collect())
            .unwrap_or_default()
    }

    fn decode(&self, ids: &[i64]) -> String {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tok.decode(&u, true).unwrap_or_default()
    }

    /// ChatML prompt (`<|im_start|>role\n…<|im_end|>`), a common default (Qwen/others). `history` is prior (role, text).
    pub fn chat_prompt(&self, system: Option<&str>, history: &[(String, String)], user: &str) -> String {
        let mut s = String::new();
        if let Some(sys) = system {
            s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }
        for (role, content) in history {
            s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        s.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
        s
    }

    /// Generate from a text prompt with **early-stop at EOS** (no compute past the natural end). `on_text` receives
    /// each newly-decoded text chunk *as it is produced* — used for live chat + SSE streaming. Returns
    /// (full_text, prompt_tokens, completion_tokens, hit_eos).
    pub fn gen(
        &self,
        lm: &dyn Model,
        prompt: &str,
        max_tokens: usize,
        add_special: bool,
        on_text: &mut dyn FnMut(&str),
    ) -> (String, usize, usize, bool) {
        let ids = self.encode(prompt, add_special);
        let mut acc: Vec<i64> = Vec::new();
        let mut prev = String::new();
        // decode the full accumulator each step and emit the byte-prefix delta — robust to BPE multi-byte/merge tokens.
        let out = lm.generate_stream(&ids, max_tokens, &self.eos, &mut |t| {
            acc.push(t);
            let text = self.decode(&acc);
            if text.starts_with(&prev) {
                let delta = &text[prev.len()..];
                if !delta.is_empty() {
                    on_text(delta);
                }
            }
            prev = text;
            true
        });
        (prev, ids.len(), out.len(), out.len() < max_tokens)
    }
}

#[cfg(not(feature = "api"))]
pub struct TextGen;

pub fn serve(lm: Box<dyn Model>, arch: &str, port: u16, textgen: Option<TextGen>) {
    let server = tiny_http::Server::http(("0.0.0.0", port)).expect("bind port");
    let openai = if cfg!(feature = "api") && textgen.is_some() {
        " · OpenAI /v1/chat/completions /v1/completions · Anthropic /v1/messages"
    } else {
        " (token-id API; build --features api + a tokenizer for OpenAI/Anthropic text endpoints)"
    };
    eprintln!("[fieldrun] serving {arch} on http://0.0.0.0:{port}  (POST /predict /generate /explain · GET /health{openai})");
    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let json = handle(&url, &body, lm.as_ref(), arch, textgen.as_ref());
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
        let _ = req.respond(tiny_http::Response::from_string(json).with_header(header));
    }
}

fn handle(url: &str, body: &str, lm: &dyn Model, arch: &str, tg: Option<&TextGen>) -> String {
    #[cfg(not(feature = "api"))]
    let _ = &tg; // only used by the OpenAI/Anthropic routes (feature `api`)
    let route = url.split('?').next().unwrap_or(url);
    match route {
        "/health" => "{\"ok\":true}".to_string(),
        "/predict" => match serde_json::from_str::<PredictReq>(body) {
            Ok(r) if !r.ids.is_empty() => format!("{{\"next\":{}}}", lm.predict(&r.ids)),
            _ => err("bad body; expected {\"ids\":[...]}"),
        },
        "/generate" => match serde_json::from_str::<GenerateReq>(body) {
            Ok(r) if !r.prompt.is_empty() => serde_json::json!({ "tokens": lm.generate(&r.prompt, r.n) }).to_string(),
            _ => err("bad body; expected {\"prompt\":[...],\"n\":N}"),
        },
        "/explain" => match serde_json::from_str::<ExplainReq>(body) {
            Ok(r) if !r.ids.is_empty() => match lm.explain(&r.ids) {
                Some(ex) => serde_json::to_string(&ex).unwrap(),
                None => err("explain not supported for this arch"),
            },
            _ => err("bad body; expected {\"ids\":[...]}"),
        },
        "/v1/models" => serde_json::json!({ "object": "list",
            "data": [{ "id": arch, "object": "model", "owned_by": "fieldrun" }] }).to_string(),
        #[cfg(feature = "api")]
        "/v1/chat/completions" | "/v1/completions" | "/v1/messages" => match tg {
            Some(tg) => openai_anthropic(route, body, lm, arch, tg),
            None => err("no tokenizer for this bundle — re-run `convert` so it copies tokenizer.json next to the bundle"),
        },
        #[cfg(not(feature = "api"))]
        "/v1/chat/completions" | "/v1/completions" | "/v1/messages" => {
            err("text endpoints need a build with `--features api` (the default build serves the token-id API)")
        }
        _ => err("unknown route (POST /predict /generate /explain /v1/chat/completions /v1/completions /v1/messages, GET /health /v1/models)"),
    }
}

#[cfg(feature = "api")]
fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(feature = "api")]
#[derive(serde::Deserialize)]
struct Msg {
    role: String,
    content: String,
}

#[cfg(feature = "api")]
fn openai_anthropic(route: &str, body: &str, lm: &dyn Model, arch: &str, tg: &TextGen) -> String {
    #[derive(serde::Deserialize)]
    struct Req {
        #[serde(default)]
        messages: Vec<Msg>,
        #[serde(default)]
        prompt: String,
        #[serde(default)]
        system: String,
        #[serde(default)]
        max_tokens: Option<usize>,
    }
    let r: Req = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return err(&format!("bad JSON body: {e}")),
    };
    let max_tokens = r.max_tokens.unwrap_or(256).clamp(1, 4096);

    if route == "/v1/completions" {
        // OpenAI text completion — raw prompt, with the tokenizer's special tokens
        let (text, pt, ct, eos) = tg.gen(lm, &r.prompt, max_tokens, true, &mut |_| {});
        return serde_json::json!({
            "id": format!("cmpl-{}", now()), "object": "text_completion", "created": now(), "model": arch,
            "choices": [{ "text": text, "index": 0, "finish_reason": if eos {"stop"} else {"length"} }],
            "usage": { "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct }
        }).to_string();
    }

    // chat (OpenAI /v1/chat/completions and Anthropic /v1/messages share the message shape)
    let mut system = if r.system.is_empty() { None } else { Some(r.system.clone()) };
    let mut history: Vec<(String, String)> = Vec::new();
    let mut last_user = String::new();
    for (i, m) in r.messages.iter().enumerate() {
        if m.role == "system" {
            system = Some(m.content.clone());
        } else if i == r.messages.len() - 1 && m.role == "user" {
            last_user = m.content.clone();
        } else {
            history.push((m.role.clone(), m.content.clone()));
        }
    }
    let prompt = tg.chat_prompt(system.as_deref(), &history, &last_user);
    let (text, pt, ct, eos) = tg.gen(lm, &prompt, max_tokens, false, &mut |_| {});

    if route == "/v1/messages" {
        // Anthropic Messages API
        serde_json::json!({
            "id": format!("msg_{}", now()), "type": "message", "role": "assistant", "model": arch,
            "content": [{ "type": "text", "text": text }],
            "stop_reason": if eos {"end_turn"} else {"max_tokens"},
            "usage": { "input_tokens": pt, "output_tokens": ct }
        }).to_string()
    } else {
        // OpenAI chat completion
        serde_json::json!({
            "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": arch,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text },
                          "finish_reason": if eos {"stop"} else {"length"} }],
            "usage": { "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct }
        }).to_string()
    }
}

/// Interactive REPL chat over the bundled tokenizer (the `--chat` mode). Maintains conversation history; ChatML prompt.
#[cfg(feature = "api")]
pub fn chat(lm: Box<dyn Model>, tg: TextGen, max_tokens: usize) {
    use std::io::Write;
    eprintln!("[fieldrun] chat — type a message, Ctrl-D to exit. (greedy, max_tokens={max_tokens}; generic ChatML template)");
    let mut history: Vec<(String, String)> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("\nyou> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            eprintln!("\n[fieldrun] bye");
            break;
        }
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        let prompt = tg.chat_prompt(None, &history, user);
        print!("bot> ");
        let _ = std::io::stdout().flush();
        // stream the reply token-by-token to the terminal as it's generated
        let (text, _, _, _) = tg.gen(lm.as_ref(), &prompt, max_tokens, false, &mut |chunk| {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
        });
        println!();
        history.push(("user".into(), user.to_string()));
        history.push(("assistant".into(), text.trim().to_string()));
    }
}

fn err(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}
