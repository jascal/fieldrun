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
        #[cfg(feature = "api")]
        let route = url.split('?').next().unwrap_or(&url).to_string();
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        // SSE streaming for the text endpoints when the client asks for `"stream": true`.
        #[cfg(feature = "api")]
        if let Some(tg) = textgen.as_ref() {
            let streamable = matches!(route.as_str(), "/v1/chat/completions" | "/v1/completions" | "/v1/messages");
            if streamable && wants_stream(&body) {
                serve_stream(req, &route, &body, lm.as_ref(), arch, tg);
                continue;
            }
        }
        let json = handle(&url, &body, lm.as_ref(), arch, textgen.as_ref());
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
        let _ = req.respond(tiny_http::Response::from_string(json).with_header(header));
    }
}

#[cfg(feature = "api")]
fn wants_stream(body: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct S {
        #[serde(default)]
        stream: bool,
    }
    serde_json::from_str::<S>(body).map(|s| s.stream).unwrap_or(false)
}

/// A `Read` that drains SSE chunks from a channel (fed by the generation thread) — tiny_http reads it lazily and
/// writes each chunk to the socket (chunked transfer), so tokens stream to the client as they're produced.
#[cfg(feature = "api")]
struct SseReader {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

#[cfg(feature = "api")]
impl std::io::Read for SseReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                    if self.buf.is_empty() {
                        return Ok(0);
                    }
                }
                Err(_) => return Ok(0), // generation done, channel closed -> EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Stream a chat/completion as Server-Sent Events. Generation runs on a scoped thread (borrowing the model — `Model:
/// Sync` makes `&dyn Model` Send) and pushes SSE frames into a channel that the response reader drains.
#[cfg(feature = "api")]
fn serve_stream(req: tiny_http::Request, route: &str, body: &str, lm: &dyn Model, arch: &str, tg: &TextGen) {
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
    let r: Req = serde_json::from_str(body).unwrap_or(Req { messages: vec![], prompt: String::new(), system: String::new(), max_tokens: None });
    let max_tokens = r.max_tokens.unwrap_or(256).clamp(1, 4096);
    let (prompt, add_special) = if route == "/v1/completions" {
        (r.prompt.clone(), true)
    } else {
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
        (tg.chat_prompt(system.as_deref(), &history, &last_user), false)
    };
    let route = route.to_string();
    let arch = arch.to_string();
    std::thread::scope(|s| {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        s.spawn(move || {
            let id = now();
            let open = sse_open(&route, &arch, id);
            if !open.is_empty() {
                let _ = tx.send(open); // OpenAI has no preamble; never send an empty chunk (the reader treats it as EOF)
            }
            let mut on_text = |chunk: &str| {
                let _ = tx.send(sse_delta(&route, &arch, id, chunk));
            };
            tg.gen(lm, &prompt, max_tokens, add_special, &mut on_text);
            let _ = tx.send(sse_close(&route, &arch, id));
            // tx dropped here -> channel closes -> reader EOFs
        });
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap();
        let resp = tiny_http::Response::new(200.into(), vec![header], SseReader { rx, buf: Vec::new(), pos: 0 }, None, None);
        let _ = req.respond(resp);
    });
}

#[cfg(feature = "api")]
fn sse_open(route: &str, arch: &str, id: u64) -> Vec<u8> {
    if route == "/v1/messages" {
        // Anthropic: message_start + content_block_start
        let start = serde_json::json!({"type":"message_start","message":{"id":format!("msg_{id}"),"type":"message",
            "role":"assistant","model":arch,"content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}});
        let cbs = serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}});
        format!("event: message_start\ndata: {start}\n\nevent: content_block_start\ndata: {cbs}\n\n").into_bytes()
    } else {
        Vec::new() // OpenAI has no preamble event
    }
}

#[cfg(feature = "api")]
fn sse_delta(route: &str, arch: &str, id: u64, text: &str) -> Vec<u8> {
    let j = match route {
        "/v1/messages" => serde_json::json!({"type":"content_block_delta","index":0,
            "delta":{"type":"text_delta","text":text}}),
        "/v1/completions" => serde_json::json!({"id":format!("cmpl-{id}"),"object":"text_completion","model":arch,
            "choices":[{"text":text,"index":0,"finish_reason":serde_json::Value::Null}]}),
        _ => serde_json::json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","model":arch,
            "choices":[{"index":0,"delta":{"content":text},"finish_reason":serde_json::Value::Null}]}),
    };
    if route == "/v1/messages" {
        format!("event: content_block_delta\ndata: {j}\n\n").into_bytes()
    } else {
        format!("data: {j}\n\n").into_bytes()
    }
}

#[cfg(feature = "api")]
fn sse_close(route: &str, arch: &str, id: u64) -> Vec<u8> {
    if route == "/v1/messages" {
        let md = serde_json::json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}});
        format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                 event: message_delta\ndata: {md}\n\n\
                 event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n").into_bytes()
    } else {
        let fin = serde_json::json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","model":arch,
            "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
        format!("data: {fin}\n\ndata: [DONE]\n\n").into_bytes()
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
        // "thinking" spinner until the first token (the gap = prompt prefill, which is slow on big models), then the
        // reply streams token-by-token. The spinner runs on its own thread + writes stderr; on the first token we stop
        // and join it (so it's done writing) before printing the reply, so the two never race on the line.
        let thinking = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let th = thinking.clone();
        let mut spinner = Some(std::thread::spawn(move || {
            let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let t0 = std::time::Instant::now();
            let mut i = 0usize;
            while th.load(std::sync::atomic::Ordering::Relaxed) {
                eprint!("\r\x1b[2K[ thinking {} {:.0}s ]", frames[i % frames.len()], t0.elapsed().as_secs_f64());
                let _ = std::io::stderr().flush();
                std::thread::sleep(std::time::Duration::from_millis(120));
                i += 1;
            }
            eprint!("\r\x1b[2K"); // clear the spinner line
            let _ = std::io::stderr().flush();
        }));
        let mut started = false;
        let (text, _, _, _) = tg.gen(lm.as_ref(), &prompt, max_tokens, false, &mut |chunk| {
            if !started {
                started = true;
                thinking.store(false, std::sync::atomic::Ordering::Relaxed);
                if let Some(h) = spinner.take() {
                    let _ = h.join();
                }
                print!("bot> ");
            }
            print!("{chunk}");
            let _ = std::io::stdout().flush();
        });
        if let Some(h) = spinner.take() {
            // no tokens were produced (empty/immediate-eos reply) — stop the spinner and still show the prompt
            thinking.store(false, std::sync::atomic::Ordering::Relaxed);
            let _ = h.join();
            print!("bot> ");
        }
        println!();
        history.push(("user".into(), user.to_string()));
        history.push(("assistant".into(), text.trim().to_string()));
    }
}

fn err(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

#[cfg(all(test, feature = "api"))]
mod tests {
    use super::*;

    #[test]
    fn wants_stream_parses() {
        assert!(wants_stream(r#"{"stream":true}"#));
        assert!(!wants_stream(r#"{"stream":false}"#));
        assert!(!wants_stream(r#"{"messages":[]}"#));
        assert!(!wants_stream("not json"));
    }

    #[test]
    fn sse_openai_format() {
        let d = String::from_utf8(sse_delta("/v1/chat/completions", "rope", 1, "Paris")).unwrap();
        assert!(d.starts_with("data: "), "{d}");
        assert!(d.contains("\"content\":\"Paris\"") && d.contains("chat.completion.chunk") && d.ends_with("\n\n"));
        let done = String::from_utf8(sse_close("/v1/chat/completions", "rope", 1)).unwrap();
        assert!(done.contains("[DONE]") && done.contains("\"finish_reason\":\"stop\""), "{done}");
    }

    #[test]
    fn sse_anthropic_format() {
        let open = String::from_utf8(sse_open("/v1/messages", "rope", 1)).unwrap();
        assert!(open.contains("message_start") && open.contains("content_block_start"));
        let d = String::from_utf8(sse_delta("/v1/messages", "rope", 1, "Hi")).unwrap();
        assert!(d.contains("content_block_delta") && d.contains("\"text\":\"Hi\""), "{d}");
    }
}
