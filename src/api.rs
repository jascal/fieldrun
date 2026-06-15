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
//!    tokenizer? then text endpoints 400; the native endpoints still work. `"stream": true` streams tokens as SSE.
//!    Tool/function calling: pass `tools` (OpenAI `{type:"function",function:{…}}` or Anthropic `{name,input_schema}`)
//!    and fieldrun declares them in the prompt and returns structured `tool_calls` / `tool_use` (see `tools.rs`). When
//!    `stream:true`, tool calls are emitted as SSE deltas (OpenAI `chat.completion.chunk` `tool_calls` / Anthropic
//!    `tool_use` content blocks) — fieldrun buffers the generation to parse the calls, then streams the parsed result,
//!    so a streaming client (e.g. opencode via the AI SDK) always gets a `text/event-stream`, never a bare JSON body.
//!    fieldrun extension: `"explain": true` attaches the structured Explanation under a `fieldrun_explanation` field
//!    (non-streaming response; clients ignore the unknown field) and, under `--serve`, also prints it to the server
//!    console; canonical structured route: POST /explain.

use serde::Deserialize;

use crate::model::Model;
use crate::retrieval::{CandCfg, Store}; // ExplainOpts (held by serve in both builds) carries these
#[cfg(feature = "api")]
use crate::explain::ExplainMode;
#[cfg(feature = "api")]
use crate::retrieval::{context_candidates, induction_rule, RuleHit};

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

/// The result of one `TextGen::gen` call. Beyond the text + token counts, it carries the prompt and generated token
/// ids so callers can run `explain_steps` — an explanation per generated token (every forward pass of the reply).
#[cfg(feature = "api")]
pub struct GenOut {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub hit_eos: bool,
    pub prompt_ids: Vec<i64>,
    pub gen_ids: Vec<i64>,
}

/// Text generation over a bundled tokenizer (the OpenAI/Anthropic + `--chat` layer). Only built with `--features api`.
#[cfg(feature = "api")]
pub struct TextGen {
    tok: tokenizers::Tokenizer,
    eos: Vec<i64>,
    // Single-slot prefix-KV cache shared across requests so a growing chat reuses the K/V of its common prefix instead
    // of re-prefilling the whole context each turn. The server handles one request at a time (the SSE generation runs
    // on a scoped thread that joins before the next request), so this lock is always uncontended.
    prefix: std::sync::Mutex<crate::model::PrefixKv>,
}

#[cfg(feature = "api")]
impl TextGen {
    /// Load `<stem>.tokenizer.json` (written by `convert`); `eos` from the bundle. None if there's no tokenizer.
    pub fn load(stem: &str, eos: Vec<i64>) -> Option<TextGen> {
        let path = format!("{stem}.tokenizer.json");
        tokenizers::Tokenizer::from_file(&path)
            .ok()
            .map(|tok| TextGen { tok, eos, prefix: std::sync::Mutex::new(crate::model::PrefixKv::default()) })
    }

    fn encode(&self, text: &str, add_special: bool) -> Vec<i64> {
        self.tok
            .encode(text, add_special)
            .map(|e| e.get_ids().iter().map(|&u| u as i64).collect())
            .unwrap_or_default()
    }

    /// The token string for an id, including special tokens (e.g. 151644 → "<|im_start|>") — for explain labels, where
    /// `decode` blanks special tokens. Returns the raw vocab token; None if the id is out of range.
    fn id_to_token(&self, id: i64) -> Option<String> {
        u32::try_from(id).ok().and_then(|u| self.tok.id_to_token(u))
    }

    fn decode(&self, ids: &[i64]) -> String {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tok.decode(&u, true).unwrap_or_default()
    }

    /// A display label for a token id, used in explain output: its decoded text quoted (`" lunch" [54809]`), or for a
    /// special token that decodes to "" its vocab name (`<|im_start|> [151644]`), with the id appended.
    pub fn token_label(&self, id: i64) -> String {
        let s = self.decode(&[id]);
        let meaning = if !s.is_empty() { format!("{s:?}") } else { self.id_to_token(id).unwrap_or_default() };
        if meaning.is_empty() {
            format!("[{id}]")
        } else {
            format!("{meaning} [{id}]")
        }
    }

    /// Sensible default reply cap when the caller/CLI doesn't set one. Token budget tracks *thinking*, not model size:
    /// a reasoning model (a tokenizer that knows a `<think>`-style token) spends hundreds-to-thousands of tokens before
    /// the answer, so 256 would truncate mid-thought — give those 2048; everything else 512. Always overridable
    /// (`--max-tokens` on the CLI, `"max_tokens"` in an API request).
    /// Does the tokenizer know the ChatML template token `<|im_start|>`? If not, it's almost certainly a base/completion
    /// model (e.g. GPT-2), and `--chat` (which wraps input in a ChatML template) will just continue text, not converse.
    pub fn knows_chatml(&self) -> bool {
        self.tok.token_to_id("<|im_start|>").is_some()
    }

    pub fn default_max_tokens(&self) -> usize {
        let reasoning = ["<think>", "<thinking>", "<|thinking|>", "<reasoning>"]
            .iter()
            .any(|t| self.tok.token_to_id(t).is_some());
        if reasoning {
            2048
        } else {
            512
        }
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
    /// (full_text, prompt_tokens, completion_tokens, hit_eos, prompt_ids, generated_ids). The two id vectors let the
    /// caller explain *every* forward pass of the reply (one per generated token), not just the end-of-prompt one.
    pub fn gen(
        &self,
        lm: &dyn Model,
        prompt: &str,
        max_tokens: usize,
        add_special: bool,
        on_text: &mut dyn FnMut(&str),
    ) -> GenOut {
        let ids = self.encode(prompt, add_special);
        let mut acc: Vec<i64> = Vec::new();
        // WINDOWED incremental decode: decode only `acc[start..]` each step and emit the byte-prefix delta — vs
        // re-decoding the WHOLE accumulator every token (O(n)/token → O(n²) over a generation). `start` is re-anchored
        // forward once the window grows past CAP, by ADV tokens, so the per-token decode stays O(window) not O(n). The
        // re-anchor moves the start far BEHIND the emission tail (≥ CAP-ADV tokens), so BPE/UTF-8 boundary effects at
        // the new start can't reach the bytes we emit, and the baseline `prev` is re-decoded from the SAME start as the
        // following tokens, so any decoder-specific leading quirk cancels in the delta. `prev` = decoded text of the
        // current window. Robust to BPE multi-byte/merge tokens (the `starts_with` guard skips a non-extending update).
        const CAP: usize = 64;
        const ADV: usize = 32;
        let mut start = 0usize;
        let mut prev = String::new();
        // Reuse the K/V of the prefix this prompt shares with the previous turn (uncontended lock; recover a poisoned
        // mutex rather than cascade a panic).
        let mut cache = self.prefix.lock().unwrap_or_else(|e| e.into_inner());
        let out = lm.generate_stream_prefix(&ids, max_tokens, &self.eos, &mut |t| {
            acc.push(t);
            if acc.len() - start > CAP {
                start += ADV;
                prev = self.decode(&acc[start..acc.len() - 1]); // re-baseline to the window WITHOUT this token, so its bytes still emit below
            }
            let text = self.decode(&acc[start..]);
            if text.starts_with(&prev) {
                let delta = &text[prev.len()..];
                if !delta.is_empty() {
                    on_text(delta);
                }
            }
            prev = text;
            true
        }, &mut cache);
        let hit_eos = out.len() < max_tokens;
        // authoritative full reply for the stored text (chat history / non-streaming response) — one O(n) decode, not per token.
        let text = self.decode(&acc);
        GenOut { text, prompt_tokens: ids.len(), completion_tokens: out.len(), hit_eos, prompt_ids: ids, gen_ids: out }
    }
}

#[cfg(not(feature = "api"))]
pub struct TextGen;

pub fn serve(lm: Box<dyn Model>, arch: &str, port: u16, textgen: Option<TextGen>, explain: ExplainOpts) {
    let server = match tiny_http::Server::http(("0.0.0.0", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[fieldrun] couldn't bind port {port}: {e} (already in use, or privileged <1024?). \
                       Try a different --serve PORT.");
            std::process::exit(1);
        }
    };
    let openai = if cfg!(feature = "api") && textgen.is_some() {
        " · OpenAI /v1/chat/completions /v1/completions · Anthropic /v1/messages"
    } else {
        " (token-id API; build --features api + a tokenizer for OpenAI/Anthropic text endpoints)"
    };
    eprintln!("[fieldrun] serving {arch} on http://0.0.0.0:{port}  (POST /predict /generate /explain · GET /health{openai})");
    for mut req in server.incoming_requests() {
        let method = req.method().to_string();
        let url = req.url().to_string();
        #[cfg(feature = "api")]
        let route = url.split('?').next().unwrap_or(&url).to_string();
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        // one log line per request so `--serve` shows activity (route + stream/tools/explain flags), not just silence.
        log_request(&method, url.split('?').next().unwrap_or(&url), &body);
        // SSE streaming for the text endpoints when the client asks for `"stream": true` — INCLUDING tool requests
        // (serve_stream buffers the generation, parses the calls, then emits them as SSE; the old code answered tool
        // requests with a non-streaming JSON body even when the client asked to stream, which an SSE client can't read).
        #[cfg(feature = "api")]
        if let Some(tg) = textgen.as_ref() {
            let streamable = matches!(route.as_str(), "/v1/chat/completions" | "/v1/completions" | "/v1/messages");
            if streamable && wants_stream(&body) {
                serve_stream(req, &route, &body, lm.as_ref(), arch, tg, &explain);
                continue;
            }
        }
        let json = handle(&url, &body, lm.as_ref(), arch, textgen.as_ref(), &explain);
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
        let _ = req.respond(tiny_http::Response::from_string(json).with_header(header));
    }
}

/// One concise console line per request under `--serve`, summarising the route and the request flags that matter for
/// debugging a client integration (streaming on/off, how many tools were offered, whether the explain extension is set).
fn log_request(method: &str, route: &str, body: &str) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let b = |k: &str| v.get(k).and_then(|x| x.as_bool()).unwrap_or(false);
    let n_arr = |k: &str| v.get(k).and_then(|x| x.as_array()).map(|a| a.len()).unwrap_or(0);
    let n_tools = n_arr("tools") + n_arr("functions");
    let mut flags: Vec<String> = Vec::new();
    if b("stream") {
        flags.push("stream".into());
    }
    if n_tools > 0 {
        flags.push(format!("tools={n_tools}"));
    }
    // explain is bool OR a level string ("route"/"circuits"/"all"); show the level when present and not off/false.
    match v.get("explain") {
        Some(serde_json::Value::Bool(true)) => flags.push("explain".into()),
        Some(serde_json::Value::String(s)) if !matches!(s.to_ascii_lowercase().as_str(), "off" | "false" | "none" | "no" | "0") => {
            flags.push(format!("explain={s}"))
        }
        _ => {}
    }
    let suffix = if flags.is_empty() { String::new() } else { format!(" [{}]", flags.join(", ")) };
    eprintln!("[fieldrun] {method} {route}{suffix}");
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

/// Stream a chat/completion as Server-Sent Events.
///
/// Two paths share one transport (`text/event-stream`):
///  - **no tools** — generation runs on a scoped thread (borrowing the model — `Model: Sync` makes `&dyn Model` Send)
///    and pushes SSE frames into a channel as each token is produced, so the client sees text live.
///  - **tools offered** — we must see the *whole* output to parse `<tool_call>` blocks out of it, so we generate fully
///    (buffered), parse, and then emit the parsed result as SSE frames (tool-call deltas, or the plain text if the
///    model just answered). The client still gets a stream — just delivered after generation rather than token-by-token.
#[cfg(feature = "api")]
fn serve_stream(req: tiny_http::Request, route: &str, body: &str, lm: &dyn Model, arch: &str, tg: &TextGen, explain: &ExplainOpts) {
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
        #[serde(default, deserialize_with = "de_explain")]
        explain: Option<ExplainMode>,
    }
    let r: Req = serde_json::from_str(body).unwrap_or(Req { messages: vec![], prompt: String::new(), system: String::new(), max_tokens: None, explain: None });
    let max_tokens = r.max_tokens.unwrap_or_else(|| tg.default_max_tokens()).clamp(1, 16384);
    let bv: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let tools = crate::tools::parse_tools(&bv);
    // Build the prompt exactly as the non-streaming handler does, so streamed and buffered responses are identical:
    // a tool preamble + top-level system are merged ahead of render_chat (which also renders prior tool round-trips).
    let (prompt, add_special) = if route == "/v1/completions" {
        (r.prompt.clone(), true)
    } else {
        let mut sys_extra = String::new();
        if !tools.is_empty() {
            sys_extra.push_str(&crate::tools::preamble(&tools));
        }
        if !r.system.is_empty() {
            if !sys_extra.is_empty() {
                sys_extra.push_str("\n\n");
            }
            sys_extra.push_str(&r.system);
        }
        (render_chat(if sys_extra.is_empty() { None } else { Some(&sys_extra) }, &r.messages), false)
    };
    // Completions don't carry tools in the OpenAI schema; tools only apply to the chat/messages routes.
    let want_tools = route != "/v1/completions" && !tools.is_empty() && !crate::tools::choice_none(&bv);

    if want_tools {
        // Buffered tool-aware path: generate the whole reply, (optionally) log the explanation, parse calls, emit SSE.
        let g = tg.gen(lm, &prompt, max_tokens, add_special, &mut |_| {});
        if let Some(mode) = r.explain {
            log_explanation(lm, tg, &g.prompt_ids, &g.gen_ids, explain, mode);
        }
        let calls = crate::tools::parse_calls(&g.text);
        let id = now();
        let frames = if route == "/v1/messages" {
            anthropic_tool_frames(arch, id, &g.text, &calls, g.hit_eos)
        } else {
            openai_tool_frames(arch, id, &g.text, &calls, g.completion_tokens, g.hit_eos)
        };
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap();
        let len = frames.len();
        let resp = tiny_http::Response::new(200.into(), vec![header], std::io::Cursor::new(frames), Some(len), None);
        let _ = req.respond(resp);
        return;
    }

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
            let g = tg.gen(lm, &prompt, max_tokens, add_special, &mut on_text);
            let _ = tx.send(sse_close(&route, &arch, id));
            if let Some(mode) = r.explain {
                // print the full explain trace (one frame per generated token) to the server console post-stream
                log_explanation(lm, tg, &g.prompt_ids, &g.gen_ids, explain, mode);
            }
            // tx dropped here -> channel closes -> reader EOFs
        });
        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap();
        let resp = tiny_http::Response::new(200.into(), vec![header], SseReader { rx, buf: Vec::new(), pos: 0 }, None, None);
        let _ = req.respond(resp);
    });
}

/// OpenAI streaming tool-call frames: a `role` chunk, then either the answer text (model didn't call a tool) or one
/// `tool_calls` delta per parsed call (each carries `index`, `id`, `type`, and the full `function.name`+`arguments`
/// string in a single fragment — the AI SDK assembles them by `index`), then a terminal `finish_reason` chunk + `[DONE]`.
#[cfg(feature = "api")]
fn openai_tool_frames(arch: &str, id: u64, text: &str, calls: &[crate::tools::ToolCall], ct: usize, eos: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut push = |delta: serde_json::Value, finish: Option<&str>| {
        let j = serde_json::json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","model":arch,
            "choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
        out.extend_from_slice(format!("data: {j}\n\n").as_bytes());
    };
    push(serde_json::json!({"role":"assistant"}), None);
    if calls.is_empty() {
        if !text.is_empty() {
            push(serde_json::json!({"content": text}), None);
        }
        push(serde_json::json!({}), Some(if eos { "stop" } else { "length" }));
    } else {
        let lead = crate::tools::leading_text(text);
        if !lead.is_empty() {
            push(serde_json::json!({"content": lead}), None);
        }
        for (i, c) in calls.iter().enumerate() {
            let args = serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".into());
            push(serde_json::json!({"tool_calls":[{"index":i,"id":format!("call_{id}_{i}"),"type":"function",
                "function":{"name":c.name,"arguments":args}}]}), None);
        }
        push(serde_json::json!({}), Some("tool_calls"));
    }
    // final usage-only chunk (clients that set stream_options.include_usage read it; others ignore the empty choices)
    let usage = serde_json::json!({"id":format!("chatcmpl-{id}"),"object":"chat.completion.chunk","model":arch,
        "choices":[],"usage":{"completion_tokens":ct}});
    out.extend_from_slice(format!("data: {usage}\n\n").as_bytes());
    out.extend_from_slice(b"data: [DONE]\n\n");
    out
}

/// Anthropic streaming tool-use frames: `message_start`, then a text content block (if the model produced leading prose
/// or no call at all) and/or one `tool_use` content block per call — each a `content_block_start` carrying the tool
/// name, an `input_json_delta` with the full arguments JSON as `partial_json`, and a `content_block_stop` — closed by
/// `message_delta` (`stop_reason`) + `message_stop`.
#[cfg(feature = "api")]
fn anthropic_tool_frames(arch: &str, id: u64, text: &str, calls: &[crate::tools::ToolCall], eos: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut emit = |event: &str, data: serde_json::Value| {
        out.extend_from_slice(format!("event: {event}\ndata: {data}\n\n").as_bytes());
    };
    emit("message_start", serde_json::json!({"type":"message_start","message":{"id":format!("msg_{id}"),
        "type":"message","role":"assistant","model":arch,"content":[],"stop_reason":serde_json::Value::Null,
        "usage":{"input_tokens":0,"output_tokens":0}}}));
    let mut idx = 0usize;
    let text_block = |out_emit: &mut dyn FnMut(&str, serde_json::Value), idx: usize, body: &str| {
        out_emit("content_block_start", serde_json::json!({"type":"content_block_start","index":idx,
            "content_block":{"type":"text","text":""}}));
        out_emit("content_block_delta", serde_json::json!({"type":"content_block_delta","index":idx,
            "delta":{"type":"text_delta","text":body}}));
        out_emit("content_block_stop", serde_json::json!({"type":"content_block_stop","index":idx}));
    };
    if calls.is_empty() {
        text_block(&mut emit, idx, text);
        emit("message_delta", serde_json::json!({"type":"message_delta",
            "delta":{"stop_reason": if eos {"end_turn"} else {"max_tokens"}},"usage":{"output_tokens":0}}));
        emit("message_stop", serde_json::json!({"type":"message_stop"}));
        return out;
    }
    let lead = crate::tools::leading_text(text);
    if !lead.is_empty() {
        text_block(&mut emit, idx, &lead);
        idx += 1;
    }
    for (i, c) in calls.iter().enumerate() {
        emit("content_block_start", serde_json::json!({"type":"content_block_start","index":idx,
            "content_block":{"type":"tool_use","id":format!("toolu_{id}_{i}"),"name":c.name,"input":{}}}));
        let args = serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".into());
        emit("content_block_delta", serde_json::json!({"type":"content_block_delta","index":idx,
            "delta":{"type":"input_json_delta","partial_json":args}}));
        emit("content_block_stop", serde_json::json!({"type":"content_block_stop","index":idx}));
        idx += 1;
    }
    emit("message_delta", serde_json::json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},
        "usage":{"output_tokens":0}}));
    emit("message_stop", serde_json::json!({"type":"message_stop"}));
    out
}

/// Default head/tail window for the typed explain trace: deep-explain the first & last N generated tokens and elide
/// the middle. Keeps a long reply's explanation readable AND fast — the per-token explain forward (Circuits/All) runs
/// only on the shown frames, so a 200-token reply costs ~2N forwards, not 200. Live-tunable in chat via `/explain tokens N`.
#[cfg(feature = "api")]
const EXPLAIN_HEAD_TAIL: usize = 4;

/// Print the full explain TRACE (one frame per generated token) to the server console (stderr). Used under `--serve`
/// when a request sets `"explain": true` — the streamed/JSON response can't always carry the extension field, so the
/// operator watches the circuits + named features for every forward pass of the reply on the console.
#[cfg(feature = "api")]
fn log_explanation(lm: &dyn Model, tg: &TextGen, prompt_ids: &[i64], gen_ids: &[i64], opts: &ExplainOpts, mode: ExplainMode) {
    if gen_ids.is_empty() {
        return;
    }
    let dec = |id: i64| tg.token_label(id);
    eprintln!("[fieldrun] {}", render_typed_trace(lm, prompt_ids, gen_ids, opts.store.as_ref(), &opts.cand, mode, &dec, 10, EXPLAIN_HEAD_TAIL));
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

fn handle(url: &str, body: &str, lm: &dyn Model, arch: &str, tg: Option<&TextGen>, explain: &ExplainOpts) -> String {
    #[cfg(not(feature = "api"))]
    let _ = (&tg, explain); // only used by the OpenAI/Anthropic routes (feature `api`)
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
            Some(tg) => openai_anthropic(route, body, lm, arch, tg, explain),
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
#[derive(serde::Deserialize, Default)]
struct Msg {
    #[serde(default)]
    role: String,
    // content is a string (OpenAI/most), an array of blocks (Anthropic), or null (an OpenAI assistant turn that is
    // only tool_calls). serde_json::Value absorbs all three.
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Vec<serde_json::Value>, // OpenAI: assistant's prior tool calls
    #[serde(default)]
    tool_call_id: Option<String>, // OpenAI: links a role:"tool" result to its call
    #[serde(default)]
    name: Option<String>, // OpenAI legacy function name on a tool/function message
}

#[cfg(feature = "api")]
impl Msg {
    /// The plain text of this message — the string content, or the concatenated `text` blocks of an array content.
    fn text(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(blocks) => blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

/// The content blocks of an Anthropic-style message (empty if content isn't an array).
#[cfg(feature = "api")]
fn content_blocks(content: &serde_json::Value) -> &[serde_json::Value] {
    content.as_array().map(|a| a.as_slice()).unwrap_or(&[])
}

/// Flatten an Anthropic `tool_result` block's `content` (a string, or an array of text blocks) to text.
#[cfg(feature = "api")]
fn tool_result_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()).or_else(|| b.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Route ONE token (predicted after `ctx`) to a KB rule or to composition — the retrieval half of explain. Returns
/// `(route, rule)`: RETRIEVED (a KB idiom's top-1 == the token, a pure lookup — `rule` names the idiom), SELECTED (the
/// token is in the KB candidate set but isn't the idiom top-1, so composition picked within a retrieved set), COMPOSED
/// (no KB rule covers it — the forge tax). With no `store`, falls back to context-only candidates (recent + induction
/// copy): a covered token is SELECTED, the strict in-context-copy target is RETRIEVED, else COMPOSED.
#[cfg(feature = "api")]
fn token_route(store: Option<&Store>, ctx: &[i64], predicted: i64, cfg: &CandCfg) -> (&'static str, String, Option<i64>) {
    match store {
        Some(s) => {
            let (kb, idiom) = s.predict(ctx);
            if kb == predicted {
                ("RETRIEVED", idiom, None)
            } else if s.candidates(ctx, cfg).contains(&predicted) {
                ("SELECTED", idiom, Some(kb)) // the KB's own top-1 — the pick the model overrode by choosing `predicted`
            } else {
                ("COMPOSED", String::new(), None)
            }
        }
        None => {
            // storeless: induction-only attribution. The strict in-context-copy target (induction) is RETRIEVED.
            let mut ind = Vec::new();
            context_candidates(ctx, 0, 1, &mut ind);
            if ind.first() == Some(&predicted) {
                ("RETRIEVED", "induction".to_string(), None)
            } else {
                let mut c = Vec::new();
                context_candidates(ctx, cfg.recent, cfg.induction, &mut c);
                // the strict induction copy-target (if any) is the context's "top-1" the model passed over.
                if c.contains(&predicted) { ("SELECTED", "context".to_string(), ind.first().copied()) } else { ("COMPOSED", String::new(), None) }
            }
        }
    }
}

/// Format a fired KB rule as a human line: the production rule made legible — its key (LHS) → ranked successors (RHS)
/// with the picked token marked, or, for induction, the copy-source position. `dec` decodes ids to display strings.
#[cfg(feature = "api")]
fn fmt_rule(r: &RuleHit, predicted: i64, dec: &dyn Fn(i64) -> String) -> String {
    let key = r.key.iter().map(|&t| dec(t)).collect::<Vec<_>>().join(" ");
    if let Some(src) = r.source {
        return format!("     rule  {}: copies {} from position {src} (tail [{key}] recurs there)", r.idiom, dec(predicted));
    }
    let succ = r
        .successors
        .iter()
        .take(6)
        .map(|&t| if t == predicted { format!("{} ◀", dec(t)) } else { dec(t) })
        .collect::<Vec<_>>()
        .join(", ");
    let rank = r.rank.map(|rk| format!("  picked rank {rk}/{}", r.successors.len())).unwrap_or_default();
    format!("     rule  {} [{key}] → {{{succ}}}{rank}", r.idiom)
}

/// Format the candidate-set route detail. A SELECTED token that no single named rule explains came from the KB's broader
/// plausible set (recent context / closed-class / unigram floor) — `rule_for` found no n-gram/induction/grammar rule for
/// it. Show that bounded, order-stable set (highest-value sources first) with the picked token marked and its position,
/// so the frame says *what the model chose within* instead of a bare "candidate-set". Store-less: the context candidates.
#[cfg(feature = "api")]
fn fmt_candidate_set(store: Option<&Store>, ctx: &[i64], predicted: i64, cfg: &CandCfg, dec: &dyn Fn(i64) -> String) -> String {
    let cands = match store {
        Some(s) => s.candidates(ctx, cfg),
        None => {
            let mut c = Vec::new();
            context_candidates(ctx, cfg.recent, cfg.induction, &mut c);
            c
        }
    };
    let total = cands.len();
    let at = match cands.iter().position(|&t| t == predicted) {
        Some(i) => format!("candidate #{} of {total}", i + 1),
        None => format!("{total} candidates"),
    };
    const SHOW: usize = 12;
    let preview = cands.iter().take(SHOW).map(|&t| if t == predicted { format!("{} ◀", dec(t)) } else { dec(t) }).collect::<Vec<_>>().join(", ");
    let more = if total > SHOW { format!(", … +{}", total - SHOW) } else { String::new() };
    format!("     candidate-set: model chose {} ({at}) — KB set {{{preview}{more}}}", dec(predicted))
}

/// Render the TYPED explain trace (the `route`/`circuits`/`all` levels). Every token gets a one-line route (RETRIEVED /
/// SELECTED / COMPOSED, free — no forward); `Circuits` additionally re-runs the faithful forward and shows the DLA
/// circuit breakdown ONLY on COMPOSED tokens (the attribution drives the verbosity — you pay the explain-forward
/// exactly where the model composed); `All` shows the circuit breakdown for every token. `cand` bounds the KB set used
/// to decide SELECTED-vs-COMPOSED. Returns the rendered trace; cheap modes never call `lm.explain`.
#[cfg(feature = "api")]
#[allow(clippy::too_many_arguments)]
fn render_typed_trace(
    lm: &dyn Model,
    prompt_ids: &[i64],
    gen_ids: &[i64],
    store: Option<&Store>,
    cand: &CandCfg,
    mode: ExplainMode,
    dec: &dyn Fn(i64) -> String,
    max_ctx: usize,
    head_tail: usize, // deep-explain only the first & last `head_tail` tokens; elide the middle (0 = all). Bounds
                      // BOTH the reader's output AND the cost — the per-token explain forward runs only on shown frames.
) -> String {
    let legend = match mode {
        ExplainMode::Route => "route only (free — no extra forward)",
        ExplainMode::Circuits => "route + DLA circuits on COMPOSED tokens (the forge tax)",
        ExplainMode::All => "route + DLA circuits on every token",
        ExplainMode::Logic => "route + logic-export provenance (per-block contributions + the decode margin; LOGIC_EXPORT.md)",
    };
    let n = gen_ids.len();
    // head_tail bounds ONLY the expensive + verbose CIRCUIT breakdown (one faithful forward each) — the route + rule
    // lines are cheap (KB lookups, no forward) and ARE the per-token attribution the trace exists for, so they show
    // for EVERY token. So `circuits N` deep-dives only the first/last N; the routing+rules stay complete.
    let circ_bounded = head_tail > 0 && n > 2 * head_tail && !matches!(mode, ExplainMode::Route);
    let show_circuits = |k: usize| !circ_bounded || k < head_tail || k >= n - head_tail;
    let note = if circ_bounded { format!("; DLA circuits on first/last {head_tail} only — /explain tokens 0 for all") } else { String::new() };
    let mut out = vec![
        format!("explain trace [{legend}{note}] — one frame per token, routed to a KB rule or to composition:"),
        "  RETRIEVED = the knowledge-base rule's top-1 IS the token (pure lookup)".to_string(),
        "  SELECTED  = the token was a KB candidate but not the rule's top-1 — the model chose it over the rule's pick".to_string(),
        "  COMPOSED  = no KB rule covers it; the network computed it (the \"forge tax\")".to_string(),
    ];
    let (mut retr, mut sel, mut comp) = (0usize, 0usize, 0usize);
    let mut ctx = prompt_ids.to_vec();
    for (k, &t) in gen_ids.iter().enumerate() {
        let (route, idiom, kb_top1) = token_route(store, &ctx, t, cand);
        match route { "RETRIEVED" => retr += 1, "SELECTED" => sel += 1, _ => comp += 1 }
        // route + rule line — ALWAYS (cheap). The rule that explains THIS token (key → successors); its idiom names the
        // route when present. RETRIEVED/SELECTED get a production-rule line (needs --store); COMPOSED has no rule.
        // n-gram/grammar rules need the KB (--store); induction is pure context, so it shows even store-less.
        let rule = if route != "COMPOSED" { store.and_then(|s| s.rule_for(&ctx, t)).or_else(|| induction_rule(&ctx, t)) } else { None };
        let via_name = match (&rule, route) {
            (Some(rh), _) => rh.idiom.clone(),
            (None, "SELECTED") => "candidate-set".to_string(),
            _ => idiom,
        };
        let via = if via_name.is_empty() { String::new() } else { format!(" via {via_name}") };
        // For SELECTED, name the KB's own top-1 — the prediction the model passed over — so the route is legible
        // (otherwise SELECTED reads just like RETRIEVED, since both can show an induction copy line below).
        let overrode = match kb_top1 {
            Some(top1) if route == "SELECTED" && top1 != t => format!("  (KB top-1 was {} — model overrode)", dec(top1)),
            _ => String::new(),
        };
        out.push(format!("\n┌─ #{k} {} ← {route}{via}{overrode}", dec(t)));
        if let Some(rh) = &rule {
            out.push(fmt_rule(rh, t, dec)); // the retrieval half: the production rule made legible
        } else if route == "SELECTED" {
            // candidate-set route: no single named rule explains the token, so show the bounded KB set it was picked from.
            out.push(fmt_candidate_set(store, &ctx, t, cand, dec));
        }
        // circuit breakdown — the expensive half — only on the first/last head_tail tokens.
        let want_circuits = matches!(mode, ExplainMode::All) || (matches!(mode, ExplainMode::Circuits) && route == "COMPOSED");
        if want_circuits && show_circuits(k) {
            if let Some(ex) = lm.explain(&ctx) {
                out.push(crate::explain::render(&ex, dec, max_ctx)); // the composition half (DLA heads/neurons)
            }
        }
        // logic-export provenance — the same data the semiring-Datalog emitter renders, as a per-token line (the
        // composition shown as per-block contributions instead of per-circuit DLA). One forward/token, so head/tail-bounded.
        if matches!(mode, ExplainMode::Logic) && show_circuits(k) {
            if let Some(prov) = crate::logic::build(lm, &ctx, store, cand, 32) {
                out.push(format!("│  ⟨logic⟩ {}", crate::logic::explain_line(&prov, dec)));
            }
        }
        ctx.push(t);
    }
    let nn = n.max(1);
    out.push(format!(
        "\nrouted {n} tokens — RETRIEVED {retr} ({:.0}%) · SELECTED {sel} ({:.0}%) · COMPOSED {comp} ({:.0}%)",
        100.0 * retr as f64 / nn as f64, 100.0 * sel as f64 / nn as f64, 100.0 * comp as f64 / nn as f64
    ));
    out.join("\n")
}

/// The KB attribution + circuit context carried by the server for the typed explain (`--store` + the candidate-set
/// config). Held for the server's lifetime and passed to the request handlers. Not feature-gated — `serve`/`handle`
/// compile in the token-id-only build too, so they thread this even when the text-explain handlers are absent.
pub struct ExplainOpts {
    pub store: Option<Store>,
    pub cand: CandCfg,
}

/// Deserialize the request's `explain` field as EITHER a bool (`true` → the free `route` level) OR a level name
/// (`"route"`/`"circuits"`/`"all"`; `"off"`/`false` → none). Unknown strings fall back to `route` (explain on).
#[cfg(feature = "api")]
fn de_explain<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<ExplainMode>, D::Error> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum E {
        Bool(bool),
        Str(String),
    }
    Ok(match Option::<E>::deserialize(d)? {
        None | Some(E::Bool(false)) => None,
        Some(E::Bool(true)) => Some(ExplainMode::Route),
        Some(E::Str(s)) => match s.to_ascii_lowercase().as_str() {
            "off" | "false" | "none" | "no" | "0" => None,
            other => ExplainMode::parse(other).or(Some(ExplainMode::Route)),
        },
    })
}

/// Typed structured explain for the API response (`fieldrun_explanation`). One object per generated token —
/// `{i, token, route, rule, explanation}` — where `route` is RETRIEVED/SELECTED/COMPOSED and `explanation` is the full
/// DLA `Explanation` (only on COMPOSED tokens for `Circuits`, on every token for `All`, and always null for `Route`).
/// Mirrors the console trace's attribution-driven verbosity: the expensive per-token forward runs only where the mode
/// asks for circuits.
#[cfg(feature = "api")]
fn typed_explanation_json(lm: &dyn Model, prompt_ids: &[i64], gen_ids: &[i64], opts: &ExplainOpts, mode: ExplainMode) -> serde_json::Value {
    let mut ctx = prompt_ids.to_vec();
    let mut arr = Vec::with_capacity(gen_ids.len());
    for &t in gen_ids {
        let (route, idiom, kb_top1) = token_route(opts.store.as_ref(), &ctx, t, &opts.cand);
        let want_circuits = matches!(mode, ExplainMode::All) || (matches!(mode, ExplainMode::Circuits) && route == "COMPOSED");
        let explanation = if want_circuits {
            lm.explain(&ctx).and_then(|ex| serde_json::to_value(ex).ok()).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };
        // the fired production rule (key → ranked successors, picked rank, induction source), when one explains the token.
        let rule = if route != "COMPOSED" { opts.store.as_ref().and_then(|s| s.rule_for(&ctx, t)) } else { None };
        let rule_name = match (&rule, route) {
            (Some(rh), _) => rh.idiom.clone(),
            (None, "SELECTED") => "candidate-set".to_string(),
            _ => idiom,
        };
        // candidate-set position for a SELECTED token no named rule covers — its place in the KB's plausible set.
        let candidate_set = if route == "SELECTED" && rule.is_none() {
            let cands = match opts.store.as_ref() {
                Some(s) => s.candidates(&ctx, &opts.cand),
                None => {
                    let mut c = Vec::new();
                    context_candidates(&ctx, opts.cand.recent, opts.cand.induction, &mut c);
                    c
                }
            };
            serde_json::json!({ "position": cands.iter().position(|&x| x == t).map(|i| i + 1), "total": cands.len() })
        } else {
            serde_json::Value::Null
        };
        let rule_detail = rule.and_then(|rh| serde_json::to_value(rh).ok()).unwrap_or(serde_json::Value::Null);
        // logic-export provenance (the semiring-Datalog view): per-block contributions + the decode margin. Only on Logic.
        let logic = if matches!(mode, ExplainMode::Logic) {
            crate::logic::build(lm, &ctx, opts.store.as_ref(), &opts.cand, 32).map(|p| {
                let mut top: Vec<(&str, f32)> = p.blocks.iter().map(|(n, ws)| (n.as_str(), ws.first().copied().unwrap_or(0.0))).collect();
                top.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                serde_json::json!({
                    "route": crate::logic::route_name(p.route), "margin": p.margin, "runner_up": p.runner_up,
                    "candidates": p.candidates.len(),
                    "top_blocks": top.iter().take(5).map(|(n, w)| serde_json::json!({ "block": n, "w": w })).collect::<Vec<_>>(),
                })
            }).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };
        arr.push(serde_json::json!({ "i": arr.len(), "token": t, "route": route, "rule": rule_name, "rule_detail": rule_detail, "kb_top1": kb_top1, "candidate_set": candidate_set, "explanation": explanation, "logic": logic }));
        ctx.push(t);
    }
    serde_json::Value::Array(arr)
}

#[cfg(feature = "api")]
fn openai_anthropic(route: &str, body: &str, lm: &dyn Model, arch: &str, tg: &TextGen, explain: &ExplainOpts) -> String {
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
        // fieldrun extension: `true`/`"route"`/`"circuits"`/`"all"` attach the typed explanation under
        // "fieldrun_explanation" (and log the trace to the server console).
        #[serde(default, deserialize_with = "de_explain")]
        explain: Option<ExplainMode>,
    }
    let r: Req = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return err(&format!("bad JSON body: {e}")),
    };
    let max_tokens = r.max_tokens.unwrap_or_else(|| tg.default_max_tokens()).clamp(1, 16384);

    if route == "/v1/completions" {
        // OpenAI text completion — raw prompt, with the tokenizer's special tokens
        let g = tg.gen(lm, &r.prompt, max_tokens, true, &mut |_| {});
        let mut v = serde_json::json!({
            "id": format!("cmpl-{}", now()), "object": "text_completion", "created": now(), "model": arch,
            "choices": [{ "text": g.text, "index": 0, "finish_reason": if g.hit_eos {"stop"} else {"length"} }],
            "usage": { "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens, "total_tokens": g.prompt_tokens + g.completion_tokens }
        });
        if let Some(mode) = r.explain {
            log_explanation(lm, tg, &g.prompt_ids, &g.gen_ids, explain, mode);
            v["fieldrun_explanation"] = typed_explanation_json(lm, &g.prompt_ids, &g.gen_ids, explain, mode);
        }
        return v.to_string();
    }

    // chat (OpenAI /v1/chat/completions and Anthropic /v1/messages share the message shape). Tools (either request
    // shape) are declared via a system preamble and parsed back out of the output; the round-trip (prior tool_calls +
    // results in `messages`) is rendered by render_chat.
    let bv: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let tools = crate::tools::parse_tools(&bv);
    let mut sys_extra = String::new();
    if !tools.is_empty() {
        sys_extra.push_str(&crate::tools::preamble(&tools));
    }
    if !r.system.is_empty() {
        if !sys_extra.is_empty() {
            sys_extra.push_str("\n\n");
        }
        sys_extra.push_str(&r.system);
    }
    let prompt = render_chat(if sys_extra.is_empty() { None } else { Some(&sys_extra) }, &r.messages);
    let g = tg.gen(lm, &prompt, max_tokens, false, &mut |_| {});
    if let Some(mode) = r.explain {
        log_explanation(lm, tg, &g.prompt_ids, &g.gen_ids, explain, mode); // also print the full explain trace to the server console under --serve
    }
    let explanation = r.explain.map(|mode| typed_explanation_json(lm, &g.prompt_ids, &g.gen_ids, explain, mode));
    let GenOut { text, prompt_tokens: pt, completion_tokens: ct, hit_eos: eos, .. } = g;
    let attach = |mut v: serde_json::Value| -> String {
        if let Some(ex) = explanation {
            v["fieldrun_explanation"] = ex;
        }
        v.to_string()
    };
    let calls = if tools.is_empty() || crate::tools::choice_none(&bv) {
        Vec::new()
    } else {
        crate::tools::parse_calls(&text)
    };

    if !calls.is_empty() {
        let lead = crate::tools::leading_text(&text);
        if route == "/v1/messages" {
            // Anthropic: optional leading text block + one tool_use block per call
            let mut content = Vec::new();
            if !lead.is_empty() {
                content.push(serde_json::json!({ "type": "text", "text": lead }));
            }
            for (i, c) in calls.iter().enumerate() {
                content.push(serde_json::json!({ "type": "tool_use", "id": format!("toolu_{}_{i}", now()),
                    "name": c.name, "input": c.arguments }));
            }
            return attach(serde_json::json!({
                "id": format!("msg_{}", now()), "type": "message", "role": "assistant", "model": arch,
                "content": content, "stop_reason": "tool_use", "usage": { "input_tokens": pt, "output_tokens": ct }
            }));
        }
        // OpenAI: message.tool_calls (arguments is a JSON *string*) + finish_reason "tool_calls"
        let tcs: Vec<serde_json::Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| serde_json::json!({ "id": format!("call_{}_{i}", now()), "type": "function",
                "function": { "name": c.name, "arguments": serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".into()) } }))
            .collect();
        let content = if lead.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(lead) };
        return attach(serde_json::json!({
            "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": arch,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": content, "tool_calls": tcs },
                          "finish_reason": "tool_calls" }],
            "usage": { "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct }
        }));
    }

    if route == "/v1/messages" {
        // Anthropic Messages API
        attach(serde_json::json!({
            "id": format!("msg_{}", now()), "type": "message", "role": "assistant", "model": arch,
            "content": [{ "type": "text", "text": text }],
            "stop_reason": if eos {"end_turn"} else {"max_tokens"},
            "usage": { "input_tokens": pt, "output_tokens": ct }
        }))
    } else {
        // OpenAI chat completion
        attach(serde_json::json!({
            "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": arch,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text },
                          "finish_reason": if eos {"stop"} else {"length"} }],
            "usage": { "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct }
        }))
    }
}

/// Build a ChatML prompt from a message list, rendering OpenAI/Anthropic tool calls + results so the model can do the
/// tool round-trip. `system_extra` (tool preamble + any top-level system) is merged with role:"system" messages, and
/// the final `<|im_start|>assistant\n` opens the model's turn.
#[cfg(feature = "api")]
fn render_chat(system_extra: Option<&str>, msgs: &[Msg]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let mut sys: Vec<String> = Vec::new();
    if let Some(e) = system_extra {
        if !e.is_empty() {
            sys.push(e.to_string());
        }
    }
    for m in msgs.iter().filter(|m| m.role == "system") {
        let t = m.text();
        if !t.is_empty() {
            sys.push(t);
        }
    }
    if !sys.is_empty() {
        let _ = write!(s, "<|im_start|>system\n{}<|im_end|>\n", sys.join("\n\n"));
    }
    for m in msgs {
        match m.role.as_str() {
            "system" => {}
            "tool" => {
                // OpenAI tool result
                let _ = write!(s, "<|im_start|>tool\n<tool_response>\n{}\n</tool_response><|im_end|>\n", m.text());
            }
            "assistant" => {
                let mut body = m.text();
                for tc in &m.tool_calls {
                    if let Some(f) = tc.get("function") {
                        let name = f.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let raw = f.get("arguments").cloned().unwrap_or_else(|| serde_json::json!("{}"));
                        let argv = match &raw {
                            serde_json::Value::String(x) => serde_json::from_str(x).unwrap_or(raw.clone()),
                            v => v.clone(),
                        };
                        let _ = write!(body, "\n<tool_call>\n{}\n</tool_call>", serde_json::json!({"name": name, "arguments": argv}));
                    }
                }
                for blk in content_blocks(&m.content) {
                    if blk.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        let name = blk.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let input = blk.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
                        let _ = write!(body, "\n<tool_call>\n{}\n</tool_call>", serde_json::json!({"name": name, "arguments": input}));
                    }
                }
                let _ = write!(s, "<|im_start|>assistant\n{body}<|im_end|>\n");
            }
            _ => {
                // user — Anthropic carries tool_result blocks inside a user message; render them as tool turns first
                let mut had_result = false;
                for blk in content_blocks(&m.content) {
                    if blk.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                        had_result = true;
                        let _ = write!(s, "<|im_start|>tool\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                            tool_result_text(blk.get("content")));
                    }
                }
                let t = m.text();
                if !t.is_empty() || !had_result {
                    let _ = write!(s, "<|im_start|>user\n{t}<|im_end|>\n");
                }
            }
        }
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

/// rustyline helper that Tab-completes the REPL slash commands (and their sub-arguments). Line editing, history, and
/// the other Helper traits are no-ops/defaults.
#[cfg(feature = "api")]
struct SlashHelper;

#[cfg(feature = "api")]
impl rustyline::completion::Completer for SlashHelper {
    type Candidate = rustyline::completion::Pair;
    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> rustyline::Result<(usize, Vec<rustyline::completion::Pair>)> {
        let upto = &line[..pos];
        if !upto.starts_with('/') {
            return Ok((pos, Vec::new()));
        }
        let pair = |s: &str| rustyline::completion::Pair { display: s.to_string(), replacement: s.to_string() };
        // sub-argument completion (after the command + a space), else complete the command name itself
        if let Some(sp) = upto.find(' ') {
            let cmd = &upto[..sp];
            let arg_start = upto.rfind(' ').unwrap() + 1;
            let frag = &upto[arg_start..];
            let subs: &[&str] = match cmd {
                "/explain" => &["on", "off", "context", "all"],
                "/format" => &["on", "off"],
                _ => &[],
            };
            Ok((arg_start, subs.iter().filter(|s| s.starts_with(frag)).map(|s| pair(s)).collect()))
        } else {
            const CMDS: &[&str] = &["/exit", "/quit", "/reset", "/clear", "/explain", "/export-logic", "/format", "/raw", "/help"];
            Ok((0, CMDS.iter().filter(|c| c.starts_with(upto)).map(|c| pair(c)).collect()))
        }
    }
}

#[cfg(feature = "api")]
impl rustyline::hint::Hinter for SlashHelper {
    type Hint = String;
}
#[cfg(feature = "api")]
impl rustyline::highlight::Highlighter for SlashHelper {}
#[cfg(feature = "api")]
impl rustyline::validate::Validator for SlashHelper {}
#[cfg(feature = "api")]
impl rustyline::Helper for SlashHelper {}

/// Interactive REPL chat over the bundled tokenizer (the `--chat` mode). Maintains conversation history; ChatML prompt.
/// `explain` starts per-reply explanation output on (toggle live with `/explain`); `raw` disables Markdown rendering;
/// `arch` names the model for messages.
#[cfg(feature = "api")]
#[allow(clippy::too_many_arguments)]
pub fn chat(lm: Box<dyn Model>, tg: TextGen, max_tokens: usize, mut explain: Option<ExplainMode>, store: Option<Store>, cand: CandCfg, raw: bool, arch: &str, bucket: Option<crate::bucketing::BucketOpts>) {
    use std::io::{IsTerminal, Write};
    // render Markdown→ANSI only when writing to a real terminal (piped output stays raw, for scripts) and not --raw.
    let mut fmt = std::io::stdout().is_terminal() && !raw;
    eprintln!("[fieldrun] chat — type a message; /help for commands, Tab completes them, ↑/↓ history, /exit or Ctrl-D \
               to quit. (greedy, max_tokens={max_tokens}; generic ChatML template)");
    eprintln!("[fieldrun] markdown rendering {} (/format to toggle){}", if fmt { "ON" } else { "OFF" },
              match explain {
                  Some(m) => format!("; explain {m:?} (/explain off|route|circuits|all)"),
                  None => String::new(),
              });
    if explain.is_some() && store.is_none() {
        eprintln!("[fieldrun] explain: no --store loaded → routing is induction-only (RETRIEVED/COMPOSED); pass --store <store.json> for full KB-rule attribution.");
    }
    if !tg.knows_chatml() {
        eprintln!("[fieldrun] heads-up: this tokenizer has no ChatML template (<|im_start|>) — it looks like a BASE \
                   model (e.g. GPT-2), not an instruct model, so chat will just CONTINUE your text and won't stop \
                   so chat runs as a text-COMPLETION REPL — type text and it continues it, stopping at the model's EOS \
                   (it won't follow instructions like an instruct model).");
    }
    let chatml = tg.knows_chatml(); // instruct model → ChatML template + history; base model → raw completion
    let mut history: Vec<(String, String)> = Vec::new();
    let mut explain_ctx: usize = 10; // how many trailing context tokens explain prints (0 = all); /explain context N
    let mut explain_tk: usize = EXPLAIN_HEAD_TAIL; // deep-explain only the first & last N reply tokens (0 = all); /explain tokens N
    let mut logic_seq: usize = 0; // auto-numbers default `/export-logic` filenames (logic-001.dl, …) when no path is given
    // --bucket: incremental density-minimization expert clustering over the session corpus (each reply token's atom),
    // reported after every reply. session_sig/pred align with the ingested atoms (prev token + decode), for /bucket dump.
    let mut session = crate::bucketing::CorpusBuckets::new();
    let (mut session_sig, mut session_pred): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
    let mut bucket_on = bucket.is_some();
    let mut bkt = bucket.unwrap_or(crate::bucketing::BucketOpts { k: 4, experts: 8 });
    // rustyline gives line editing, history (↑/↓), and Tab-completion of slash commands. It only owns the terminal
    // during readline; the generation/streaming below runs in normal mode exactly as before.
    let cfg = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .auto_add_history(true)
        .build();
    let mut rl = match rustyline::Editor::<SlashHelper, rustyline::history::MemHistory>::with_history(cfg, rustyline::history::MemHistory::new()) {
        Ok(mut e) => {
            e.set_helper(Some(SlashHelper));
            e
        }
        Err(e) => {
            eprintln!("[fieldrun] couldn't start the line editor: {e}");
            return;
        }
    };
    loop {
        let line = match rl.readline("\nyou> ") {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => continue, // Ctrl-C cancels the current line
            Err(rustyline::error::ReadlineError::Eof) => {
                eprintln!("[fieldrun] bye");
                break;
            }
            Err(e) => {
                eprintln!("[fieldrun] input error: {e}");
                break;
            }
        };
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        // slash commands
        if let Some(cmd) = user.strip_prefix('/') {
            let mut parts = cmd.split_whitespace();
            match parts.next().unwrap_or("") {
                "exit" | "quit" | "q" => {
                    eprintln!("[fieldrun] bye");
                    break;
                }
                "reset" | "clear" => {
                    history.clear();
                    eprintln!("[fieldrun] (conversation reset)");
                }
                "explain" => match parts.next() {
                    Some("off") | Some("0") | Some("false") => {
                        explain = None;
                        eprintln!("[fieldrun] explain OFF");
                    }
                    Some(m) if ExplainMode::parse(m).is_some() => {
                        explain = ExplainMode::parse(m);
                        eprintln!("[fieldrun] explain {:?}  (route=free · circuits=DLA on composed · all=DLA on every token)", explain.unwrap());
                    }
                    None => {
                        explain = Some(ExplainMode::Route);
                        eprintln!("[fieldrun] explain Route (free); /explain circuits|all for the composition breakdown");
                    }
                    Some("context") | Some("ctx") => {
                        explain_ctx = match parts.next() {
                            Some("all") | Some("full") | Some("0") => 0,
                            Some(n) => n.parse().unwrap_or(explain_ctx),
                            None => explain_ctx,
                        };
                        eprintln!("[fieldrun] explain context window = {}",
                                  if explain_ctx == 0 { "all".to_string() } else { explain_ctx.to_string() });
                    }
                    Some("tokens") | Some("tok") => {
                        explain_tk = match parts.next() {
                            Some("all") | Some("full") | Some("0") => 0,
                            Some(n) => n.parse().unwrap_or(explain_tk),
                            None => explain_tk,
                        };
                        eprintln!("[fieldrun] explain shows {} (deep-explain first/last N reply tokens, elide the middle)",
                                  if explain_tk == 0 { "ALL tokens".to_string() } else { format!("first/last {explain_tk}") });
                    }
                    Some(other) => {
                        eprintln!("[fieldrun] /explain {other}? use: off | route (free) | circuits (DLA on composed) | all | context N | tokens N");
                    }
                },
                "format" | "md" | "markdown" => {
                    fmt = match parts.next() {
                        Some("on") | Some("1") | Some("true") => true,
                        Some("off") | Some("0") | Some("false") => false,
                        _ => !fmt,
                    };
                    eprintln!("[fieldrun] markdown rendering {}", if fmt { "ON" } else { "OFF" });
                }
                "raw" => {
                    fmt = false;
                    eprintln!("[fieldrun] markdown rendering OFF (raw)");
                }
                // /export-logic [file.dl] <prompt> — generate the WHOLE reply to <prompt> (greedy, to EOS / max_tokens) in the
                // SAME chat context a real turn uses, and emit it as a semiring-Datalog decode TRACE: one runnable .dl per
                // generated token (prefix.000.dl, prefix.001.dl, …). gen_ids IS the trajectory; we walk it exactly like the
                // explain trace, so each file is the faithful provenance of that one decision (reuses logic::build). Combine
                // the parts into ONE step-indexed program with `fieldrun stitch`. Rope archs only (needs residual_decomp).
                "export-logic" | "logic-export" | "export_logic" => {
                    let toks: Vec<&str> = parts.collect();
                    // first token is the file/prefix iff it ends in .dl; otherwise auto-number and treat all of it as the prompt.
                    let (path, prompt) = match toks.first() {
                        Some(f) if f.ends_with(".dl") => (f.to_string(), toks[1..].join(" ")),
                        _ => {
                            logic_seq += 1;
                            (format!("logic-{logic_seq:03}.dl"), toks.join(" "))
                        }
                    };
                    if prompt.trim().is_empty() {
                        eprintln!("[fieldrun] /export-logic [file.dl] <prompt> — generate the WHOLE reply to <prompt> and emit it \
                                   as a decode TRACE (one .dl per token: prefix.000.dl, …). Stitch into one program: \
                                   `fieldrun stitch prefix.*.dl -o whole.dl`. e.g. /export-logic out.dl The capital of France is");
                    } else {
                        // build the context exactly as a turn would (ChatML + history for instruct; raw + BOS for base), then
                        // greedily generate the whole reply — gen_ids is the trajectory we walk to emit one .dl per step.
                        let (p, add_special) = if chatml {
                            (tg.chat_prompt(None, &history, &prompt), false)
                        } else {
                            (prompt.clone(), true)
                        };
                        let g = tg.gen(lm.as_ref(), &p, max_tokens, add_special, &mut |_| {});
                        let dec = |id: i64| tg.token_label(id);
                        let stem_pfx = path.strip_suffix(".dl").unwrap_or(&path);
                        eprintln!("[fieldrun] /export-logic: reply is {} tokens — exporting one .dl per step…", g.gen_ids.len());
                        // walk the trajectory: provenance at each step, context advancing by the model's own pick.
                        let mut ctx = g.prompt_ids.clone();
                        let (mut written, mut faithful, mut unsupported) = (0usize, 0usize, false);
                        for (k, &t) in g.gen_ids.iter().enumerate() {
                            match crate::logic::build(lm.as_ref(), &ctx, store.as_ref(), &cand, 48) {
                                Some(prov) => {
                                    let dl = crate::logic::emit_dl(&prov, &ctx, &dec);
                                    let pth = format!("{stem_pfx}.{k:03}.dl");
                                    if std::fs::write(&pth, &dl).is_ok() {
                                        written += 1;
                                        if dl.contains("✓ FAITHFUL") { faithful += 1; }
                                    }
                                    eprint!("\r[fieldrun] /export-logic: step {}/{}…", k + 1, g.gen_ids.len());
                                    let _ = std::io::stderr().flush();
                                }
                                None => { unsupported = true; break; }
                            }
                            ctx.push(t);
                        }
                        eprint!("\r\x1b[2K");
                        if unsupported {
                            eprintln!("[fieldrun] /export-logic: arch {arch} has no residual_decomp — logic export is rope-only (Qwen2.5/Llama)");
                        } else if written == 0 {
                            eprintln!("[fieldrun] /export-logic: the model produced no tokens for that prompt");
                        } else {
                            eprintln!("[fieldrun] /export-logic → {stem_pfx}.{{000..{:03}}}.dl  ({written} steps, {faithful} FAITHFUL ✓{}) — stitch into one program: fieldrun stitch {stem_pfx}.*.dl -o {stem_pfx}.whole.dl",
                                      written.saturating_sub(1), if g.hit_eos { "" } else { ", stopped at max_tokens" });
                            eprintln!("[fieldrun] reply: {}", g.text.trim());
                        }
                    }
                }
                // /bucket — incremental expert bucketing over the session corpus (density-minimization atoms). Toggle,
                // tune E/K, reset, show the current clustering, or dump the partition (.json) / Datalog model (.dl).
                "bucket" => match parts.next() {
                    Some("off") | Some("0") | Some("false") => { bucket_on = false; eprintln!("[fieldrun] bucket OFF"); }
                    Some("on") | Some("1") | Some("true") => { bucket_on = true; eprintln!("[fieldrun] bucket ON (K={}, E={}) — clustering reported after each reply", bkt.k, bkt.experts); }
                    Some("experts") | Some("e") => { if let Some(n) = parts.next().and_then(|s| s.parse().ok()) { bkt.experts = n; } eprintln!("[fieldrun] bucket experts E = {}", bkt.experts); }
                    Some("k") => { if let Some(n) = parts.next().and_then(|s| s.parse().ok()) { bkt.k = n; } eprintln!("[fieldrun] bucket competitors K = {}", bkt.k); }
                    Some("reset") => { session = crate::bucketing::CorpusBuckets::new(); session_sig.clear(); session_pred.clear(); eprintln!("[fieldrun] bucket session reset"); }
                    Some("dump") => {
                        let path = parts.next().unwrap_or("session-experts.dl").to_string();
                        if session.n_tokens() == 0 {
                            eprintln!("[fieldrun] bucket: nothing to dump yet (turn it on and send a message)");
                        } else if path.ends_with(".json") {
                            match serde_json::to_string_pretty(&session.partition(bkt.experts)).ok().and_then(|j| std::fs::write(&path, j).ok()) {
                                Some(()) => eprintln!("[fieldrun] bucket: wrote partition JSON → {path}"),
                                None => eprintln!("[fieldrun] bucket: dump failed"),
                            }
                        } else {
                            match std::fs::write(&path, session.emit_datalog(bkt.experts, &session_sig, &session_pred, 0.2)) {
                                Ok(()) => eprintln!("[fieldrun] bucket: wrote Datalog lookup/selection model → {path}  (run: souffle {path} -D-)"),
                                Err(e) => eprintln!("[fieldrun] bucket: cannot write {path}: {e}"),
                            }
                        }
                    }
                    None => {
                        bucket_on = true;
                        if session.n_tokens() == 0 {
                            eprintln!("[fieldrun] bucket ON — no atoms yet; send a message and the clustering reports after the reply");
                        } else {
                            eprintln!("[fieldrun] session expert clustering ({} tokens, K={}, E={}):\n{}", session.n_tokens(), bkt.k, bkt.experts, session.render(bkt.experts));
                        }
                    }
                    Some(other) => eprintln!("[fieldrun] /bucket {other}? use: on | off | experts N | k N | reset | dump [path.dl|.json] | (no arg = show)"),
                },
                "help" => eprintln!("[fieldrun] commands: /exit (or /quit) · /reset (clear history) · \
                                     /explain [on|off] (circuits + features) · /explain context <N|all> · \
                                     /export-logic [file.dl] <prompt> (semiring-Datalog export) · \
                                     /bucket [on|off|experts N|k N|reset|dump path] (incremental expert clustering) · \
                                     /format [on|off] (markdown) · /help"),
                other => eprintln!("[fieldrun] unknown command /{other} — try /help"),
            }
            continue;
        }
        // instruct model → ChatML template + conversation history; base model → raw text completion (add the
        // tokenizer's special tokens, e.g. a BOS) so it just continues the text and stops at the model's EOS.
        let (prompt, add_special) = if chatml {
            (tg.chat_prompt(None, &history, user), false)
        } else {
            (user.to_string(), true)
        };
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
        // Stop the spinner + print the "bot> " prefix only when there's VISIBLE output — for raw that's the first
        // token; for formatted it's the first COMPLETED line (Markdown renders a line at a time). Until then the
        // spinner keeps running, so on a slow model you see "[ thinking Ns ]", never an empty "bot> " sitting there
        // looking like an editable input prompt while it's still generating. `in_code` carries an open ``` fence.
        let mut linebuf = String::new();
        let mut in_code = false;
        let gen_out = tg.gen(lm.as_ref(), &prompt, max_tokens, add_special, &mut |chunk| {
            if !fmt {
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
                return;
            }
            linebuf.push_str(chunk);
            while let Some(nl) = linebuf.find('\n') {
                let line = linebuf[..nl].to_string();
                linebuf.drain(..=nl);
                if !started {
                    started = true;
                    thinking.store(false, std::sync::atomic::Ordering::Relaxed);
                    if let Some(h) = spinner.take() {
                        let _ = h.join();
                    }
                    print!("bot> ");
                }
                println!("{}", crate::mdfmt::render_line(&line, &mut in_code));
            }
            let _ = std::io::stdout().flush();
        });
        // Generation finished. If nothing visible was emitted yet (empty reply, or a formatted reply that never hit a
        // newline), stop the spinner now and print the prefix + whatever remains in the buffer.
        if !started {
            thinking.store(false, std::sync::atomic::Ordering::Relaxed);
            if let Some(h) = spinner.take() {
                let _ = h.join();
            }
            print!("bot> ");
            if fmt && !linebuf.is_empty() {
                println!("{}", crate::mdfmt::render_line(&linebuf, &mut in_code));
            } else {
                println!();
            }
        } else if fmt {
            // flush a trailing partial line (reply with no final newline); otherwise we're already on a fresh line
            if !linebuf.is_empty() {
                println!("{}", crate::mdfmt::render_line(&linebuf, &mut in_code));
            }
        } else {
            println!(); // raw stream had no trailing newline
        }
        if !gen_out.hit_eos {
            // ran into the length cap rather than stopping at EOS — say so, so a truncated reply isn't mistaken for
            // a broken model (reasoning models especially blow past a small cap mid-thought).
            eprintln!("[fieldrun] (stopped at max_tokens={max_tokens} — raise with --max-tokens N for longer replies)");
        }
        // per-reply explain TRACE: the circuits + features behind EVERY forward pass of the reply — one frame per
        // generated token, a debugger-style stack of the model "looping and thinking" (not just the first decision).
        // Decoded via the bundled tokenizer. Off by default (toggle with /explain).
        if let Some(mode) = explain {
            // Route is free (no forward); Circuits/All re-run the faithful forward (only on COMPOSED tokens for Circuits).
            // For a circuit mode on an arch with no explain support, the route still renders; only the breakdown is empty.
            if matches!(mode, ExplainMode::Circuits | ExplainMode::All) && lm.explain(&gen_out.prompt_ids).is_none() {
                eprintln!("[fieldrun] (no circuit explain for arch {arch} — showing route only; /explain route to silence)");
            }
            let dec = |id: i64| tg.token_label(id);
            eprintln!("\n[explain]\n{}", render_typed_trace(lm.as_ref(), &gen_out.prompt_ids, &gen_out.gen_ids, store.as_ref(), &cand, mode, &dec, explain_ctx, explain_tk));
        }
        // --bucket: ingest each reply token's irreducible atom (descend prompt_ids ++ gen_ids[..i]) into the session
        // corpus, then report the running expert clustering. One descent forward per reply token — heavy, opt-in.
        if bucket_on {
            let full: Vec<i64> = gen_out.prompt_ids.iter().chain(&gen_out.gen_ids).copied().collect();
            let start = gen_out.prompt_ids.len();
            if start < full.len() {
                eprint!("[fieldrun] [bucket] analysing {} reply tokens…", full.len() - start);
                let _ = std::io::stderr().flush();
            }
            let mut added = 0usize;
            for i in start..full.len() {
                if let Some(a) = crate::bucketing::atom_at(lm.as_ref(), &full[..i], bkt.k) {
                    session.ingest(a);
                    session_sig.push(if i > 0 { full[i - 1] } else { -1 }); // signature = the previous token
                    session_pred.push(full[i]); // decode = the model's own generated token
                    added += 1;
                }
            }
            eprint!("\r\x1b[2K");
            if added == 0 && session.n_tokens() == 0 {
                eprintln!("[fieldrun] [bucket] arch {arch} exposes no descent substrate (rope/Qwen only) — bucket OFF");
                bucket_on = false;
            } else {
                eprintln!("[bucket] session expert clustering — {} tokens ({added} this reply), K={}, E={}:", session.n_tokens(), bkt.k, bkt.experts);
                eprint!("{}", session.render(bkt.experts));
            }
        }
        if chatml {
            // only instruct models carry conversation history (base completion is stateless per turn)
            history.push(("user".into(), user.to_string()));
            history.push(("assistant".into(), gen_out.text.trim().to_string()));
        }
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
    fn candidate_set_frame_shows_set_and_position() {
        // store-less: context_candidates = last `recent` tokens (most-recent first), no induction.
        let cfg = CandCfg { recent: 3, induction: 0, quad: 0, tri: 0, bi: 0, skel: 0, uni: 0, closed: false };
        let dec = |t: i64| format!("T{t}");
        // ctx tail [10,20,30] → candidates [30,20,10]; predicted 20 is the 2nd, marked, of 3.
        let line = fmt_candidate_set(None, &[40, 10, 20, 30], 20, &cfg, &dec);
        assert!(line.contains("model chose T20"), "{line}");
        assert!(line.contains("candidate #2 of 3"), "{line}");
        assert!(line.contains("T20 ◀"), "{line}");
        // a token not in the set reports the size without a position.
        let miss = fmt_candidate_set(None, &[40, 10, 20, 30], 99, &cfg, &dec);
        assert!(miss.contains("3 candidates"), "{miss}");
    }

    #[test]
    fn render_chat_openai_tool_roundtrip() {
        // user → assistant(tool_calls) → tool result: the prior call + result must land in the prompt as
        // <tool_call>/<tool_response>, and the prompt must open the assistant's next turn.
        let msgs: Vec<Msg> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": serde_json::Value::Null,
             "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "15C sunny"}
        ]))
        .unwrap();
        let p = render_chat(Some("You are helpful."), &msgs);
        assert!(p.contains("<|im_start|>system\nYou are helpful.<|im_end|>"), "{p}");
        assert!(p.contains("<tool_call>") && p.contains("get_weather") && p.contains("\"city\":\"Paris\""), "{p}");
        assert!(p.contains("<tool_response>") && p.contains("15C sunny"), "{p}");
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
    }

    #[test]
    fn render_chat_anthropic_tool_roundtrip() {
        // Anthropic carries tool_use (assistant) + tool_result (user) as content blocks.
        let msgs: Vec<Msg> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather",
                "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "15C"}]}
        ]))
        .unwrap();
        let p = render_chat(None, &msgs);
        assert!(p.contains("<tool_call>") && p.contains("get_weather"), "{p}");
        assert!(p.contains("<tool_response>") && p.contains("15C"), "{p}");
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

    #[test]
    fn openai_tool_frames_stream_calls() {
        // A parsed tool call must stream as an OpenAI tool_calls delta (index + id + function) and finish "tool_calls".
        let calls = vec![crate::tools::ToolCall { name: "get_weather".into(), arguments: serde_json::json!({"city":"Paris"}) }];
        let s = String::from_utf8(openai_tool_frames("rope", 7, "let me check", &calls, 5, true)).unwrap();
        assert!(s.contains(r#""delta":{"role":"assistant"}"#), "{s}");
        assert!(s.contains("\"tool_calls\":[{") && s.contains("\"index\":0") && s.contains("get_weather"), "{s}");
        // arguments must be a JSON *string* on the wire (OpenAI), not an object
        assert!(s.contains(r#""arguments":"{\"city\":\"Paris\"}""#), "{s}");
        assert!(s.contains("\"finish_reason\":\"tool_calls\"") && s.trim_end().ends_with("[DONE]"), "{s}");
        // leading prose before the call is streamed as a content delta
        assert!(s.contains(r#""content":"let me check""#), "{s}");
    }

    #[test]
    fn openai_tool_frames_plain_answer() {
        // No call (model answered despite tools offered) → text streams as content + finish "stop", still an SSE stream.
        let s = String::from_utf8(openai_tool_frames("rope", 7, "Hello there", &[], 2, true)).unwrap();
        assert!(s.contains(r#""content":"Hello there""#) && s.contains("\"finish_reason\":\"stop\""), "{s}");
        assert!(!s.contains("tool_calls"), "{s}");
        assert!(s.trim_end().ends_with("[DONE]"), "{s}");
    }

    #[test]
    fn anthropic_tool_frames_stream_calls() {
        let calls = vec![crate::tools::ToolCall { name: "get_weather".into(), arguments: serde_json::json!({"city":"Paris"}) }];
        let s = String::from_utf8(anthropic_tool_frames("rope", 7, "", &calls, true)).unwrap();
        assert!(s.contains("event: message_start") && s.contains("event: message_stop"), "{s}");
        assert!(s.contains("\"type\":\"tool_use\"") && s.contains("get_weather"), "{s}");
        // arguments arrive as input_json_delta.partial_json (the Anthropic streaming tool-input shape)
        assert!(s.contains("\"type\":\"input_json_delta\"") && s.contains("partial_json"), "{s}");
        assert!(s.contains("\"stop_reason\":\"tool_use\""), "{s}");
    }
}
