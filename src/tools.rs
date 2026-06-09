//! Tool / function calling for the OpenAI- and Anthropic-compatible API.
//!
//! fieldrun speaks plain text to a generic ChatML prompt, so "tool use" is three model-agnostic pieces, all best-effort:
//!  1. **declare** — render the caller's tool schemas into a Hermes/Qwen-style system preamble that asks the model to
//!     emit `<tool_call>{…}</tool_call>` JSON (the most widely-trained open format).
//!  2. **parse** — pull tool calls back out of the generated text across the common formats: Hermes/Qwen
//!     `<tool_call>…</tool_call>`, Mistral `[TOOL_CALLS] […]`, Llama `<|python_tag|>`/bare JSON, and a generic JSON
//!     object/array fallback. Argument key is normalised across `arguments` (OpenAI/Hermes/Mistral) and `parameters`
//!     (Llama), and a JSON-string `arguments` is decoded.
//!  3. **round-trip** — prior tool calls + their results (OpenAI `tool_calls` + `role:"tool"`, or Anthropic `tool_use`
//!     + `tool_result` content blocks) are rendered back into the prompt by the API layer so the model can continue.
//!
//! The canonical structured form (OpenAI `tool_calls` / Anthropic `tool_use`) is what we return to the client.

use serde_json::{json, Value};

/// A tool the caller offered, normalised across OpenAI/Anthropic request shapes.
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value, // JSON Schema for the arguments
}

/// A tool call we parsed out of the model's output.
pub struct ToolCall {
    pub name: String,
    pub arguments: Value, // parsed arguments object
}

/// Extract tool definitions from a request body, accepting OpenAI (`tools:[{type:"function",function:{…}}]` or the
/// legacy top-level `functions:[{…}]`) and Anthropic (`tools:[{name,description,input_schema}]`) shapes.
pub fn parse_tools(body: &Value) -> Vec<ToolDef> {
    let mut out = Vec::new();
    if let Some(arr) = body.get("tools").and_then(|t| t.as_array()) {
        for t in arr {
            if let Some(f) = t.get("function") {
                // OpenAI: {type:"function", function:{name,description,parameters}}
                out.push(def(f, "parameters"));
            } else {
                // Anthropic: {name, description, input_schema}
                out.push(def(t, "input_schema"));
            }
        }
    }
    if let Some(arr) = body.get("functions").and_then(|t| t.as_array()) {
        for f in arr {
            out.push(def(f, "parameters")); // legacy OpenAI top-level functions
        }
    }
    out.retain(|t| !t.name.is_empty());
    out
}

fn def(v: &Value, params_key: &str) -> ToolDef {
    ToolDef {
        name: v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
        description: v.get("description").and_then(|d| d.as_str()).unwrap_or("").to_string(),
        parameters: v.get(params_key).or_else(|| v.get("parameters")).cloned().unwrap_or_else(|| json!({})),
    }
}

/// `tool_choice:"none"` (OpenAI) / `tool_choice:{type:"none"}` — the caller forbids tool calls this turn.
pub fn choice_none(body: &Value) -> bool {
    match body.get("tool_choice") {
        Some(Value::String(s)) => s == "none",
        Some(Value::Object(o)) => o.get("type").and_then(|t| t.as_str()) == Some("none"),
        _ => false,
    }
}

/// A Hermes/Qwen-style system preamble: lists the tools as JSON and asks for `<tool_call>` output.
pub fn preamble(tools: &[ToolDef]) -> String {
    let list: Vec<Value> = tools
        .iter()
        .map(|t| json!({ "name": t.name, "description": t.description, "parameters": t.parameters }))
        .collect();
    let tools_json = serde_json::to_string(&list).unwrap_or_else(|_| "[]".into());
    format!(
        "You may call one or more functions to assist with the user's request. You are provided function signatures \
         inside <tools></tools>:\n<tools>\n{tools_json}\n</tools>\n\nTo call a function, return a JSON object \
         {{\"name\": <function-name>, \"arguments\": <arguments-object>}} inside <tool_call></tool_call> tags — one \
         <tool_call> block per call. If no function is needed, answer the user directly."
    )
}

/// Parse any tool calls out of generated text, trying each supported format in turn. Empty if the model just answered.
pub fn parse_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // 1. Hermes / Qwen: one or more <tool_call> … </tool_call> blocks.
    let mut rest = text;
    while let Some(s) = rest.find("<tool_call>") {
        let after = &rest[s + "<tool_call>".len()..];
        match after.find("</tool_call>") {
            Some(e) => {
                push_json_calls(after[..e].trim(), &mut calls);
                rest = &after[e + "</tool_call>".len()..];
            }
            None => {
                push_json_calls(after.trim(), &mut calls); // unterminated (hit the cap) — take what's there
                break;
            }
        }
    }
    if !calls.is_empty() {
        return calls;
    }

    // 2. Mistral: [TOOL_CALLS] then a JSON array.
    if let Some(p) = text.find("[TOOL_CALLS]") {
        push_json_calls(text[p + "[TOOL_CALLS]".len()..].trim(), &mut calls);
        if !calls.is_empty() {
            return calls;
        }
    }

    // 3. Llama <|python_tag|> prefix, then 4. generic bare JSON object/array of calls.
    let t = text.trim();
    let t = t.strip_prefix("<|python_tag|>").unwrap_or(t).trim();
    if t.starts_with('{') || t.starts_with('[') {
        push_json_calls(t, &mut calls);
    }
    calls
}

/// The plain text the model emitted *before* any tool call (prose preceding the first marker), trimmed.
pub fn leading_text(text: &str) -> String {
    let cut = ["<tool_call>", "[TOOL_CALLS]", "<|python_tag|>"]
        .iter()
        .filter_map(|m| text.find(m))
        .min();
    match cut {
        Some(0) | None if text.trim_start().starts_with('{') || text.trim_start().starts_with('[') => String::new(),
        Some(i) => text[..i].trim().to_string(),
        None => text.trim().to_string(),
    }
}

/// Parse the leading JSON value of `s` as a tool call (object) or array of calls, pushing any found.
fn push_json_calls(s: &str, out: &mut Vec<ToolCall>) {
    if let Some(v) = first_json_value(s) {
        match v {
            Value::Array(items) => {
                for it in items {
                    if let Some(c) = as_call(&it) {
                        out.push(c);
                    }
                }
            }
            obj @ Value::Object(_) => {
                if let Some(c) = as_call(&obj) {
                    out.push(c);
                }
            }
            _ => {}
        }
    }
}

fn as_call(v: &Value) -> Option<ToolCall> {
    // Llama sometimes wraps as {type:"function", function:{name, arguments}} like the request — unwrap that too.
    let v = v.get("function").filter(|f| f.is_object()).unwrap_or(v);
    let name = v.get("name").and_then(|n| n.as_str())?.to_string();
    let args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or_else(|| json!({}));
    // `arguments` may be a JSON string (OpenAI on the wire) or an object — normalise to a value.
    let args = match args {
        Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    Some(ToolCall { name, arguments: args })
}

/// First complete JSON value at the start of `s` (object or array), tolerating trailing prose after it.
fn first_json_value(s: &str) -> Option<Value> {
    serde_json::Deserializer::from_str(s.trim_start()).into_iter::<Value>().next().and_then(|r| r.ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermes_single_and_multi() {
        let c = parse_calls("<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].name, "get_weather");
        assert_eq!(c[0].arguments["city"], "Paris");
        let multi = parse_calls(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>",
        );
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[1].name, "b");
        assert_eq!(multi[1].arguments["x"], 1);
    }

    #[test]
    fn mistral_and_llama_and_generic() {
        let m = parse_calls("[TOOL_CALLS] [{\"name\":\"f\",\"arguments\":{\"a\":2}}]");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "f");
        // Llama uses "parameters" and may prefix <|python_tag|>
        let l = parse_calls("<|python_tag|>{\"name\":\"lookup\",\"parameters\":{\"q\":\"x\"}}");
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].arguments["q"], "x");
        // generic bare object, with trailing prose tolerated
        let g = parse_calls("{\"name\":\"g\",\"arguments\":{}} sure, calling g");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "g");
    }

    #[test]
    fn arguments_as_json_string() {
        let c = parse_calls("<tool_call>{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":5}\"}</tool_call>");
        assert_eq!(c[0].arguments["k"], 5); // the string was re-parsed into an object
    }

    #[test]
    fn plain_answer_has_no_calls_and_leading_text() {
        assert!(parse_calls("The capital of France is Paris.").is_empty());
        assert_eq!(leading_text("Let me check.<tool_call>{\"name\":\"f\"}</tool_call>"), "Let me check.");
        assert_eq!(leading_text("just a normal answer"), "just a normal answer");
    }

    #[test]
    fn parse_tools_openai_and_anthropic() {
        let oa = json!({"tools":[{"type":"function","function":{"name":"w","description":"d","parameters":{"type":"object"}}}]});
        let t = parse_tools(&oa);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].name, "w");
        assert_eq!(t[0].parameters["type"], "object");
        let an = json!({"tools":[{"name":"a","description":"d","input_schema":{"type":"object"}}]});
        assert_eq!(parse_tools(&an)[0].name, "a");
        assert!(choice_none(&json!({"tool_choice":"none"})));
        assert!(!choice_none(&json!({"tool_choice":"auto"})));
    }
}
