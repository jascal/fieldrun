# Quickstart — from a clean machine to chatting

A copy-paste walkthrough: install Rust, build `fieldrun`, convert a small model, and chat with it. No Python, no
PyTorch, no GPU required. (Model replies below are illustrative — a 0.5B model is small, so expect simple-but-coherent.)

## 0. Prerequisites (one-time)

You need **Rust ≥ 1.82** and a **C compiler** (for the tokenizer).

```bash
# macOS: the C compiler + Apple Accelerate come with the Xcode command-line tools
xcode-select --install            # skip if already installed
# Linux: a C compiler (build-essential / gcc) is usually already present

# Rust (skip if you have it; otherwise:)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version                   # need 1.82+ ; if older: rustup update
```

## 1. Build + install

```bash
git clone https://github.com/jascal/fieldrun
cd fieldrun
cargo install --path .            # ~3–5 min first build; puts `fieldrun` on your PATH (~/.cargo/bin)
fieldrun --help                   # sanity check
```

Speed knob (optional): a tuned BLAS makes dense models much faster on CPU —
- **macOS** (Apple Silicon *or* Intel): `cargo install --path . --features accelerate`
- **Linux**: `cargo install --path . --features openblas` (needs `libopenblas`)

> A harmless macOS link warning (`ld: warning: … Accelerate … built for newer macOS …`) can be silenced with
> `MACOSX_DEPLOYMENT_TARGET=15.0 cargo install …`.

## 2. Convert a model → a runnable bundle

`convert` pulls a checkpoint from Hugging Face by repo id and writes a flat bundle (no torch). Qwen2.5 is ungated, so
no login is needed:

```bash
fieldrun convert --model Qwen/Qwen2.5-0.5B-Instruct --arch rope --dtype f16
```
```
[hub] Qwen/Qwen2.5-0.5B-Instruct (main): single safetensors
[hub] model.safetensors: 988 MB ✓
[convert] 290 arrays -> ~/.cache/fieldrun/bundles/Qwen2.5-0.5B-Instruct/… (arch=rope, dtype=f16, 1 shard, no torch)
```
~1 GB total, into `~/.cache/fieldrun/`. (Gated models like Gemma need `huggingface-cli login` first.)

## 3. Chat (the default mode)

```bash
fieldrun --bundle Qwen2.5-0.5B-Instruct
```
```
[fieldrun] loaded bundle (988 MB)
[fieldrun] chat — type a message; /help for commands, Tab completes them, ↑/↓ history, /exit or Ctrl-D to quit.
[fieldrun] markdown rendering ON (/format to toggle)

you> What is the capital of France?
[ thinking ⠹ 1s ]
bot> The capital of France is **Paris**.

you> Give me three uses for a paperclip.
bot> 1. **Hold papers together** — its original job.
     2. **Reset a device** — straighten it to press a pinhole reset button.
     3. **Improvised hook** — to fish a dropped item out of a tight gap.

you> /exit
[fieldrun] bye
```
Tab-completes the `/` commands, ↑/↓ recalls history, replies render Markdown live (bold/lists; LaTeX `\(x^2\)` → `x²`).
`/help` lists commands; `/format off` for plain text.

## 4. (Optional) See *why* it answered

```bash
fieldrun --bundle Qwen2.5-0.5B-Instruct --explain     # or type /explain on mid-chat
```
After each reply you get the live circuits — attention heads (what each `reads` and `writes` to the logits) and the
top MLP features behind the prediction. `/explain context all` shows the whole prompt; `/explain off` stops.

## 5. (Optional) Serve an OpenAI/Anthropic-compatible API

```bash
fieldrun --bundle Qwen2.5-0.5B-Instruct --serve 8731
curl -s localhost:8731/v1/chat/completions -d '{"messages":[{"role":"user","content":"Capital of France?"}]}'
```
Supports `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, SSE streaming (`"stream":true`), and tool calling
(`"tools":[…]`).

## Picking a model for your machine

`--bundle <name>` runs a **local** bundle (it doesn't download) — `convert` is what pulls + builds. Rough sizing:

| RAM | good choices (dtype) | notes |
|-----|----------------------|-------|
| 8 GB | `Qwen2.5-0.5B-Instruct` (f16), `Qwen2.5-1.5B-Instruct` (f16) | stay ≤1.5B; ungated; snappy |
| 16–24 GB | up to ~4–7B dense (f16 + accelerate), or a 30B-class MoE via `--dtype int8` (expert-offload) | MoE pages experts from disk |
| big disk, modest RAM | large MoE (`qwen3moe`/`mla`) in int8 | experts mmap'd on disk, only hot ones resident |

Files: bundles in `~/.cache/fieldrun/bundles/`, raw downloads in `~/.cache/fieldrun/hub/` (safe to delete after a
successful convert). Everything runs local, CPU by default — no framework at runtime.
