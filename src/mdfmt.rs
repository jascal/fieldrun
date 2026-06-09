//! A tiny, dependency-free Markdown→ANSI renderer for the chat REPL — enough to make a model's Markdown replies
//! readable in a terminal, not a full CommonMark engine. It works one completed line at a time (so it can stream),
//! carrying only a "are we inside a ``` fence" bit between lines.
//!
//! Handled: headings (`#`…), bullets (`-`/`*`/`+`), block quotes (`>`), horizontal rules, fenced code blocks, and the
//! inline spans **bold**, *italic*, `code`, ~~strike~~. Terminals can't render LaTeX, so math is *transliterated*: the
//! `\(…\)`/`\[…\]`/`$…$` delimiters are stripped and the common commands become Unicode — `\theta`→θ, `\sum`→Σ,
//! `\cos`→cos, `\frac{a}{b}`→(a)/(b), `e^{2}`→e², `x_i`→xᵢ — with anything unrecognised degrading to plain text.

// ANSI: bold/italic toggles use their specific off-codes (22/23) so they don't reset a surrounding style; code/quote
// use a colour that we reset to default-foreground (39), so nesting inside a heading keeps the heading's bold.
const BOLD: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const ITAL: &str = "\x1b[3m";
const ITAL_OFF: &str = "\x1b[23m";
const STRIKE: &str = "\x1b[9m";
const STRIKE_OFF: &str = "\x1b[29m";
const CODE: &str = "\x1b[36m"; // cyan
const FG_OFF: &str = "\x1b[39m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Render one completed line of Markdown to an ANSI string. `in_code` tracks an open ``` fence across lines.
pub fn render_line(line: &str, in_code: &mut bool) -> String {
    let trimmed = line.trim_start();
    // fenced code block: ``` toggles; the fence line itself is shown dim, body lines verbatim (no inline parsing).
    if trimmed.starts_with("```") {
        *in_code = !*in_code;
        return format!("{DIM}{line}{RESET}");
    }
    if *in_code {
        return format!("{CODE}{line}{FG_OFF}");
    }
    // horizontal rule
    let body = trimmed.trim_end();
    if body.len() >= 3 && (body.chars().all(|c| c == '-') || body.chars().all(|c| c == '*') || body.chars().all(|c| c == '_')) {
        return format!("{DIM}────────────{RESET}");
    }
    // heading: #.. → bold (h1/h2 also underlined)
    if let Some(h) = trimmed.strip_prefix('#') {
        let level = 1 + h.chars().take_while(|&c| c == '#').count();
        let text = trimmed.trim_start_matches('#').trim();
        let ul = if level <= 2 { "\x1b[4m" } else { "" };
        return format!("{BOLD}{ul}{}{RESET}", inline(text));
    }
    // block quote
    if let Some(q) = trimmed.strip_prefix('>') {
        return format!("{DIM}│{RESET} {}", inline(q.trim_start()));
    }
    // bullet list (keep the original indent, swap the marker for •)
    let indent = &line[..line.len() - trimmed.len()];
    for m in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(m) {
            return format!("{indent}• {}", inline(rest));
        }
    }
    // numbered list "N. " / "N) " — keep the marker, format the rest
    if let Some(p) = numbered_prefix(trimmed) {
        return format!("{indent}{}{}", &trimmed[..p], inline(&trimmed[p..]));
    }
    inline(line)
}

/// Length of a leading "12. " / "12) " ordered-list marker, if present.
fn numbered_prefix(s: &str) -> Option<usize> {
    let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let after = &s[digits..];
    if after.starts_with(". ") || after.starts_with(") ") {
        Some(digits + 2)
    } else {
        None
    }
}

/// Inline formatting. LaTeX is transliterated ONLY inside math delimiters (`\(…\)`, `\[…\]`, `$…$`); everything else is
/// plain text with Markdown spans. So an underscore/caret in ordinary text or an identifier (e.g. `<|im_end|>`, a file
/// name, `my_var`) is left alone — not turned into a sub/superscript.
pub fn inline(s: &str) -> String {
    let mut out = String::new();
    let mut text = String::new(); // pending non-math text
    let mut rest = s;
    'scan: while !rest.is_empty() {
        for (op, cl) in [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")] {
            if let Some(after) = rest.strip_prefix(op) {
                if let Some(end) = after.find(cl) {
                    out.push_str(&spans(&text));
                    text.clear();
                    out.push_str(&latex(&after[..end])); // math content
                    rest = &after[end + cl.len()..];
                    continue 'scan;
                }
            }
        }
        // single `$…$` — only when a closing `$` exists (so a lone `$5` stays literal, not math)
        if let Some(after) = rest.strip_prefix('$') {
            if let Some(end) = after.find('$') {
                out.push_str(&spans(&text));
                text.clear();
                out.push_str(&latex(&after[..end]));
                rest = &after[end + 1..];
                continue 'scan;
            }
        }
        let ch = rest.chars().next().unwrap();
        text.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out.push_str(&spans(&text));
    out
}

/// Markdown inline spans: `code`, **bold**/__bold__, *italic*, ~~strike~~. Unmatched delimiters are left literal.
fn spans(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // `code`
        if c == '`' {
            if let Some(e) = find(&b, i + 1, '`') {
                out.push_str(CODE);
                out.extend(&b[i + 1..e]);
                out.push_str(FG_OFF);
                i = e + 1;
                continue;
            }
        }
        // **bold** or __bold__
        if (c == '*' || c == '_') && i + 1 < b.len() && b[i + 1] == c {
            if let Some(e) = find2(&b, i + 2, c) {
                out.push_str(BOLD);
                out.push_str(&spans(&b[i + 2..e].iter().collect::<String>()));
                out.push_str(BOLD_OFF);
                i = e + 2;
                continue;
            }
        }
        // *italic* (single star only — single `_` is too common in identifiers/math to treat as italic)
        if c == '*' {
            if let Some(e) = find(&b, i + 1, '*') {
                out.push_str(ITAL);
                out.extend(&b[i + 1..e]);
                out.push_str(ITAL_OFF);
                i = e + 1;
                continue;
            }
        }
        // ~~strike~~
        if c == '~' && i + 1 < b.len() && b[i + 1] == '~' {
            if let Some(e) = find2(&b, i + 2, '~') {
                out.push_str(STRIKE);
                out.extend(&b[i + 2..e]);
                out.push_str(STRIKE_OFF);
                i = e + 2;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn find(b: &[char], from: usize, c: char) -> Option<usize> {
    (from..b.len()).find(|&i| b[i] == c)
}

/// Index of the first of a doubled delimiter `cc` at or after `from`.
fn find2(b: &[char], from: usize, c: char) -> Option<usize> {
    (from..b.len().saturating_sub(1)).find(|&i| b[i] == c && b[i + 1] == c)
}

/// Transliterate LaTeX to Unicode/plain text: strip math delimiters, map commands, render `\frac`, super/subscripts.
pub fn latex(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match c {
            '\\' => {
                let next = b.get(i + 1).copied();
                match next {
                    // math delimiters \( \) \[ \] and latex spacers \, \; \! → drop
                    Some('(') | Some(')') | Some('[') | Some(']') | Some(',') | Some(';') | Some('!') => i += 2,
                    Some('\\') => {
                        out.push(' ');
                        i += 2;
                    } // \\ line break
                    Some(ch) if ch.is_ascii_alphabetic() => {
                        let start = i + 1;
                        let mut j = start;
                        while j < b.len() && b[j].is_ascii_alphabetic() {
                            j += 1;
                        }
                        let cmd: String = b[start..j].iter().collect();
                        if (cmd == "frac" || cmd == "dfrac" || cmd == "tfrac") && b.get(j) == Some(&'{') {
                            if let Some((num, k)) = brace_group(&b, j) {
                                if let Some((den, k2)) = brace_group(&b, k) {
                                    out.push_str(&format!("({})/({})", latex(&num), latex(&den)));
                                    i = k2;
                                    continue;
                                }
                            }
                        }
                        match command(&cmd) {
                            Some(rep) => out.push_str(rep),
                            None => out.push_str(&cmd), // unknown command → show the word (no backslash)
                        }
                        i = j;
                    }
                    _ => {
                        i += 1;
                    } // lone backslash → drop
                }
            }
            '^' | '_' => {
                let (inner, k) = script_arg(&b, i + 1);
                let conv = latex(&inner);
                out.push_str(&to_script(&conv, c == '_').unwrap_or_else(|| {
                    let mark = if c == '_' { "_" } else { "^" };
                    if conv.chars().count() == 1 { format!("{mark}{conv}") } else { format!("{mark}({conv})") }
                }));
                i = k;
            }
            '{' | '}' => i += 1, // strip stray grouping braces
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// The argument of a `^`/`_`: a `{…}` group or the single following char. Returns (inner, index past it).
fn script_arg(b: &[char], from: usize) -> (String, usize) {
    if b.get(from) == Some(&'{') {
        if let Some((inner, k)) = brace_group(b, from) {
            return (inner, k);
        }
    }
    match b.get(from) {
        Some(&ch) => (ch.to_string(), from + 1),
        None => (String::new(), from),
    }
}

/// Read a `{…}` group starting at `b[at]=='{'`, honouring nesting. Returns (inner, index past the closing brace).
fn brace_group(b: &[char], at: usize) -> Option<(String, usize)> {
    if b.get(at) != Some(&'{') {
        return None;
    }
    let mut depth = 0;
    let mut inner = String::new();
    for (off, &c) in b[at..].iter().enumerate() {
        match c {
            '{' => {
                depth += 1;
                if depth > 1 {
                    inner.push(c);
                }
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((inner, at + off + 1));
                }
                inner.push(c);
            }
            _ => inner.push(c),
        }
    }
    None
}

/// Convert a string to Unicode super/subscript, or None if any char has no mapping.
fn to_script(s: &str, sub: bool) -> Option<String> {
    s.chars().map(|c| if sub { subscript(c) } else { superscript(c) }).collect()
}

fn superscript(c: char) -> Option<char> {
    Some(match c {
        '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷',
        '8' => '⁸', '9' => '⁹', '+' => '⁺', '-' => '⁻', '=' => '⁼', '(' => '⁽', ')' => '⁾', 'n' => 'ⁿ',
        'i' => 'ⁱ', 'a' => 'ᵃ', 'b' => 'ᵇ', 'c' => 'ᶜ', 'd' => 'ᵈ', 'e' => 'ᵉ', 'x' => 'ˣ', 'y' => 'ʸ',
        'T' => 'ᵀ', ' ' => ' ',
        _ => return None,
    })
}

fn subscript(c: char) -> Option<char> {
    Some(match c {
        '0' => '₀', '1' => '₁', '2' => '₂', '3' => '₃', '4' => '₄', '5' => '₅', '6' => '₆', '7' => '₇',
        '8' => '₈', '9' => '₉', '+' => '₊', '-' => '₋', '=' => '₌', '(' => '₍', ')' => '₎', 'a' => 'ₐ',
        'e' => 'ₑ', 'i' => 'ᵢ', 'j' => 'ⱼ', 'o' => 'ₒ', 'x' => 'ₓ', 'n' => 'ₙ', 'm' => 'ₘ', 't' => 'ₜ',
        ' ' => ' ',
        _ => return None,
    })
}

/// Map a LaTeX command name (without the backslash) to its Unicode/plain replacement, or None to fall back.
fn command(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        // greek (lower)
        "alpha" => "α", "beta" => "β", "gamma" => "γ", "delta" => "δ", "epsilon" => "ε", "varepsilon" => "ε",
        "zeta" => "ζ", "eta" => "η", "theta" => "θ", "vartheta" => "ϑ", "iota" => "ι", "kappa" => "κ",
        "lambda" => "λ", "mu" => "μ", "nu" => "ν", "xi" => "ξ", "pi" => "π", "rho" => "ρ", "sigma" => "σ",
        "tau" => "τ", "upsilon" => "υ", "phi" => "φ", "varphi" => "φ", "chi" => "χ", "psi" => "ψ", "omega" => "ω",
        // greek (upper)
        "Gamma" => "Γ", "Delta" => "Δ", "Theta" => "Θ", "Lambda" => "Λ", "Xi" => "Ξ", "Pi" => "Π",
        "Sigma" => "Σ", "Phi" => "Φ", "Psi" => "Ψ", "Omega" => "Ω",
        // operators / relations / arrows
        "times" => "×", "cdot" => "·", "div" => "÷", "pm" => "±", "mp" => "∓", "ast" => "∗",
        "leq" => "≤", "le" => "≤", "geq" => "≥", "ge" => "≥", "neq" => "≠", "ne" => "≠", "approx" => "≈",
        "equiv" => "≡", "sim" => "∼", "propto" => "∝", "ll" => "≪", "gg" => "≫",
        "rightarrow" => "→", "to" => "→", "leftarrow" => "←", "Rightarrow" => "⇒", "Leftarrow" => "⇐",
        "leftrightarrow" => "↔", "mapsto" => "↦", "uparrow" => "↑", "downarrow" => "↓",
        "infty" => "∞", "partial" => "∂", "nabla" => "∇", "sum" => "∑", "prod" => "∏", "int" => "∫",
        "oint" => "∮", "sqrt" => "√", "angle" => "∠", "perp" => "⊥", "parallel" => "∥",
        "in" => "∈", "notin" => "∉", "ni" => "∋", "subset" => "⊂", "subseteq" => "⊆", "supset" => "⊃",
        "supseteq" => "⊇", "cup" => "∪", "cap" => "∩", "emptyset" => "∅", "varnothing" => "∅",
        "forall" => "∀", "exists" => "∃", "neg" => "¬", "land" => "∧", "lor" => "∨", "oplus" => "⊕",
        "otimes" => "⊗", "star" => "⋆", "bullet" => "•", "dots" => "…", "ldots" => "…", "cdots" => "⋯",
        "Re" => "ℜ", "Im" => "ℑ", "hbar" => "ℏ", "ell" => "ℓ", "aleph" => "ℵ", "deg" => "°",
        // function names: keep the word (drop the backslash) — these need no symbol
        "cos" => "cos", "sin" => "sin", "tan" => "tan", "sec" => "sec", "csc" => "csc", "cot" => "cot",
        "log" => "log", "ln" => "ln", "exp" => "exp", "lim" => "lim", "max" => "max", "min" => "min",
        "det" => "det", "gcd" => "gcd", "arg" => "arg", "dim" => "dim", "ker" => "ker", "deg2" => "deg",
        "left" => "", "right" => "", "quad" => "  ", "qquad" => "    ", "text" => "", "mathrm" => "", "mathbf" => "",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(s: &str) -> String {
        // strip ANSI escapes for assertions
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for x in chars.by_ref() {
                    if x == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn inline_markdown() {
        assert!(inline("a **bold** b").contains(BOLD));
        assert_eq!(plain(&inline("a **bold** b")), "a bold b");
        assert_eq!(plain(&inline("an *em* word")), "an em word");
        assert_eq!(plain(&inline("call `f(x)` now")), "call f(x) now");
    }

    #[test]
    fn inline_only_latexes_math() {
        // plain text / identifiers with _ or ^ are left ALONE (the <|im_end| → imₑnd bug)
        assert_eq!(plain(&inline("<|im_end|>")), "<|im_end|>");
        assert_eq!(plain(&inline("set my_var^2 = foo_bar")), "set my_var^2 = foo_bar");
        // but real math inside delimiters IS transliterated
        assert_eq!(plain(&inline("value \\(x^2\\) and \\(\\theta\\)")), "value x² and θ");
        assert!(plain(&inline("inline $a_i$ here")).contains("aᵢ"));
        // markdown spans still apply around math
        assert!(inline("**bold** then \\(x^2\\)").contains(BOLD));
    }

    #[test]
    fn latex_inline_and_display() {
        // the user's example
        assert_eq!(latex("\\( \\theta \\)").trim(), "θ");
        let disp = latex("\\[ e^{i\\theta} = \\cos(\\theta) + i\\sin(\\theta) \\]");
        assert!(disp.contains("cos(θ)"), "{disp}");
        assert!(disp.contains("sin(θ)"), "{disp}");
        assert!(disp.contains("e^(iθ)"), "{disp}"); // multi-char exponent → ^(…)
    }

    #[test]
    fn latex_scripts_and_frac() {
        assert_eq!(latex("x^2"), "x²");
        assert_eq!(latex("a_i"), "aᵢ");
        assert_eq!(latex("\\frac{a}{b}"), "(a)/(b)");
        assert_eq!(latex("2 \\times 3 \\leq 7"), "2 × 3 ≤ 7");
        // n,=,0 all have subscript forms, so no fallback parens; ∞ has no superscript, so the exponent falls back to ^∞
        assert_eq!(latex("\\sum_{n=0}^{\\infty}"), "∑ₙ₌₀^∞");
    }

    #[test]
    fn blocks() {
        let mut code = false;
        assert!(render_line("# Title", &mut code).contains(BOLD));
        assert_eq!(plain(&render_line("   - item", &mut code)), "   • item");
        assert_eq!(plain(&render_line("3. **Form**:", &mut code)), "3. Form:");
        // code fence keeps body verbatim
        render_line("```rust", &mut code);
        assert!(code);
        assert!(render_line("let x = **not bold**;", &mut code).contains("**not bold**"));
        render_line("```", &mut code);
        assert!(!code);
    }
}
