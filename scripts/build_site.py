#!/usr/bin/env python3
"""Build the GitHub Pages slide-reader site from the PIC *supplement* PDF.

The site is a two-pane teaching reader for the explanatory supplement
("Projective Incidence Calculus — A Guide to the Composition Core…"):

  - LEFT  : a seminar slide deck (site/slides.json), pitched at an upper-
            undergraduate / lower-graduate audience and drawn from the
            supplement's pedagogy.
  - RIGHT : the supplement's own text, with each slide's quotes highlighted
            in place, above an embedded copy of the supplement PDF.

Two PDFs ship alongside the site:
  - docs/supplement.pdf — the supplement (the text rendered in the reader,
                          and the canonical figures/tables/equations view).
  - docs/paper.pdf      — the original research paper, kept as a reference
                          link ("What a Transformer Retrieves and What It
                          Computes").

Pipeline:  supplement PDF --(pdftotext)--> sections --(+ site/slides.json
highlights)--> docs/index.html rendered through site/template.html.

Re-runs automatically in CI (.github/workflows/pages.yml) whenever a PDF,
the slide definitions, or the template change, so dropping a new supplement
into paper/ regenerates the site.

The supplement is rendered by a different LaTeX class than the draft paper:
section numbers and titles land on *separate* lines ("3" / "The transformer
mathematics you need"), there is a table of contents, and the body is far
more math-heavy. So heading detection is driven by the table of contents:
a body line that is a bare section number/letter is only treated as a
heading when the following line matches that entry's TOC title. This makes
it robust to the page numbers, equation tags, and decimals that otherwise
look exactly like section numbers.

Requires: pdftotext (poppler-utils). Stdlib-only Python.
"""
import json
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
SITE_DIR = ROOT / "site"
OUT_DIR = ROOT / "docs"

# A standalone page number or equation tag: "23", "(4)".
NOISE_RE = re.compile(r"^\(?\d{1,3}\)?$")
# Trailing dot-leaders + page number in a TOC line: " . . . . . 18".
TOC_LEADER_RE = re.compile(r"\s*\.(\s*\.)+.*$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_text(pdf: Path) -> str:
    res = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"pdftotext failed: {res.stderr}")
    return res.stdout


def is_noise_line(s: str) -> bool:
    """True for standalone page numbers, equation tags, and bare math fragments.

    pdftotext scatters the supplement's equations into short fragment lines
    ("X", "j", "(1)", "Gvw = ⟨Uv , Uw ⟩") between prose paragraphs; we keep
    only lines that carry real words so the reader text and the slide-quote
    matcher see clean sentences. A "word" is a run of >=3 letters; prose has
    at least two."""
    t = s.strip()
    if not t:
        return False
    if NOISE_RE.match(t):
        return True
    return len(re.findall(r"[A-Za-zÀ-ɏ]{3,}", t)) < 2


def is_prose(p: str) -> bool:
    """Keep a joined paragraph only if it reads as prose (>=3 real words)."""
    return len(re.findall(r"[A-Za-zÀ-ɏ]{3,}", p)) >= 3


def join_paragraph(lines: list[str]) -> str:
    text = " ".join(l.strip() for l in lines if l.strip())
    text = re.sub(r"([a-z])- ([a-z])", r"\1\2", text)   # de-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def clean_toc_title(rest: str) -> str:
    rest = TOC_LEADER_RE.sub("", rest)          # drop dot-leaders + page no.
    rest = re.sub(r"\s+\d+$", "", rest)         # or a bare trailing page no.
    return norm(rest)


def parse_toc(lines: list[str], lo: int, hi: int) -> list:
    """Return the table of contents as an ordered list of (id, level, label, title).

    Headings in the body are then matched by *title* in this order — the bare
    section numbers pdftotext emits are unreliable (it drops section 2's "2"
    entirely and floats equation tags like "(3)" between "3.2" and its title),
    but the title line survives intact, and the ordering disambiguates titles
    that recur as body text (e.g. the "Projective Incidence Calculus" column
    header inside Definition 1)."""
    entries = []
    for raw in lines[lo:hi]:
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^(\d+)\.(\d+)\s+(.+)$", s)
        if m:
            n, k, t = int(m.group(1)), int(m.group(2)), clean_toc_title(m.group(3))
            entries.append((f"sec-{n}-{k}", 2, f"{n}.{k} {t}", t))
            continue
        m = re.match(r"^(\d+)\s+(.+)$", s)
        if m:
            n, t = int(m.group(1)), clean_toc_title(m.group(2))
            entries.append((f"sec-{n}", 1, f"{n} {t}", t))
            continue
        m = re.match(r"^([A-Z])\s+(.+)$", s)
        if m:
            x, t = m.group(1), clean_toc_title(m.group(2))
            entries.append((f"sec-{x.lower()}", 1, f"{x} {t}", t))
    return entries


def parse_sections(text: str):
    """Return (preamble_lines, [(id, level, heading, [paragraphs])]).

    preamble_lines[0] is the title; the rest are subtitle/byline lines.
    Sections include a synthetic 'Abstract' (id 'sec-abstract') followed by
    the numbered body sections and lettered appendices, ids 'sec-1',
    'sec-3-2', 'sec-a', … chosen so slides.json can target them directly."""
    lines = [l.replace("\f", "").rstrip() for l in text.split("\n")]

    def next_nonblank(idx: int) -> int:
        k = idx + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        return k

    def find(token: str, start: int = 0) -> int:
        for i in range(start, len(lines)):
            if lines[i].strip() == token:
                return i
        return -1

    abstract_idx = find("Abstract")
    contents_idx = find("Contents", max(abstract_idx, 0))
    if abstract_idx < 0 or contents_idx < 0:
        sys.exit("supplement parse: could not find 'Abstract'/'Contents' markers")

    # --- preamble (title block, before the abstract) ---
    head = [l.strip() for l in lines[:abstract_idx] if l.strip()]
    title = head[0] if head else "Projective Incidence Calculus"
    rest = head[1:]
    cut = next((i for i, l in enumerate(rest) if l.startswith("An explanatory")), len(rest))
    preamble = [title]
    if rest[:cut]:
        preamble.append(" ".join(rest[:cut]))     # subtitle
    tail = rest[cut:]
    date = tail.pop() if tail and re.search(r"\b\d{4}$", tail[-1]) else None
    if tail:
        preamble.append(" ".join(tail))           # "An explanatory companion to …"
    if date:
        preamble.append(date)                     # standalone date line

    # --- locate the section-1 TOC title, then the body start ---
    sec1_title = None
    for raw in lines[contents_idx + 1: contents_idx + 30]:
        m = re.match(r"^1\s+([A-Za-z].+)$", raw.strip())
        if m and not re.match(r"^1\.\d", raw.strip()):
            sec1_title = clean_toc_title(m.group(1))
            break
    if not sec1_title:
        sys.exit("supplement parse: could not read section-1 title from the TOC")

    body_start = -1
    for i in range(contents_idx + 1, len(lines)):
        if lines[i].strip() == "1":
            j = next_nonblank(i)
            if j < len(lines) and norm(lines[j].strip()) == norm(sec1_title):
                body_start = i
                break
    if body_start < 0:
        sys.exit("supplement parse: could not find the body's section 1")

    entries = parse_toc(lines, contents_idx + 1, body_start)

    # --- abstract section ---
    abstract = (
        "sec-abstract", 1, "Abstract",
        _paragraphs(lines, abstract_idx + 1, contents_idx),
    )

    # --- numbered body: match each TOC title (in order) as a standalone line ---
    sections = [abstract]
    cur = None
    buf: list[str] = []
    titles = [norm(t) for (_, _, _, t) in entries]

    def flush():
        nonlocal buf
        if cur is not None and buf:
            p = join_paragraph(buf)
            if p and is_prose(p):
                cur[3].append(p)
        buf = []

    ti = 0
    for i in range(body_start, len(lines)):
        s = lines[i].strip()
        if ti < len(entries) and s and norm(s) == titles[ti]:
            flush()
            if cur is not None:
                sections.append(cur)
            sid, level, label, _ = entries[ti]
            cur = (sid, level, label, [])
            ti += 1
        elif not s:
            flush()
        elif not is_noise_line(s):
            buf.append(s)
    flush()
    if cur is not None:
        sections.append(cur)

    seen = [e[0] for e in entries if e[0] in {s[0] for s in sections}]
    if len(seen) < len(entries):
        miss = [e[2] for e in entries if e[0] not in {s[0] for s in sections}]
        print(f"  [warn] TOC headings not located in body (title wrapped?): {miss}")
    return preamble, sections


def _paragraphs(lines: list[str], lo: int, hi: int) -> list[str]:
    out, buf = [], []
    for raw in lines[lo:hi]:
        s = raw.strip()
        if not s:
            if buf:
                p = join_paragraph(buf)
                if p and is_prose(p):
                    out.append(p)
                buf = []
        elif not is_noise_line(s):
            buf.append(s)
    if buf:
        p = join_paragraph(buf)
        if p and is_prose(p):
            out.append(p)
    return out


# inline enumerators "(a) … (b) …" / "(i) … (ii) …" that pdftotext flattens into a
# run-on; we break them onto their own lines, but only when >=2 appear in one
# paragraph (so a lone "i(v)" or "(v)" in prose is never touched).
ENUM_RE = re.compile(r" (\((?:[a-e]|i{1,3}|iv|vi?|vii|viii|ix|x)\)) ")


def break_enumerations(html_para: str) -> str:
    if len(ENUM_RE.findall(html_para)) < 2:
        return html_para
    return ENUM_RE.sub(r'<br><span class="enum">\1</span> ', html_para)


def render_paragraph(text: str, marks: list[tuple[int, int, int]]) -> str:
    """marks: list of (start, end, slide_idx) spans on the raw text."""
    out, pos = [], 0
    for s, e, idx in sorted(marks):
        if s < pos:
            continue
        out.append(escape(text[pos:s]))
        out.append(f'<mark class="hl s{idx}">{escape(text[s:e])}</mark>')
        pos = e
    out.append(escape(text[pos:]))
    return "".join(out)


def quote_regex(quote: str) -> re.Pattern:
    """Match the quote as a phrase, tolerant of any run of whitespace between
    words (the reader text is whitespace-collapsed, but be safe)."""
    return re.compile(r"\s+".join(re.escape(w) for w in quote.split()))


# ---- recreated figures -----------------------------------------------------
# The supplement's three figures are TikZ vector graphics that pdftotext cannot
# extract, so they never reach the reader text. We rebuild faithful HTML/SVG
# versions (styled by template.html's .fig rules, so they read on both the dark
# slide pane and the light reader pane), embed them in the relevant slides via
# [[FIGn]] tokens, and inject them at their "Figure n:" captions in the reader.
# The full-fidelity originals remain in the embedded supplement PDF.

def _fig1() -> str:
    # composed % are the measured values (labelled on each bar, as in the
    # original); the retrieved/selected split is schematic.
    rows = [("Qwen<br>0.5B", 38, 90, 22, "15"),
            ("Pythia<br>70M", 54, 84, 12, "7.8"),
            ("160M", 53, 82, 15, "10.0"),
            ("410M", 51, 81, 18, "11.6"),
            ("1B", 50, 78, 22, "14.8")]
    bars = "".join(
        f'<div class="fig1-col"><div class="fig1-cval">{c}</div>'
        f'<div class="fig1-bar"><i class="c" style="height:{ch}px"></i>'
        f'<i class="s" style="height:{sh}px"></i><i class="r" style="height:{rh}px"></i></div>'
        f'<div class="fig1-lab">{lab}</div></div>'
        for lab, rh, sh, ch, c in rows
    )
    leg = ('<div class="fig1-leg">'
           '<span><i style="background:#2c6a9b"></i>retrieved</span>'
           '<span><i style="background:#b8cce4"></i>selected</span>'
           '<span><i style="background:#c0504d"></i>composed</span></div>')
    cap = ('<div class="cap">The measured route split. The composed band (red, labelled %) '
           'grows with scale, from Qwen2.5-0.5B through the Pythia ladder 70M to 1B; retrieval '
           'and selection cover the rest. Composed values are measured; the retrieved/selected '
           'split is schematic.</div>')
    return f'<div class="fig"><div class="fig1-bars">{bars}</div>{leg}{cap}</div>'


FIG1 = _fig1()

FIG2 = (
    '<div class="fig"><div class="fig2">'
    '<div class="top"><b>One kernel G</b>'
    'L<sub>v</sub> = Σ<sub>j</sub> c<sub>j</sub><sup>v</sup> &nbsp;·&nbsp; '
    'G<sub>vw</sub> = ⟨U<sub>v</sub>, U<sub>w</sub>⟩ &nbsp;·&nbsp; a semiring-weighted program Π</div>'
    '<div class="arrow">↓</div>'
    '<div class="row">'
    '<div class="card"><b class="t1">I. Logic (T=1)</b>log-semiring; probabilistic incidence '
    'calculus; softmax recovered as an incidence frequency</div>'
    '<div class="card"><b class="t0">II. Geometry (T=0)</b>tropical semiring; the arg max surface '
    'is the Laguerre power diagram (greedy decode)</div>'
    '<div class="card"><b>III. Computation</b>any T, one program; a semiring-weighted Datalog Π; '
    'evaluation is inference</div>'
    '</div>'
    '<div class="maslov">↞ Maslov dequantization (T→0) links Logic and Geometry ↠</div>'
    '<div class="cap">One object, three readings of a single program carrying the Gram kernel G, '
    'selected by the temperature T.</div>'
    '</div></div>'
)

FIG3 = (
    '<div class="fig fig3">'
    '<svg viewBox="0 0 360 210" role="img" aria-label="Power-diagram schematic">'
    '<polygon points="195,108 235,210 0,210 0,80" fill="#f3dede"/>'
    '<polygon points="195,108 175,0 0,0 0,80" fill="#ffffff"/>'
    '<polygon points="195,108 175,0 360,0 360,95" fill="#ffffff"/>'
    '<polygon points="195,108 360,95 360,210 235,210" fill="#ffffff"/>'
    '<g stroke="#b9b2a3" stroke-width="1.4">'
    '<line x1="195" y1="108" x2="175" y2="0"/>'
    '<line x1="195" y1="108" x2="0" y2="80"/>'
    '<line x1="195" y1="108" x2="235" y2="210"/>'
    '<line x1="195" y1="108" x2="360" y2="95"/></g>'
    '<g fill="#1e3a5f">'
    '<circle cx="90" cy="44" r="3.2"/><circle cx="272" cy="48" r="3.2"/>'
    '<circle cx="300" cy="162" r="3.2"/><circle cx="92" cy="166" r="3.2"/></g>'
    '<line x1="172" y1="150" x2="207" y2="138" stroke="#b3541e" stroke-width="1.4" stroke-dasharray="4 3"/>'
    '<circle cx="172" cy="150" r="4" fill="#c0504d"/>'
    '<g font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#333">'
    '<text x="76" y="40">U_b</text><text x="262" y="44">U_a</text>'
    '<text x="286" y="178">U_v*</text><text x="78" y="186">U_t (winner)</text>'
    '<text x="158" y="148" fill="#c0504d">r</text></g>'
    '<text x="118" y="124" font-family="Helvetica,Arial,sans-serif" font-size="9.5" fill="#b3541e">'
    'margin = Δ / ‖U_t−U_v*‖</text>'
    '</svg>'
    '<div class="cap">Schematic of the residual-space power diagram: each cell is where one token '
    'wins the arg&nbsp;max; the residual r sits in the winning U_t cell, and the normalised margin '
    'is its perpendicular distance to the nearest facet.</div>'
    '</div>'
)

FIGURES = {"[[FIG1]]": FIG1, "[[FIG2]]": FIG2, "[[FIG3]]": FIG3}
# the reader injects each figure before the paragraph that opens its caption:
CAPTION_FIGS = [("Figure 1:", FIG1), ("Figure 2:", FIG2), ("Figure 3:", FIG3)]


# ---- recreated §1 notation table -------------------------------------------
# pdftotext flattens the symbol table and drops the entire Symbol column; we
# rebuild it (stable content) and inject it in place of the mangled rows.
_NOTATION = [
    ("Spaces and vectors", [
        ("ℝ<sup>d</sup>", "the d-dimensional real representation space (§3)"),
        ("⟨u, v⟩", "inner product / generalised dot product of u and v (§3)"),
        ("‖u‖", "Euclidean length (norm) of u, ‖u‖ = √⟨u, u⟩ (§3)"),
        ("r", "the residual-stream vector at the final position (§3)"),
        ("d<sub>j</sub>", "the write of source j into the residual stream, r = Σ<sub>j</sub> d<sub>j</sub> (§3)"),
        ("U<sub>v</sub>", "the unembedding direction (output direction) of token v (§3)"),
        ("b<sub>v</sub>", "the (optional) output bias for token v (§3)"),
    ]),
    ("The readout", [
        ("L<sub>v</sub>", "the logit (score) of token v, L<sub>v</sub> = ⟨r, U<sub>v</sub>⟩ + b<sub>v</sub> = Σ<sub>j</sub> c<sub>j</sub><sup>v</sup> (§3)"),
        ("c<sub>j</sub><sup>v</sup>", "source j's vote for token v, c<sub>j</sub><sup>v</sup> = ⟨d<sub>j</sub>, U<sub>v</sub>⟩ (DLA) (§3)"),
        ("softmax(L)<sub>v</sub>", "exp(L<sub>v</sub>) / Σ<sub>w</sub> exp(L<sub>w</sub>), the output probability of v (§3)"),
        ("arg max<sub>v</sub>", "the token attaining the largest value (greedy prediction) (§3)"),
        ("G<sub>vw</sub>", "Gram matrix of token directions, G<sub>vw</sub> = ⟨U<sub>v</sub>, U<sub>w</sub>⟩ (§3)"),
    ]),
    ("Diagnostics", [
        ("PR", "participation ratio, the effective number of contributing sources (§3)"),
        ("µ<sub>t</sub>", "readout multiplicity: how many sources' own arg max is t (§3)"),
        ("Δ", "winning margin L<sub>t</sub> − L<sub>v*</sub> (v* is the runner-up) (§3)"),
        ("D<sub>j</sub>", "differential incidence c<sub>j</sub><sup>t</sup> − c<sub>j</sub><sup>v*</sup> = ⟨d<sub>j</sub>, U<sub>t</sub> − U<sub>v*</sub>⟩ (§6)"),
    ]),
    ("Logic and incidence", [
        ("I", "a finite set of incidences (possible worlds / samples) (§4)"),
        ("i(A)", "the incidence set of proposition A, i(A) ⊆ I (§4)"),
        ("P(A)", "probability of A, recovered as |i(A)| / |I| (§4)"),
        ("|·|", "cardinality of a set (or absolute value of a number) (§4)"),
    ]),
    ("Semiring / temperature", [
        ("⊕, ⊗", 'the "add" and "multiply" of a semiring (§5)'),
        ("T", "temperature: T = 1 gives softmax, T → 0 gives greedy arg max (§5)"),
        ("⊕<sub>v</sub>", "semiring sum over tokens (e.g. max<sub>v</sub> in the tropical case) (§5)"),
    ]),
]


def _notation_table() -> str:
    rows = ["<tr><th>Symbol</th><th>Meaning (section of first use)</th></tr>"]
    for grp, items in _NOTATION:
        rows.append(f'<tr class="grp"><td colspan="2">{grp}</td></tr>')
        for sym, mean in items:
            rows.append(f'<tr><td class="sym">{sym}</td><td>{mean}</td></tr>')
    return '<table class="notation">' + "".join(rows) + "</table>"


NOTATION_TABLE = _notation_table()


# ---- boxed asides ----------------------------------------------------------
# The supplement's coloured boxes flatten into prose. We can't recover their
# exact extent, but their titles are distinctive, so we set off the paragraph
# that carries each one as a callout and bold the title. Each pattern occurs
# ONLY as a box title in the supplement, so matching anywhere is safe.
ASIDE_RE = re.compile(
    r"Aside:"
    r"|What is a semiring\?|What is a kernel\?|What is ablation\?|What is Maslov dequantization\?"
    r"|Worked example:"
    r"|Scope of the claims:"
    r"|Two measured quantities that organise everything"
    r"|Repair, recovery, and rescue:"
    r"|Three things make this more than algebra"
    r"|The one-sentence version\."
    r"|What to remember\."
)

# ---- references (§10): split entries at their [SP]/[1]/[F1]/[A1] labels -----
REF_RE = re.compile(r"(\[(?:SP|\d{1,2}|[A-Z]\d{1,2})\])")


def format_references(html_para: str) -> str:
    parts = REF_RE.split(html_para)        # [lead, label, body, label, body, ...]
    if len(parts) < 3:                     # no citation label -> ordinary prose
        return f"<p>{html_para}</p>"
    out = []
    lead = parts[0].strip()
    if lead:                               # a group header glued before the first entry
        out.append(f'<div class="refgrp">{lead}</div>')
    for i in range(1, len(parts), 2):
        out.append(f'<div class="refentry"><span class="refnum">{parts[i]}</span>{parts[i + 1]}</div>')
    return "".join(out)


def main() -> None:
    supp = next(iter(sorted(PAPER_DIR.glob("*supplement*.pdf"))), None)
    if not supp:
        sys.exit("no supplement PDF found in paper/ (expected paper/*supplement*.pdf)")
    paper = next(
        (p for p in sorted(PAPER_DIR.glob("*.pdf")) if "supplement" not in p.name.lower()),
        None,
    )

    slides = json.loads((SITE_DIR / "slides.json").read_text())
    template = (SITE_DIR / "template.html").read_text()

    # expand [[FIGn]] tokens in the slides into the recreated figures
    for slide in slides:
        for tok, fightml in FIGURES.items():
            if tok in slide["h"]:
                slide["h"] = slide["h"].replace(tok, fightml)

    preamble, sections = parse_sections(extract_text(supp))

    # locate highlight quotes: marks[(sec_i, para_i)] -> [(start, end, slide_idx)]
    marks: dict[tuple[int, int], list] = {}
    missing = []
    for idx, slide in enumerate(slides):
        for quote in slide.get("quotes", []):
            rx = quote_regex(quote)
            hit = False
            for si, (_, _, _, paras) in enumerate(sections):
                for pi, para in enumerate(paras):
                    m = rx.search(para)
                    if m:
                        marks.setdefault((si, pi), []).append((m.start(), m.end(), idx))
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                missing.append((idx, quote[:60]))

    # render paper pane
    parts = []
    if preamble:
        parts.append(f'<div id="ptitle">{escape(preamble[0])}</div>')
        for p in preamble[1:]:
            parts.append(f'<div id="pauth">{escape(p)}</div>')
    for si, (sid, level, heading, paras) in enumerate(sections):
        tag = "h2" if level == 1 else "h3"
        chunks = []
        for pi, p in enumerate(paras):
            # §1: replace the mangled symbol table (and the rest of the section is that table)
            if sid == "sec-1" and p.startswith("Meaning (section of first use)"):
                chunks.append(NOTATION_TABLE)
                break
            html_p = break_enumerations(render_paragraph(p, marks.get((si, pi), [])))
            if sid == "sec-10":                          # references: one entry per line
                chunks.append(format_references(html_p))
                continue
            fig = next((f for pre, f in CAPTION_FIGS if p.startswith(pre)), None)
            if fig:
                chunks.append(fig)                       # figure, then its caption paragraph
            m = ASIDE_RE.search(p)
            if m:                                        # set off a boxed aside as a callout
                title = escape(m.group(0))
                html_p = html_p.replace(title, f'<span class="lead">{title}</span>', 1)
                chunks.append(f'<div class="aside">{html_p}</div>')
            else:
                chunks.append(f"<p>{html_p}</p>")
        parts.append(f'<section id="{sid}"><{tag}>{escape(heading)}</{tag}>{"".join(chunks)}</section>')

    page = template.replace("__PAPER__", "\n".join(parts)).replace(
        "__SLIDES__", json.dumps(slides, ensure_ascii=False)
    )

    # Validate BEFORE writing anything: a slide whose target section is not found in the
    # supplement must fail without clobbering the last-good docs/index.html on disk.
    ids = {sid for sid, _, _, _ in sections}
    bad = [s["target"] for s in slides if s["target"] not in ids]
    if bad:
        sys.exit(f"slide targets missing from supplement: {bad}\n(available: {sorted(ids)})")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "index.html").write_text(page)
    (OUT_DIR / ".nojekyll").write_text("")
    # The reader text is the supplement; ship the supplement PDF (figures/tables/equations are
    # vector graphics pdftotext can't extract) and the original research paper alongside it.
    shutil.copyfile(supp, OUT_DIR / "supplement.pdf")
    if paper:
        shutil.copyfile(paper, OUT_DIR / "paper.pdf")
    else:
        print("  [warn] no original paper PDF found; the 'Original paper' link will 404")

    found = sum(len(v) for v in marks.values())
    print(f"built docs/index.html from {supp.name}: {len(sections)} sections, "
          f"{found} highlights placed, {len(missing)} quotes unmatched")
    for idx, q in missing:
        print(f"  [warn] slide {idx}: quote not found: {q!r}…")


if __name__ == "__main__":
    main()
