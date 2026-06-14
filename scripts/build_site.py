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
        body = "".join(
            f"<p>{render_paragraph(p, marks.get((si, pi), []))}</p>"
            for pi, p in enumerate(paras)
        )
        parts.append(f'<section id="{sid}"><{tag}>{escape(heading)}</{tag}>{body}</section>')

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
