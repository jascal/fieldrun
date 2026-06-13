#!/usr/bin/env python3
"""Build the GitHub Pages slide-reader site from the draft paper PDF.

Pipeline:  paper/*.pdf --(pdftotext)--> sections --(+ site/slides.json
highlights)--> docs/index.html rendered through site/template.html.

Re-runs automatically in CI (.github/workflows/pages.yml) whenever the
PDF, the slide definitions, or the template change, so dropping a new
draft into paper/ regenerates the site.

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

HEADING_RE = re.compile(r"^(\d+\.\d+|\d+\.)\s+[A-Z]|^(Abstract|References|Acknowledgements)\s*$")

# The DRAFT watermark is a Type-3 font overlay with no Unicode map, so pdftotext
# pulls its diagonal glyphs as standalone, mis-grouped fragments ("D", "RA", "FT"
# — one set per page) that otherwise scatter through the reader text. They are
# exactly the contiguous substrings of "DRAFT" and never collide with the paper's
# own short caps (PR, PIC, DLA, VHL aren't substrings of DRAFT), so any whole line
# equal to one of them is watermark debris and is dropped in clean_line().
_WM = "DRAFT"
WATERMARK_FRAGMENTS = {_WM[i:j] for i in range(len(_WM)) for j in range(i + 1, len(_WM) + 1)}

# pdftotext splits the paper's small-caps route names; undo that.
CLEANUPS = [
    (re.compile(r"\bR\s+ETRIEVED\b"), "RETRIEVED"),
    (re.compile(r"\bS\s+ELECTED\b"), "SELECTED"),
    (re.compile(r"\bC\s+OM\s*-?\s*POSED\b"), "COMPOSED"),
    (re.compile(r"\bC\s+OMPOSED\b"), "COMPOSED"),
]


def extract_text(pdf: Path) -> str:
    res = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"pdftotext failed: {res.stderr}")
    return res.stdout


def clean_line(line: str) -> str | None:
    s = line.rstrip()
    if s.strip() in WATERMARK_FRAGMENTS:             # DRAFT watermark glyph fragments
        return None
    if re.fullmatch(r"\d{1,3}", s.strip()):          # page numbers / figure axis ticks
        return None
    if re.fullmatch(r"[\d.%\s]+", s.strip() or "x") and len(s.strip()) <= 6:
        return None                                   # stray figure values
    return s


def join_paragraph(lines: list[str]) -> str:
    text = " ".join(l.strip() for l in lines if l.strip())
    text = re.sub(r"([a-z])- ([a-z])", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text).strip()
    for rx, rep in CLEANUPS:
        text = rx.sub(rep, text)
    return text


def heading_id(heading: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)?", heading)
    if m:
        return f"sec-{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "")
    return "sec-" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def parse_sections(text: str):
    """Return (preamble_paragraphs, [(id, level, heading, [paragraphs])])."""
    lines = [clean_line(l) for l in text.splitlines()]
    lines = [l for l in lines if l is not None]

    sections, preamble = [], []
    cur = None        # (heading, level, [para line-buffers])
    buf: list[str] = []

    def flush_para(store):
        nonlocal buf
        if buf:
            p = join_paragraph(buf)
            if p:
                store.append(p)
            buf = []

    i = 0
    prev_state = "blank"  # 'blank' | 'head' | 'text'
    while i < len(lines):
        line = lines[i]
        pagebreak = line.startswith("\f")
        if pagebreak:
            line = line.lstrip("\f")
        is_head = bool(HEADING_RE.match(line.strip()))
        # Top-level "N. Title" lines also occur as enumerated list items mid-
        # paragraph; real top-level headings follow a blank line, a page break,
        # or another heading. Subsection "N.M Title" lines are unambiguous.
        if is_head and re.match(r"^\d+\.\s", line.strip()):
            if prev_state == "text" and not pagebreak:
                is_head = False
        if is_head:
            heading = line.strip()
            # absorb wrapped heading remainder ("...the Tropical Deci-/sion Surface")
            while heading.endswith("-") or (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and len(lines[i + 1].split()) <= 2
                and not HEADING_RE.match(lines[i + 1].strip())
                and not lines[i + 1].strip()[0].isdigit()
                and heading[-1] not in ".?!:"
                and len(heading.split()) >= 2
            ):
                nxt = lines[i + 1].strip()
                heading = (heading[:-1] + nxt) if heading.endswith("-") else heading + " " + nxt
                i += 1
            target = cur[2] if cur else preamble
            flush_para(target)
            if cur:
                sections.append(cur)
            level = 2 if re.match(r"^\d+\.\d+", heading) else 1
            cur = (heading, level, [])
            prev_state = "head"
        elif not line.strip():
            flush_para(cur[2] if cur else preamble)
            prev_state = "blank"
        else:
            buf.append(line)
            prev_state = "text"
        i += 1
    flush_para(cur[2] if cur else preamble)
    if cur:
        sections.append(cur)
    return preamble, sections


def render_paragraph(text: str, marks: list[tuple[int, int, int]]) -> str:
    """marks: list of (start, end, slide_idx) spans on the raw text."""
    marks = sorted(marks)
    out, pos = [], 0
    for s, e, idx in marks:
        if s < pos:
            continue
        out.append(escape(text[pos:s]))
        out.append(f'<mark class="hl s{idx}">{escape(text[s:e])}</mark>')
        pos = e
    out.append(escape(text[pos:]))
    return "".join(out)


def quote_regex(quote: str) -> re.Pattern:
    for rx, rep in CLEANUPS:
        quote = rx.sub(rep, quote)
    return re.compile(re.escape(quote).replace(r"\ ", r"\s+"))


def main() -> None:
    pdfs = sorted(PAPER_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit("no PDF found in paper/")
    pdf = pdfs[0]
    slides = json.loads((SITE_DIR / "slides.json").read_text())
    template = (SITE_DIR / "template.html").read_text()

    preamble, sections = parse_sections(extract_text(pdf))

    # locate highlight quotes: section_paras[(sec_i, para_i)] -> [(s, e, slide_idx)]
    marks: dict[tuple[int, int], list] = {}
    missing = []
    for idx, slide in enumerate(slides):
        for quote in slide.get("quotes", []):
            rx = quote_regex(quote)
            hit = False
            for si, (_, _, paras) in enumerate(sections):
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
    for si, (heading, level, paras) in enumerate(sections):
        sid = heading_id(heading)
        tag = "h2" if level == 1 else "h3"
        body = "".join(
            f"<p>{render_paragraph(p, marks.get((si, pi), []))}</p>"
            for pi, p in enumerate(paras)
        )
        parts.append(f'<section id="{sid}"><{tag}>{escape(heading)}</{tag}>{body}</section>')

    page = template.replace("__PAPER__", "\n".join(parts)).replace(
        "__SLIDES__", json.dumps(slides, ensure_ascii=False)
    )

    # Validate BEFORE writing anything: a malformed draft (a slide's target section not found
    # in the PDF) must fail without leaving a stale/broken docs/index.html behind. Otherwise a
    # bad draft both fails CI *and* clobbers the last-good site on disk.
    ids = {heading_id(h) for h, _, _ in sections}
    bad = [s["target"] for s in slides if s["target"] not in ids]
    if bad:
        sys.exit(f"slide targets missing from paper: {bad}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "index.html").write_text(page)
    (OUT_DIR / ".nojekyll").write_text("")
    # Ship the source PDF alongside the site so the reader can link/embed it (figures and tables are
    # vector graphics pdftotext can't extract — paper.pdf is the canonical figures-and-tables view).
    shutil.copyfile(pdf, OUT_DIR / "paper.pdf")

    found = sum(len(v) for v in marks.values())
    print(f"built docs/index.html from {pdf.name}: {len(sections)} sections, "
          f"{found} highlights placed, {len(missing)} quotes unmatched")
    for idx, q in missing:
        print(f"  [warn] slide {idx}: quote not found: {q!r}…")


if __name__ == "__main__":
    main()
