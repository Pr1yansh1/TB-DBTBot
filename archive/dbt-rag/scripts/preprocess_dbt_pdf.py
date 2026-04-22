#!/usr/bin/env python3
"""
Preprocess DBT skills manual PDF into a simple RAG-friendly structure:

processed/
  overview/
    book_overview.txt
    fichas/
      ficha_general_01.txt
      ficha_general_01a.txt
      ...
  mindfulness/
    intro.txt
    fichas/
      ficha_01.txt
      ficha_01a.txt
      ...
  interpersonal_effectiveness/
    intro.txt
    fichas/
      ficha_01.txt
      ...
  emotion_regulation/
    intro.txt
    fichas/
      ficha_01.txt
      ...
  distress_tolerance/
    intro.txt
    fichas/
      ficha_01.txt
      ...
  index.json

- Extracts ONLY "Fichas" (skips "Hojas de trabajo").
- Also writes "intro.txt" per module (text between module heading and first ficha).
- Writes overview/book_overview.txt (front matter + general section, excluding most TOC noise).
- Uses PyMuPDF (fitz) and regex + heuristics to handle:
  - hyphenation across line breaks
  - broken line wraps
  - duplicated headers/footers/page numbers
  - ficha titles split onto the next line
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ----------------------------
# Config / Regex
# ----------------------------

MODULES = {
    # canonical_slug: (display name fragments we might see)
    "mindfulness": [
        r"mindfulness",
    ],
    "interpersonal_effectiveness": [
        r"efectividad\s+interpersonal",
    ],
    "emotion_regulation": [
        r"regulación\s+emocional",
    ],
    "distress_tolerance": [
        r"tolerancia\s+al\s+malestar",
    ],
    "general": [
        r"general(?:es)?",
    ],
}

# Headings that often mark module starts
MODULE_HEADING_RES: Dict[str, re.Pattern] = {
    slug: re.compile(rf"^\s*Habilidades\s+de\s+({alt})\s*$", re.IGNORECASE)
    for slug, alts in MODULES.items()
    for alt in alts
}
# Special "general skills" heading in TOC/front matter
GENERAL_SKILLS_HEADING_RE = re.compile(
    r"^\s*Habilidades\s+generales\s*:\s*orientación\s+y\s+análisis\s+conductual\s*$",
    re.IGNORECASE,
)

# "Fichas de <module>" headings
FICHAS_SECTION_RE = re.compile(r"^\s*Fichas\s+de\s+.+\s*$", re.IGNORECASE)

# Worksheet headings (we skip these docs entirely)
HOJA_RE = re.compile(r"^\s*Hoja(?:s)?\s+de\s+trabajo\b", re.IGNORECASE)

# Ficha header line
# Examples:
#  - "Ficha general 1: Objetivos ..."
#  - "Ficha general 1A. Opciones ..."
#  - "Ficha de mindfulness 4c: Ideas ..."
#  - "Ficha de tolerancia al malestar 11b: Práctica ..."
FICHA_HEADER_RE = re.compile(
    r"^\s*Ficha\s+"
    r"(?:(?:de)\s+)?"
    r"(?P<category>general|mindfulness|efectividad\s+interpersonal|regulación\s+emocional|tolerancia\s+al\s+malestar)\s+"
    r"(?P<num>\d+)\s*(?P<suffix>[A-Za-z])?\s*"
    r"(?P<punct>[:\.])?\s*"
    r"(?P<title>.*)\s*$",
    re.IGNORECASE,
)

# Lines that are almost certainly just page numbers
PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Common footer/header junk patterns (tune as needed)
LIKERT_RE = re.compile(r"\b1\s+2\s+3\s+4\s+5\b")
UNDERLINE_HEAVY_RE = re.compile(r"^[\s_]{6,}$")

# TOC-ish lines
TOC_START_RE = re.compile(r"^\s*Índice\s*$", re.IGNORECASE)
TOC_EXIT_HINT_RE = re.compile(
    r"^\s*(Introducción\s+a\s+este\s+libro|Cómo\s+está\s+organizado\s+este\s+libro|Habilidades\s+generales)\b",
    re.IGNORECASE,
)


# ----------------------------
# Data models
# ----------------------------

@dataclass
class DocMeta:
    doc_id: str
    module: str                 # one of MODULES slugs, including "general"
    number: str                 # like "1", "1a", "11b"
    title: str
    start_page: int             # 1-indexed (human)
    end_page: int               # 1-indexed (human)
    out_path: str               # relative to processed/


# ----------------------------
# Text normalization helpers
# ----------------------------

def dehyphenate(text: str) -> str:
    # Join words split by hyphen at line break: "adapta-\ntiva" => "adaptativa"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def normalize_whitespace(text: str) -> str:
    # Normalize weird whitespace without destroying paragraph breaks too much
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Section headings are often short and not ending with punctuation.
    if len(s) <= 70 and not s.endswith((".", ":", ";")):
        # Many headings are Title Case or ALL CAPS-ish; we keep it simple:
        if sum(c.isalpha() for c in s) >= 5 and (s.isupper() or s.istitle()):
            return True
    # Module headings / "Fichas de ..." / worksheets headings
    if GENERAL_SKILLS_HEADING_RE.match(s):
        return True
    if any(p.match(s) for p in MODULE_HEADING_RES.values()):
        return True
    if FICHAS_SECTION_RE.match(s) or HOJA_RE.match(s):
        return True
    return False

def should_drop_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PAGE_NUM_ONLY_RE.match(s):
        return True
    if UNDERLINE_HEAVY_RE.match(s):
        return True
    # Likert scales and repeated rating lines are high-noise for fichas too
    if LIKERT_RE.search(s) and ("Nada" in s or "Algo" in s or "Muy" in s):
        return True
    # Common form fields
    if re.search(r"\b(Nombre|Fecha)\s*:\s*$", s, re.IGNORECASE):
        return True
    return False

def join_wrapped_lines(lines: List[str]) -> List[str]:
    """
    Heuristic line joining:
    - Preserve blank lines as paragraph breaks.
    - Join a line with the next if it's a soft wrap (no punctuation, next line starts lowercase).
    - Do NOT join if current/next looks like a heading or a ficha header.
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        if not cur.strip():
            out.append("")
            i += 1
            continue

        # Skip junk lines early
        if should_drop_line(cur):
            i += 1
            continue

        # If this is a ficha header, keep as-is
        if FICHA_HEADER_RE.match(cur) or looks_like_heading(cur):
            out.append(cur.strip())
            i += 1
            continue

        # Try merge with next if it looks like a hard wrap
        if i + 1 < len(lines):
            nxt = lines[i + 1].rstrip()
            if nxt.strip() and not should_drop_line(nxt):
                if not FICHA_HEADER_RE.match(nxt) and not looks_like_heading(nxt):
                    cur_ends = cur.strip()[-1:]
                    nxt_starts = nxt.strip()[:1]
                    # If cur doesn't end with sentence-ish punctuation and next starts lowercase, likely wrap
                    if cur_ends not in ".:;!?" and nxt_starts.islower():
                        merged = cur.strip() + " " + nxt.strip()
                        out.append(merged)
                        i += 2
                        continue

        out.append(cur.strip())
        i += 1

    # Collapse multiple blank lines
    cleaned: List[str] = []
    prev_blank = False
    for l in out:
        is_blank = (l.strip() == "")
        if is_blank and prev_blank:
            continue
        cleaned.append(l)
        prev_blank = is_blank
    return cleaned


# ----------------------------
# Extraction
# ----------------------------

def extract_pages(pdf_path: Path) -> List[str]:
    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for p in range(doc.page_count):
        text = doc.load_page(p).get_text("text")
        text = dehyphenate(text)
        text = normalize_whitespace(text)
        pages.append(text)
    doc.close()
    return pages

def classify_category_to_module(category_raw: str) -> str:
    c = category_raw.strip().lower()
    if c.startswith("general"):
        return "general"
    if "mindfulness" in c:
        return "mindfulness"
    if "efectividad" in c:
        return "interpersonal_effectiveness"
    if "regulación" in c or "regulacion" in c:
        return "emotion_regulation"
    if "tolerancia" in c:
        return "distress_tolerance"
    return "general"

def normalize_ficha_number(num: str, suffix: Optional[str]) -> str:
    if suffix:
        return f"{num}{suffix.lower()}"
    return num

def safe_filename(s: str) -> str:
    # Minimal safe filename mapping
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s

def build_line_stream(pages: List[str]) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1indexed, line_text) after normalization/joining.
    """
    stream: List[Tuple[int, str]] = []
    for i, page_text in enumerate(pages):
        raw_lines = page_text.split("\n")
        raw_lines = [l.rstrip("\n") for l in raw_lines]
        joined = join_wrapped_lines(raw_lines)
        for line in joined:
            if line.strip() == "":
                stream.append((i + 1, ""))
            else:
                stream.append((i + 1, line))
    return stream

def extract_fichas(stream: List[Tuple[int, str]]) -> Tuple[List[DocMeta], Dict[str, str], Dict[str, Tuple[int,int,int,int]]]:
    """
    Scans the line stream to segment ficha docs.

    Returns:
      - metas: list of DocMeta
      - docs: doc_id -> text
      - spans: doc_id -> (start_idx, end_idx, start_page, end_page)
    """
    metas: List[DocMeta] = []
    docs: Dict[str, str] = {}
    spans: Dict[str, Tuple[int,int,int,int]] = {}

    current_id: Optional[str] = None
    current_lines: List[str] = []
    current_meta: Optional[Tuple[str, str, str, int]] = None  # (module, number, title, start_page)
    start_idx: Optional[int] = None
    start_page: Optional[int] = None

    def flush(end_idx: int, end_page: int):
        nonlocal current_id, current_lines, current_meta, start_idx, start_page
        if not current_id or not current_meta:
            return
        module, number, title, spage = current_meta
        # Clean up extra blank lines
        text = "\n".join(current_lines).strip()
        if not text:
            # Drop empty docs
            current_id = None
            current_lines = []
            current_meta = None
            start_idx = None
            start_page = None
            return

        docs[current_id] = text
        spans[current_id] = (start_idx if start_idx is not None else 0, end_idx, spage, end_page)
        metas.append(
            DocMeta(
                doc_id=current_id,
                module=module,
                number=number,
                title=title,
                start_page=spage,
                end_page=end_page,
                out_path="",  # filled later
            )
        )
        current_id = None
        current_lines = []
        current_meta = None
        start_idx = None
        start_page = None

    i = 0
    while i < len(stream):
        pageno, line = stream[i]
        m = FICHA_HEADER_RE.match(line)
        if m:
            # If it's actually a worksheet line masquerading, skip (extra safety)
            if HOJA_RE.match(line):
                i += 1
                continue

            # Start of new ficha: flush previous
            if current_id is not None:
                flush(end_idx=i - 1, end_page=stream[i - 1][0])

            category = (m.group("category") or "").strip()
            num = (m.group("num") or "").strip()
            suffix = m.group("suffix")
            title = (m.group("title") or "").strip()

            module = classify_category_to_module(category)
            number = normalize_ficha_number(num, suffix)

            # Title may be on next line if empty / truncated
            if not title:
                j = i + 1
                while j < len(stream):
                    _, nxt = stream[j]
                    if nxt.strip() == "":
                        j += 1
                        continue
                    if FICHA_HEADER_RE.match(nxt) or looks_like_heading(nxt) or HOJA_RE.match(nxt):
                        break
                    title = nxt.strip()
                    break

            title = title.strip() if title else f"Ficha {number}"

            current_id = f"{module}.ficha.{number}"
            current_lines = [line.strip()]
            current_meta = (module, number, title, pageno)
            start_idx = i
            start_page = pageno

            i += 1
            continue

        # If inside a ficha, keep collecting lines until next ficha header
        if current_id is not None:
            # Drop obvious worksheet section markers inside text (rare but helps)
            if HOJA_RE.match(line) and looks_like_heading(line):
                i += 1
                continue
            if should_drop_line(line):
                i += 1
                continue
            current_lines.append(line)
        i += 1

    # Flush last
    if current_id is not None:
        flush(end_idx=len(stream) - 1, end_page=stream[-1][0])

    return metas, docs, spans

def find_module_heading_positions(stream: List[Tuple[int, str]]) -> Dict[str, int]:
    """
    Returns module_slug -> first line index where the module heading appears.
    Note: general skills heading is treated as "general".
    """
    pos: Dict[str, int] = {}
    for i, (_p, line) in enumerate(stream):
        s = line.strip()
        if not s:
            continue
        if "general" not in pos and GENERAL_SKILLS_HEADING_RE.match(s):
            pos["general"] = i
            continue
        for slug, pat in MODULE_HEADING_RES.items():
            if slug in pos:
                continue
            if pat.match(s):
                pos[slug] = i
    return pos

def extract_module_intro(
    stream: List[Tuple[int, str]],
    module_slug: str,
    module_pos: Dict[str, int],
    ficha_spans: Dict[str, Tuple[int,int,int,int]],
) -> str:
    """
    Intro = text from module heading until first ficha of that module.
    Filters out TOC-like lines and "Fichas de ..." / "Hojas de trabajo ..." headings.
    """
    if module_slug not in module_pos:
        return ""

    start_i = module_pos[module_slug] + 1

    # Find first ficha start index for this module
    first_ficha_start: Optional[int] = None
    for doc_id, (si, _ei, _sp, _ep) in ficha_spans.items():
        if doc_id.startswith(f"{module_slug}.ficha."):
            if first_ficha_start is None or si < first_ficha_start:
                first_ficha_start = si

    end_i = first_ficha_start if first_ficha_start is not None else len(stream)

    lines: List[str] = []
    for i in range(start_i, end_i):
        _p, line = stream[i]
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if should_drop_line(s):
            continue
        if FICHAS_SECTION_RE.match(s):
            continue
        if HOJA_RE.match(s):
            continue
        # Skip list-y index noise inside intro
        if s.lower().startswith("ficha ") or s.lower().startswith("hoja de trabajo"):
            continue
        lines.append(s)

    # Clean blank runs
    cleaned: List[str] = []
    prev_blank = False
    for l in lines:
        b = (l.strip() == "")
        if b and prev_blank:
            continue
        cleaned.append(l)
        prev_blank = b

    return "\n".join(cleaned).strip()

def build_book_overview(stream: List[Tuple[int, str]], module_pos: Dict[str, int]) -> str:
    """
    Book overview = from start up to first of the 4 core module headings
    (mindfulness/interpersonal/emotion_regulation/distress_tolerance),
    with a heuristic to skip most TOC listing noise.
    """
    # Find earliest core module heading
    core = [m for m in ["mindfulness", "interpersonal_effectiveness", "emotion_regulation", "distress_tolerance"] if m in module_pos]
    cut = min((module_pos[m] for m in core), default=len(stream))

    # Heuristic: skip TOC region if we see "Índice" until an exit hint
    in_toc = False
    out_lines: List[str] = []
    for i in range(0, cut):
        _p, line = stream[i]
        s = line.strip()
        if not s:
            out_lines.append("")
            continue

        if TOC_START_RE.match(s):
            in_toc = True
            continue
        if in_toc and TOC_EXIT_HINT_RE.match(s):
            in_toc = False
            # include the exit hint line
            out_lines.append(s)
            continue
        if in_toc:
            # drop most TOC lines
            continue

        if should_drop_line(s):
            continue

        out_lines.append(s)

    # Collapse blank lines
    cleaned: List[str] = []
    prev_blank = False
    for l in out_lines:
        b = (l.strip() == "")
        if b and prev_blank:
            continue
        cleaned.append(l)
        prev_blank = b

    return "\n".join(cleaned).strip()


# ----------------------------
# Writing outputs
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.strip() + "\n", encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the DBT manual PDF")
    ap.add_argument("--out", default="DBT-RAG-documents/processed", help="Output directory")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = extract_pages(pdf_path)
    stream = build_line_stream(pages)

    module_pos = find_module_heading_positions(stream)

    metas, docs, spans = extract_fichas(stream)

    # Prepare output directories
    ensure_dir(out_dir)
    ensure_dir(out_dir / "overview" / "fichas")

    for m in ["mindfulness", "interpersonal_effectiveness", "emotion_regulation", "distress_tolerance"]:
        ensure_dir(out_dir / m / "fichas")

    # Write module intros (only for the 4 core modules)
    for m in ["mindfulness", "interpersonal_effectiveness", "emotion_regulation", "distress_tolerance"]:
        intro = extract_module_intro(stream, m, module_pos, spans)
        if intro:
            write_text(out_dir / m / "intro.txt", intro)
        else:
            # Still create file so structure is predictable
            write_text(out_dir / m / "intro.txt", "")

    # Write overview file
    overview_text = build_book_overview(stream, module_pos)
    write_text(out_dir / "overview" / "book_overview.txt", overview_text)

    # Write fichas
    # - general fichas go to overview/fichas/
    # - module fichas go to module/fichas/
    index_payload = {
        "source": {"pdf": pdf_path.name},
        "docs": [],
    }

    for meta in sorted(metas, key=lambda m: (m.module, m.number)):
        text = docs.get(meta.doc_id, "").strip()
        if not text:
            continue

        if meta.module == "general":
            rel = Path("overview") / "fichas" / f"ficha_general_{safe_filename(meta.number).zfill(2) if meta.number.isdigit() else safe_filename(meta.number)}.txt"
        else:
            rel = Path(meta.module) / "fichas" / f"ficha_{safe_filename(meta.number).zfill(2) if meta.number.isdigit() else safe_filename(meta.number)}.txt"

        abs_path = out_dir / rel
        write_text(abs_path, text)

        meta.out_path = str(rel.as_posix())
        index_payload["docs"].append(asdict(meta))

    # Write index.json
    (out_dir / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    counts = {}
    for d in index_payload["docs"]:
        counts[d["module"]] = counts.get(d["module"], 0) + 1

    print(f"✅ PDF: {pdf_path.name}")
    print(f"✅ Output: {out_dir}")
    print("✅ Fichas written:")
    for k in ["general", "mindfulness", "interpersonal_effectiveness", "emotion_regulation", "distress_tolerance"]:
        if k in counts:
            print(f"   - {k}: {counts[k]}")
    print(f"✅ Index: {out_dir / 'index.json'}")
    print(f"✅ Overview: {out_dir / 'overview' / 'book_overview.txt'}")
    for m in ["mindfulness", "interpersonal_effectiveness", "emotion_regulation", "distress_tolerance"]:
        print(f"✅ Intro: {out_dir / m / 'intro.txt'}")


if __name__ == "__main__":
    main()

