# scripts/preprocess_dbt_pdf.py
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pypdf import PdfReader


# ----------------------------
# Config / constants (V1)
# ----------------------------

# Major section headings (module context). Keep conservative, but allow more variants.
SECTION_HEADING_RE = re.compile(
    r"^(HABILIDADES\s+DE\s+MINDFULNESS|HABILIDADES\s+DE\s+EFECTIVIDAD\s+INTERPERSONAL|"
    r"HABILIDADES\s+DE\s+REGULACI[ÓO]N\s+EMOCIONAL|HABILIDADES\s+DE\s+TOLERANCIA\s+AL\s+MALESTAR|"
    r"HABILIDADES\s+GENERALES.*|FICHAS\s+DE\s+MINDFULNESS|FICHAS\s+DE\s+TOLERANCIA\s+AL\s+MALESTAR|"
    r"FICHAS\s+DE\s+REGULACI[ÓO]N\s+EMOCIONAL|FICHAS\s+DE\s+EFECTIVIDAD\s+INTERPERSONAL|"
    r"HOJAS\s+DE\s+TRABAJO\s+DE\s+MINDFULNESS|HOJAS\s+DE\s+TRABAJO\s+DE\s+TOLERANCIA\s+AL\s+MALESTAR|"
    r"HOJAS\s+DE\s+TRABAJO\s+DE\s+REGULACI[ÓO]N\s+EMOCIONAL|HOJAS\s+DE\s+TRABAJO\s+DE\s+EFECTIVIDAD\s+INTERPERSONAL)\s*$",
    flags=re.IGNORECASE,
)

# Unit starts. Must tolerate:
#  - trailing parenthetical references: "(Fichas ...)"
#  - trailing subtitles on the same line
#  - line breaks between pieces of the header (handled by lookahead logic)
#
# We match a "header core" with an optional suffix we ignore.
UNIT_HEADER_CORE_RE = re.compile(
    r"^(?P<prefix>FICHA(?:\s+GENERAL)?|FICHA\s+DE|HOJA\s+DE\s+TRABAJO(?:\s+GENERAL)?|HOJA\s+DE\s+TRABAJO\s+DE)"
    r"\s+"
    r"(?P<category>[A-ZÁÉÍÓÚÜÑ0-9\s]+?)"
    r"\s+"
    r"(?P<number>[0-9]{1,3}(?:[A-Z])?(?:[a-z])?)"
    r"(?P<suffix>\s*(?:\(|—|-).*)?$",
    flags=re.IGNORECASE,
)

# Common noise: single page numbers, repeated headers/footers.
PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Keywords for module inference.
MODULE_KEYWORDS = [
    ("mindfulness", ["MINDFULNESS", "MENTE SABIA", "OBSERVAR", "DESCRIBIR", "PARTICIPAR"]),
    ("interpersonal_effectiveness", ["EFECTIVIDAD INTERPERSONAL", "DEAR MAN", "AVES", "VIDA", "VALIDACIÓN"]),
    ("emotion_regulation", ["REGULACIÓN EMOCIONAL", "ACCIÓN OPUESTA", "VERIFICAR LOS HECHOS", "EMOCIONES"]),
    ("distress_tolerance", ["TOLERANCIA AL MALESTAR", "STOP", "TIP", "ACEPTACIÓN RADICAL", "MEJORAR EL MOMENTO"]),
    ("general", ["HABILIDADES GENERALES", "ANÁLISIS CONDUCTUAL", "TEORÍA BIOSOCIAL"]),
]

# Skill tags (help retrieval be precise).
SKILL_TAG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("STOP", re.compile(r"\bSTOP\b", re.IGNORECASE)),
    ("TIP", re.compile(r"\bTIP\b", re.IGNORECASE)),
    ("DEAR_MAN", re.compile(r"\bDEAR\s+MAN\b", re.IGNORECASE)),
    ("CHECK_THE_FACTS", re.compile(r"VERIFICAR\s+LOS\s+HECHOS", re.IGNORECASE)),
    ("OPPOSITE_ACTION", re.compile(r"ACCIÓN\s+OPUESTA", re.IGNORECASE)),
    ("RADICAL_ACCEPTANCE", re.compile(r"ACEPTACIÓN\s+RADICAL", re.IGNORECASE)),
    ("WISE_MIND", re.compile(r"MENTE\s+SABIA", re.IGNORECASE)),
    ("VALIDATION", re.compile(r"\bVALIDACIÓN\b", re.IGNORECASE)),
    ("SELF_SOOTHE", re.compile(r"CALMA(?:RTE)?\s+CON\s+LOS\s+CINCO\s+SENTIDOS", re.IGNORECASE)),
    ("DISTRACT_ACCEPTS", re.compile(r"\bACEPTAS\b", re.IGNORECASE)),
]


@dataclass(frozen=True)
class Unit:
    unit_id: str
    unit_type: str  # FICHA or HOJA DE TRABAJO
    category: str
    number: str
    title_line: str
    major_section: str
    module: str
    skill_tag: str
    start_page: int  # 1-indexed
    end_page: int    # 1-indexed, inclusive
    source_pdf: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    unit_id: str
    module: str
    skill_tag: str
    title_line: str
    source_pdf: str
    start_page: int
    end_page: int
    text: str


# ----------------------------
# Text normalization utilities
# ----------------------------

def _strip_weird_spaces(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _is_probable_header_footer(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if PAGE_NUM_RE.match(l):
        return True
    return False


_FORM_FIELD_RE = re.compile(
    r"(^\s*(fecha|nombre|firma|día|lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b)|"
    r"(_{3,})|"
    r"(/ ?/ ?/)|"
    r"(\b\d{1,3}\s+\d{1,3}\s+\d{1,3}\b)",  # e.g., "1 2 3 4 5" scales, but keep only if not too dominant
    flags=re.IGNORECASE,
)


def _is_low_value_form_line(line: str) -> bool:
    """
    Drop lines that are basically worksheet boilerplate / fill-in templates.
    Keep it deterministic and conservative.
    """
    l = line.strip()
    if not l:
        return False

    # lots of underscores or slashes placeholders
    if re.search(r"_{5,}", l):
        return True
    if re.search(r"(?:/ ?){6,}", l):  # "/ / / / / /"
        return True

    # lines that are mostly punctuation + blanks
    alnum = sum(ch.isalnum() for ch in l)
    if alnum <= 3 and len(l) >= 8:
        return True

    # very common field prompts that add little semantic value
    if re.match(r"^(fecha\s+de\s+|nombre\s*:|fecha\s*:)", l, flags=re.IGNORECASE):
        return True

    return False


def _normalize_lines(raw_text: str) -> List[str]:
    out: List[str] = []
    for line in raw_text.splitlines():
        line = _strip_weird_spaces(line)
        if not line:
            out.append("")
            continue
        if _is_probable_header_footer(line):
            continue
        # Keep form lines out (V1). This makes retrieval much cleaner.
        if _is_low_value_form_line(line):
            continue
        out.append(line)

    # collapse blank runs to max 2
    collapsed: List[str] = []
    blank_run = 0
    for l in out:
        if l == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append("")
        else:
            blank_run = 0
            collapsed.append(l)
    return collapsed


def _join_wrapped_lines(lines: List[str]) -> str:
    paragraphs: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            paragraphs.append(" ".join(buf).strip())
            buf = []

    for line in lines:
        if line == "":
            flush()
            continue
        if buf and buf[-1].endswith("-") and len(buf[-1]) > 1:
            buf[-1] = buf[-1][:-1] + line
        else:
            buf.append(line)

    flush()
    return "\n\n".join(p for p in paragraphs if p.strip())


def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unit"


# ----------------------------
# PDF extraction
# ----------------------------

def extract_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return pages


def detect_module(major_section: str, category: str, title_line: str, unit_text: str) -> str:
    hay = " ".join([major_section, category, title_line, unit_text[:2000]]).upper()
    for module, keys in MODULE_KEYWORDS:
        for k in keys:
            if k.upper() in hay:
                return module
    return "general"


def detect_skill_tag(title_line: str, unit_text: str) -> str:
    hay = (title_line + "\n" + unit_text[:3000])
    for tag, pat in SKILL_TAG_PATTERNS:
        if pat.search(hay):
            return tag
    return _slugify(title_line)[:60].upper()


def iter_lines_by_page(pages: List[str]) -> Iterable[Tuple[int, List[str]]]:
    for idx, page_text in enumerate(pages, start=1):
        lines = _normalize_lines(page_text)
        yield idx, lines


# ----------------------------
# Header detection (fix)
# ----------------------------

def _try_parse_unit_header(lines: List[str], i: int) -> Optional[Tuple[int, str, str, str, str]]:
    """
    Try to parse a unit header starting at lines[i].
    Returns (consumed_lines, unit_type, category, number, raw_title_line)
    """
    if i >= len(lines):
        return None

    l0 = lines[i].strip()
    if not l0:
        return None

    # Candidate combinations: header might be split across lines.
    candidates: List[Tuple[int, str]] = [(1, l0)]

    if i + 1 < len(lines) and lines[i + 1].strip():
        candidates.append((2, f"{l0} {lines[i+1].strip()}"))

    if i + 2 < len(lines) and lines[i + 1].strip() and lines[i + 2].strip():
        candidates.append((3, f"{l0} {lines[i+1].strip()} {lines[i+2].strip()}"))

    for consumed, cand in candidates:
        # Normalize multiple spaces
        cand_norm = re.sub(r"\s+", " ", cand).strip()

        # Some PDFs include "HOJA DE TRABAJO ... 13 (Fichas ...)" on one line
        # We accept suffix and ignore it.
        m = UNIT_HEADER_CORE_RE.match(cand_norm)
        if not m:
            continue

        prefix = m.group("prefix").strip().upper()
        category = m.group("category").strip()
        number = m.group("number").strip()

        # Clean category: remove stray parentheses if they got absorbed
        category = re.sub(r"\s*\(.*$", "", category).strip()

        if prefix.startswith("HOJA"):
            unit_type = "HOJA DE TRABAJO"
        else:
            unit_type = "FICHA"

        raw_title = f"{unit_type} {category} {number}".strip()
        return (consumed, unit_type, category, number, raw_title)

    return None


def _maybe_grab_subtitle(lines: List[str], start_i: int) -> str:
    """
    Grab one descriptive subtitle line after a header, if present.
    Deterministic rules:
      - Skip blank lines
      - Skip parenthetical reference lines "(Hojas de trabajo ...)"
      - Stop if next is another header
      - Take first remaining line as subtitle if it looks like prose (not all-caps heading)
    """
    j = start_i
    while j < len(lines):
        nxt = lines[j].strip()
        if not nxt:
            j += 1
            continue
        if nxt.startswith("(") and nxt.endswith(")"):
            j += 1
            continue
        # stop if a new unit header begins
        if _try_parse_unit_header(lines, j) is not None:
            return ""
        # avoid taking another major section heading as subtitle
        if SECTION_HEADING_RE.match(nxt):
            return ""
        # avoid all-caps short headings (often noise)
        if len(nxt) <= 80 and nxt.upper() == nxt and sum(ch.isalpha() for ch in nxt) >= 8:
            return ""
        return nxt
    return ""


# ----------------------------
# Unit parsing
# ----------------------------

def parse_units(pages: List[str], source_pdf_name: str) -> List[Unit]:
    """
    Parse units per page (not a fully flattened stream), so we can handle
    multi-line headers within a page deterministically.
    """
    units_raw: List[dict] = []
    current_section = "Unknown section"

    current_unit: Optional[dict] = None
    current_start_page = 1
    current_end_page = 1

    for page_no, page_lines in iter_lines_by_page(pages):
        i = 0
        while i < len(page_lines):
            line = page_lines[i].strip()

            # Update major section context
            if line and SECTION_HEADING_RE.match(line):
                current_section = line.strip()
                i += 1
                continue

            hdr = _try_parse_unit_header(page_lines, i)
            if hdr is not None:
                consumed, unit_type, category, number, raw_title = hdr

                # close previous unit
                if current_unit is not None:
                    current_unit["end_page"] = current_end_page
                    units_raw.append(current_unit)

                subtitle = _maybe_grab_subtitle(page_lines, i + consumed)
                title_line = raw_title if not subtitle else f"{raw_title} — {subtitle}"

                current_unit = {
                    "unit_type": unit_type,
                    "category": category,
                    "number": number,
                    "title_line": title_line,
                    "major_section": current_section,
                    "start_page": page_no,
                    "end_page": page_no,
                    "text_lines": [],
                }
                current_start_page = page_no
                current_end_page = page_no

                i += consumed
                continue

            # regular content
            if current_unit is not None:
                current_unit["text_lines"].append(page_lines[i])
                current_end_page = page_no

            i += 1

    # close last
    if current_unit is not None:
        current_unit["end_page"] = current_end_page
        units_raw.append(current_unit)

    # materialize
    units: List[Unit] = []
    for idx, u in enumerate(units_raw, start=1):
        unit_text = _join_wrapped_lines(u["text_lines"]).strip()

        module = detect_module(u["major_section"], u["category"], u["title_line"], unit_text)
        skill_tag = detect_skill_tag(u["title_line"], unit_text)

        units.append(
            Unit(
                unit_id=f"{idx:05d}",
                unit_type=u["unit_type"],
                category=u["category"],
                number=u["number"],
                title_line=u["title_line"],
                major_section=u["major_section"],
                module=module,
                skill_tag=skill_tag,
                start_page=int(u["start_page"]),
                end_page=int(u["end_page"]),
                source_pdf=source_pdf_name,
                text=unit_text,
            )
        )

    return units


# ----------------------------
# Chunking (V1)
# ----------------------------

def _split_preserve_lists(text: str) -> List[str]:
    blocks: List[str] = []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    for p in paras:
        if re.search(r"(^|\n)\d+\.\s", p) or "••" in p:
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            buf: List[str] = []
            for ln in lines:
                is_new_item = bool(re.match(r"^\d+\.\s+", ln)) or ln.startswith("••")
                if is_new_item and buf:
                    blocks.append("\n".join(buf).strip())
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                blocks.append("\n".join(buf).strip())
        else:
            blocks.append(p)

    return blocks


def make_chunks(
    unit: Unit,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Deterministic chunking:
      - split into stable blocks (paragraphs / list items)
      - pack blocks up to max_chars
      - overlap by reusing last blocks until overlap_chars target is met (block-based, not raw char slice)
    """
    blocks = _split_preserve_lists(unit.text)
    chunks: List[Chunk] = []

    current_blocks: List[str] = []
    current_len = 0

    def finalize(blocks_for_chunk: List[str]):
        text = "\n\n".join(blocks_for_chunk).strip()
        chunk_id = f"{unit.unit_id}-{len(chunks)+1:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                unit_id=unit.unit_id,
                module=unit.module,
                skill_tag=unit.skill_tag,
                title_line=unit.title_line,
                source_pdf=unit.source_pdf,
                start_page=unit.start_page,
                end_page=unit.end_page,
                text=text,
            )
        )
        return text

    def overlap_from(prev_blocks: List[str]) -> List[str]:
        if overlap_chars <= 0:
            return []
        out: List[str] = []
        total = 0
        # take from the end, whole blocks only
        for b in reversed(prev_blocks):
            out.insert(0, b)
            total += len(b) + 2
            if total >= overlap_chars:
                break
        return out

    for b in blocks:
        b = b.strip()
        if not b:
            continue

        if not current_blocks:
            current_blocks = [b]
            current_len = len(b)
            continue

        if current_len + len(b) + 4 <= max_chars:
            current_blocks.append(b)
            current_len += len(b) + 4
            continue

        # finalize current chunk
        prev_text = finalize(current_blocks)

        # start next with overlap blocks + new block
        ov = overlap_from(current_blocks)
        current_blocks = ov + [b] if ov else [b]
        current_len = sum(len(x) for x in current_blocks) + 4 * max(0, len(current_blocks) - 1)

    if current_blocks:
        finalize(current_blocks)

    return chunks


# ----------------------------
# IO
# ----------------------------

def write_units(units: List[Unit], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_units.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for u in units:
            fname = f"{u.unit_id}_{_slugify(u.title_line)[:90]}.txt"
            path = out_dir / fname
            path.write_text(u.text.strip() + "\n", encoding="utf-8")

            rec = asdict(u).copy()
            rec["path"] = str(path.resolve())
            rec.pop("text", None)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return manifest_path


def write_chunks(chunks: List[Chunk], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_chunks.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for c in chunks:
            fname = f"{c.chunk_id}_{_slugify(c.title_line)[:70]}.txt"
            path = out_dir / fname

            header = (
                f"TITLE: {c.title_line}\n"
                f"MODULE: {c.module}\n"
                f"SKILL_TAG: {c.skill_tag}\n"
                f"SOURCE_PDF: {c.source_pdf}\n"
                f"PAGES: {c.start_page}-{c.end_page}\n"
                f"CHUNK_ID: {c.chunk_id}\n"
                f"UNIT_ID: {c.unit_id}\n"
                "\n"
            )
            path.write_text(header + c.text.strip() + "\n", encoding="utf-8")

            rec = asdict(c).copy()
            rec["path"] = str(path.resolve())
            rec.pop("text", None)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return manifest_path


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=str, help="Path to DBT manual PDF")
    ap.add_argument(
        "--out-root",
        default="DBT-RAG-documents/processed",
        type=str,
        help="Output root directory (relative to current working dir by default)",
    )
    ap.add_argument("--max-chars", default=1200, type=int)
    ap.add_argument("--overlap-chars", default=200, type=int)
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()

    # IMPORTANT: out-root is relative to *current directory*
    out_root = Path(args.out_root).expanduser()
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()

    pages = extract_pages(pdf_path)
    units = parse_units(pages, source_pdf_name=pdf_path.name)

    units_dir = out_root / "units"
    chunks_dir = out_root / "chunks"

    units_manifest = write_units(units, units_dir)

    all_chunks: List[Chunk] = []
    for u in units:
        all_chunks.extend(make_chunks(u, max_chars=args.max_chars, overlap_chars=args.overlap_chars))

    chunks_manifest = write_chunks(all_chunks, chunks_dir)

    print(f"✅ Extracted {len(units)} units")
    print(f"✅ Wrote units to: {units_dir}")
    print(f"✅ Units manifest: {units_manifest}")
    print(f"✅ Produced {len(all_chunks)} chunks")
    print(f"✅ Wrote chunks to: {chunks_dir}")
    print(f"✅ Chunks manifest: {chunks_manifest}")

    if len(units) < 50:
        print(
            "⚠️  Very few units detected. This usually means header detection is missing the PDF’s real header patterns.\n"
            "    Open a few extracted pages and check whether headers are broken across lines or include extra suffix text."
        )


if __name__ == "__main__":
    main()

