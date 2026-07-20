import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pdftext.extraction import dictionary_output

from rag_core import DEFAULT_CHUNKS_PATH

HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NOISE_HEADINGS = {
    "references", "notes", "citations", "bibliography",
    "external links", "further reading", "see also",
}


def _font_size(span: dict) -> float:
    return round(span["font"]["size"], 1)


def _iter_spans(pages: list[dict]):
    for page in pages:
        for block in page["blocks"]:
            for line in block["lines"]:
                yield from line["spans"]


def body_font_size(pages: list[dict]) -> float:
    """The font size carrying the most characters is the body copy."""
    weights: Counter[float] = Counter()
    for span in _iter_spans(pages):
        weights[_font_size(span)] += len(span["text"])
    return weights.most_common(1)[0][0]


def heading_levels(pages: list[dict], body: float) -> dict[float, int]:
    """Anything set larger than the body copy is a heading. Rank those sizes
    largest-first so the biggest becomes h1, the next h2, and so on."""
    larger = sorted({_font_size(s) for s in _iter_spans(pages) if _font_size(s) > body}, reverse=True)
    return {size: min(level, 6) for level, size in enumerate(larger, start=1)}


def line_markdown(line: dict) -> str:
    """Join a line's spans into plain text.

    Span URLs are deliberately dropped. A link's anchor text already carries
    the meaning, while the href is dense, semantically empty tokens: emitting
    them costs ~44% of every chunk's tokens, which pushed ~45% of chunks past
    the embedding model's 256-token limit (silently truncating them), and left
    raw markdown URLs in the model's answers.
    """
    return "".join(s["text"] for s in line["spans"])


def block_markdown(block: dict, levels: dict[float, int]) -> str:
    sizes: Counter[float] = Counter()
    for line in block["lines"]:
        for span in line["spans"]:
            sizes[_font_size(span)] += len(span["text"])
    if not sizes:
        return ""

    lines = []
    for line in block["lines"]:
        joined = line_markdown(line).strip()
        if joined:
            lines.append(joined)
    if not lines:
        return ""

    text = " ".join(lines)
    level = levels.get(sizes.most_common(1)[0][0])
    return f"{'#' * level} {text}" if level else text


def extract_markdown(pdf_path: Path) -> str:
    """Read the PDF's own text layer and rebuild markdown from it.

    These are born-digital PDFs, so the text and its font metadata are already
    in the file; heading levels fall out of relative font size. Running OCR and
    layout models over them (as marker does) costs ~290s per article to
    rediscover structure that can be read directly in under a second.
    """
    pages = dictionary_output(str(pdf_path), keep_chars=False)
    levels = heading_levels(pages, body_font_size(pages))
    blocks = (block_markdown(b, levels) for page in pages for b in page["blocks"])
    return "\n\n".join(b for b in blocks if b)


def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


@dataclass
class Unit:
    text: str
    title: str | None
    heading: str | None
    is_reference: bool


def parse_heading(paragraph: str) -> tuple[int, str] | None:
    """Return (level, heading text) if this paragraph is a markdown heading."""
    match = HEADING_LINE_RE.match(paragraph)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def build_units(text: str, chunk_size: int) -> list[Unit]:
    """Walk the markdown paragraph by paragraph, tracking the article title
    (first H1) and the nearest heading, and flagging paragraphs that fall
    under a references/citations-style section."""
    title: str | None = None
    heading: str | None = None
    is_reference = False
    units: list[Unit] = []

    for paragraph in split_into_paragraphs(text):
        parsed = parse_heading(paragraph)
        if parsed:
            level, heading_text = parsed
            heading = heading_text
            if level == 1 and title is None:
                title = heading_text
            is_reference = heading_text.lower() in NOISE_HEADINGS
            units.append(Unit("#" * level + " " + heading_text, title, heading, is_reference))
            continue

        pieces = split_long_paragraph(paragraph, chunk_size) if len(paragraph) > chunk_size else [paragraph]
        units.extend(Unit(piece, title, heading, is_reference) for piece in pieces)

    return units


def pack_chunks(units: list[Unit], chunk_size: int, overlap: int) -> list[dict]:
    """Greedily pack units into ~chunk_size-character chunks. Overlap is
    carried forward at unit granularity (whole paragraphs/sentences), never
    a raw character slice, so a chunk boundary never lands mid-word."""
    chunks: list[dict] = []
    current: list[Unit] = []
    current_len = 0

    def flush() -> None:
        if not current:
            return
        body = "\n\n".join(u.text for u in current)
        title = current[0].title or ""
        heading = current[-1].heading or ""
        parts = [p for p in (title, heading) if p]
        if len(parts) == 2 and parts[0] == parts[1]:
            parts = parts[:1]
        breadcrumb = " > ".join(parts)
        text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
        chunks.append({
            "text": text,
            "title": title,
            "heading": heading,
            "is_reference": any(u.is_reference for u in current),
        })

    for unit in units:
        unit_len = len(unit.text)
        if current and current_len + unit_len + 2 > chunk_size:
            flush()
            carry: list[Unit] = []
            carry_len = 0
            for u in reversed(current):
                if carry_len + len(u.text) > overlap:
                    break
                carry.insert(0, u)
                carry_len += len(u.text) + 2
            current = carry
            current_len = carry_len
        current.append(unit)
        current_len += unit_len + 2

    flush()
    return chunks


def process_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> list[dict]:
    markdown = extract_markdown(pdf_path)
    units = build_units(markdown, chunk_size)
    chunks = pack_chunks(units, chunk_size, overlap)
    return [
        {
            "id": f"{pdf_path.stem}::{i}",
            "source": pdf_path.name,
            "chunk_index": i,
            "text": chunk["text"],
            "title": chunk["title"],
            "heading": chunk["heading"],
            "is_reference": chunk["is_reference"],
            "char_count": len(chunk["text"]),
        }
        for i, chunk in enumerate(chunks)
    ]


def already_processed_sources(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    sources: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sources.add(json.loads(line)["source"])
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDFs and chunk it for RAG")
    parser.add_argument("input_path", nargs="?", default="data", help="A single PDF, or a folder of PDFs to process")
    parser.add_argument("--output", default=DEFAULT_CHUNKS_PATH, help="Output JSONL path")
    # all-MiniLM-L6-v2 truncates at 256 tokens, silently: past that, a chunk is
    # still stored and shown to the LLM in full, but only its opening is
    # searchable. This prose runs ~4.5 chars/token, so 1000 chars puts the
    # median body chunk near 196 and keeps the bulk of them under the limit.
    # (What still overruns is mostly reference lists, which tokenize badly and
    # are excluded from retrieval anyway.)
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Characters of overlap carried into the next chunk")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    pdf_paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: a PDF's chunks are only ever written after all of them
    # are computed, so any source already present in the output is complete
    # and safe to skip (e.g. after an interrupted previous run).
    done_sources = already_processed_sources(output_path)
    remaining = [p for p in pdf_paths if p.name not in done_sources]
    if done_sources:
        print(f"Skipping {len(pdf_paths) - len(remaining)} already-processed PDF(s).")
    if not remaining:
        print("Nothing to do; all PDFs already processed.")
        return

    total_chunks = 0
    mode = "a" if done_sources else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for pdf_path in remaining:
            print(f"Processing {pdf_path.name}...")
            records = process_pdf(pdf_path, args.chunk_size, args.overlap)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            total_chunks += len(records)

    print(f"Wrote {total_chunks} chunks from {len(remaining)} PDFs to {output_path}")


if __name__ == "__main__":
    main()
