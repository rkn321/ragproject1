import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
NOISE_HEADINGS = {
    "references", "notes", "citations", "bibliography",
    "external links", "further reading", "see also",
}


def extract_markdown(pdf_path: Path, converter: PdfConverter) -> str:
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def split_into_paragraphs(text: str) -> list[str]:
    text = IMAGE_RE.sub("", text)
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
    """Return (level, clean heading text) if this paragraph is a markdown
    heading line. Marker sometimes emits headings like
    '# <span id="page-4-3"></span>**References**' (an HTML anchor before the
    bold text), so strip HTML tags and markdown emphasis markers rather than
    relying on a single regex to capture the clean text directly."""
    match = HEADING_LINE_RE.match(paragraph)
    if not match:
        return None
    level = len(match.group(1))
    text = HTML_TAG_RE.sub("", match.group(2)).strip("* _").strip()
    return level, text


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
            # Store the cleaned heading text, not the raw paragraph, so
            # leftover HTML anchors don't leak into the embedded chunk text.
            units.append(Unit("#" * level + " " + heading_text, title, heading, is_reference))
            continue

        pieces = split_long_paragraph(paragraph, chunk_size) if len(paragraph) > chunk_size else [paragraph]
        units.extend(Unit(piece, title, heading, is_reference) for piece in pieces)

    return units


def pack_chunks(units: list[Unit], chunk_size: int, overlap: int) -> list[dict]:
    """Greedily pack units into ~chunk_size-character chunks. Overlap is
    carried forward at unit granularity (whole paragraphs/sentences), never
    a raw character slice, so a chunk boundary never lands mid-link or
    mid-word."""
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


def process_pdf(pdf_path: Path, converter: PdfConverter, chunk_size: int, overlap: int) -> list[dict]:
    markdown = extract_markdown(pdf_path, converter)
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
    parser = argparse.ArgumentParser(description="Extract text from PDFs with marker and chunk it for RAG")
    parser.add_argument("input_dir", nargs="?", default="data", help="Folder of PDFs to process")
    parser.add_argument("--output", default="chunks/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Max characters per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Characters of overlap carried into the next chunk")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {input_dir}")

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

    converter = PdfConverter(artifact_dict=create_model_dict())

    total_chunks = 0
    mode = "a" if done_sources else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for pdf_path in remaining:
            print(f"Processing {pdf_path.name}...")
            records = process_pdf(pdf_path, converter, args.chunk_size, args.overlap)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            total_chunks += len(records)

    print(f"Wrote {total_chunks} chunks from {len(remaining)} PDFs to {output_path}")


if __name__ == "__main__":
    main()
