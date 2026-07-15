import argparse
import json
import re
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def extract_markdown(pdf_path: Path, converter: PdfConverter) -> str:
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


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


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Pack paragraphs into ~chunk_size-character chunks, carrying a bit of
    trailing context from one chunk into the next so retrieval doesn't lose
    context at a chunk boundary."""
    units: list[str] = []
    for paragraph in split_into_paragraphs(text):
        if len(paragraph) > chunk_size:
            units.extend(split_long_paragraph(paragraph, chunk_size))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:]
            current = f"{tail}\n\n{unit}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def process_pdf(pdf_path: Path, converter: PdfConverter, chunk_size: int, overlap: int) -> list[dict]:
    markdown = extract_markdown(pdf_path, converter)
    chunks = chunk_text(markdown, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "id": f"{pdf_path.stem}::{i}",
            "source": pdf_path.name,
            "chunk_index": i,
            "text": chunk,
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


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

    converter = PdfConverter(artifact_dict=create_model_dict())

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as f:
        for pdf_path in pdf_paths:
            print(f"Processing {pdf_path.name}...")
            records = process_pdf(pdf_path, converter, args.chunk_size, args.overlap)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chunks += len(records)

    print(f"Wrote {total_chunks} chunks from {len(pdf_paths)} PDFs to {output_path}")


if __name__ == "__main__":
    main()
