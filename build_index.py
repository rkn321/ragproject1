import argparse
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag_core import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, DEFAULT_PERSIST_DIR


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunked text and store it in a local Chroma vector index")
    parser.add_argument("--chunks", default="data_ext_vector/chunks.jsonl", help="Path to the chunks JSONL file")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Directory to persist the Chroma index")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/upsert batch size")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}")

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model '{args.model}'...")
    model = SentenceTransformer(args.model)

    client = chromadb.PersistentClient(path=args.persist_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(args.collection)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "title": c.get("title") or "",
            "heading": c.get("heading") or "",
            "is_reference": bool(c.get("is_reference", False)),
        }
        for c in chunks
    ]

    print("Embedding chunks...")
    embeddings = model.encode(
        documents, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    print("Writing to the vector store...")
    for start in range(0, len(ids), args.batch_size):
        end = start + args.batch_size
        # upsert is keyed on id, so re-running after new chunks are appended
        # to the JSONL (e.g. more PDFs processed later) just adds the new
        # ones and leaves existing entries untouched.
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(f"Indexed {len(ids)} chunks into collection '{args.collection}' at {args.persist_dir}")


if __name__ == "__main__":
    main()
