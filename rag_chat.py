import argparse
import os

import chromadb
from anthropic import Anthropic
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

SYSTEM_PROMPT = """You are a helpful assistant answering questions using only the provided \
reference excerpts from Wikipedia articles about horology (watches, clocks, and timekeeping).

Rules:
- Answer using only the information in the excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so plainly instead of guessing.
- Cite the source article for each claim, like (Source: Balance_spring.pdf).
"""


def build_context(results: dict) -> str:
    blocks = [
        f"[Source: {meta['source']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return "\n\n---\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a RAG assistant over the indexed Wikipedia articles")
    parser.add_argument("--persist-dir", default="vector_store", help="Directory the Chroma index was persisted to")
    parser.add_argument("--collection", default="wikipedia_articles", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model (must match build_index.py)")
    parser.add_argument("--claude-model", default="claude-sonnet-5", help="Anthropic model to use for generation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per question")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set the ANTHROPIC_API_KEY environment variable before running this script.")

    print("Loading embedding model and vector index...")
    embedder = SentenceTransformer(args.model)
    client = chromadb.PersistentClient(path=args.persist_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(args.collection)
    if collection.count() == 0:
        raise SystemExit(f"Collection '{args.collection}' is empty. Run build_index.py first.")

    anthropic_client = Anthropic(api_key=api_key)

    print(f"Ready ({collection.count()} chunks indexed). Ask a question, or type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        query_embedding = embedder.encode([question], normalize_embeddings=True).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=args.top_k)
        context = build_context(results)

        history.append({"role": "user", "content": f"Reference excerpts:\n\n{context}\n\nQuestion: {question}"})

        response = anthropic_client.messages.create(
            model=args.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        answer = "".join(block.text for block in response.content if block.type == "text")
        print(f"\nAssistant: {answer}\n")

        # Keep only the bare question (not the full retrieved context) in
        # long-term history, so follow-up turns don't re-send duplicate
        # excerpts and balloon the request size.
        history[-1] = {"role": "user", "content": question}
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
