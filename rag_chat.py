import argparse

from rag_core import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_PERSIST_DIR,
    DEFAULT_TOP_K,
    NO_MATCH_MESSAGE,
    SYSTEM_PROMPT,
    Retriever,
    build_retrieval_query,
    build_user_turn,
    chat,
    ensure_ollama_ready,
    has_relevant_match,
    source_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a RAG assistant over the indexed Wikipedia articles")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Directory the Chroma index was persisted to")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="sentence-transformers model (must match build_index.py)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama model tag to use for generation")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve per question")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Reject retrieval if the best match is farther than this")
    parser.add_argument(
        "--include-references", action="store_true",
        help="Include reference/citation-section chunks in retrieval (excluded by default as low-value noise)",
    )
    args = parser.parse_args()

    print("Loading embedding model and vector index...")
    retriever = Retriever(args.persist_dir, args.collection, args.model)
    ensure_ollama_ready(args.llm_model)

    print(f"Ready ({retriever.collection.count()} chunks indexed). Ask a question, or type 'exit' to quit.\n")
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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

        retrieval_query = build_retrieval_query(question, history)
        chunks = retriever.retrieve(retrieval_query, args.top_k, args.include_references)

        if not has_relevant_match(chunks, args.max_distance):
            print(f"\nAssistant: {NO_MATCH_MESSAGE}\n")
            continue

        history.append({"role": "user", "content": build_user_turn(question, chunks)})
        try:
            answer = chat(args.llm_model, history)
        except RuntimeError as exc:
            print(f"\nAssistant: [Error] {exc}\n")
            history.pop()  # drop the turn that failed so it doesn't pollute future context
            continue
        print(f"\nAssistant: {answer}")
        print(f"(Retrieved from: {source_summary(chunks)})\n")

        # Keep only the bare question (not the full retrieved context) in
        # long-term history, so follow-up turns don't re-send duplicate
        # excerpts and balloon the request size.
        history[-1] = {"role": "user", "content": question}
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
