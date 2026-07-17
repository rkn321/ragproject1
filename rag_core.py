from dataclasses import dataclass

import chromadb
import ollama
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Centralized so build_index.py, rag_chat.py, and app.py can't silently drift
# out of sync (e.g. indexing with one embedding model but querying with
# another, which would silently degrade retrieval quality).
DEFAULT_PERSIST_DIR = "vector_store"
DEFAULT_COLLECTION = "wikipedia_articles"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_TOP_K = 5

# Chroma's default distance metric is squared L2. Embeddings are normalized
# to unit length, so squared L2 and cosine distance are monotonically
# related; this cutoff is an approximate heuristic (calibrated by comparing
# on-topic vs. clearly-unrelated queries), not an exact similarity bound.
DEFAULT_MAX_DISTANCE = 1.7

SYSTEM_PROMPT = """You are a helpful assistant answering questions using only the provided \
reference excerpts from Wikipedia articles about horology (watches, clocks, and timekeeping).

Rules:
- Answer using only the information in the excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so plainly instead of guessing.
- Cite the source article for each claim, like (Source: Balance_spring.pdf).
"""

NO_MATCH_MESSAGE = (
    "I don't have any indexed content relevant to that question. "
    "Try asking about the watches/clocks/horology articles that were indexed."
)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    heading: str
    distance: float


class Retriever:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(collection)
        if self.collection.count() == 0:
            raise SystemExit(f"Collection '{collection}' is empty. Run build_index.py first.")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, include_references: bool = False) -> list[RetrievedChunk]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        where = None if include_references else {"is_reference": False}
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)
        return [
            RetrievedChunk(text=doc, source=meta["source"], heading=meta.get("heading", ""), distance=dist)
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ]


def build_retrieval_query(question: str, history: list[dict]) -> str:
    """Enrich the retrieval query with the previous user turn so vague
    follow-ups (e.g. "What is it made of?") still retrieve relevant chunks,
    since the embedding model has no other way to resolve the pronoun."""
    previous_questions = [m["content"] for m in history if m["role"] == "user"]
    if not previous_questions:
        return question
    return f"{previous_questions[-1]}\n{question}"


def has_relevant_match(chunks: list[RetrievedChunk], max_distance: float = DEFAULT_MAX_DISTANCE) -> bool:
    return bool(chunks) and chunks[0].distance <= max_distance


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)


def build_user_turn(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Reference excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"


def source_summary(chunks: list[RetrievedChunk]) -> str:
    """De-duplicated, ordered list of source article names actually
    retrieved for an answer, independent of whether the LLM's own citations
    are accurate."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.source not in seen:
            seen.append(chunk.source)
    return ", ".join(seen)


def ensure_ollama_ready(model: str) -> None:
    # ollama's client catches httpx.ConnectError internally and re-raises it
    # as a plain ConnectionError, so that (not httpx.ConnectError) is what
    # actually needs catching here.
    try:
        ollama.show(model)
    except ollama.ResponseError as exc:
        raise SystemExit(
            f"Ollama model '{model}' isn't available ({exc}). Pull it first with: ollama pull {model}"
        ) from exc
    except ConnectionError as exc:
        raise SystemExit(str(exc)) from exc


def chat(llm_model: str, messages: list[dict]) -> str:
    try:
        response = ollama.chat(model=llm_model, messages=messages)
    except ConnectionError as exc:
        raise RuntimeError(str(exc)) from exc
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return response["message"]["content"]
