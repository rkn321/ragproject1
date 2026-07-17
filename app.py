import argparse

from flask import Flask, jsonify, render_template_string, request

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

PAGE = """
<!doctype html>
<title>Horology RAG Chatbot</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #log { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; height: 60vh; overflow-y: auto; margin-bottom: 1rem; }
  .msg { margin-bottom: 1rem; white-space: pre-wrap; }
  .user { font-weight: 600; }
  .sources { font-size: 0.85em; color: #666; margin-top: 0.25rem; }
  form { display: flex; gap: 0.5rem; }
  input[type=text] { flex: 1; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; }
</style>
<h2>Horology RAG Chatbot</h2>
<p>Ask questions about the indexed Wikipedia articles ({{ chunk_count }} chunks indexed). Answers are grounded in retrieved excerpts and cite sources.</p>
<div id="log"></div>
<form id="form">
  <input type="text" id="input" autocomplete="off" placeholder="Ask a question..." />
  <button type="submit">Send</button>
</form>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  let history = [];

  function addMessage(role, text, sources) {
    const div = document.createElement('div');
    div.className = 'msg';
    const label = role === 'user' ? 'You' : 'Assistant';
    div.innerHTML = '<span class="' + (role === 'user' ? 'user' : '') + '">' + label + ':</span> '
      + text.replace(/</g, '&lt;');
    if (sources) {
      const s = document.createElement('div');
      s.className = 'sources';
      s.textContent = 'Retrieved from: ' + sources;
      div.appendChild(s);
    }
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    addMessage('user', message);

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message, history}),
    });
    const data = await res.json();
    if (data.error) {
      addMessage('assistant', 'Error: ' + data.error);
      return;
    }
    addMessage('assistant', data.answer, data.sources);
    history.push({role: 'user', content: message});
    history.push({role: 'assistant', content: data.answer});
  });
</script>
"""


def create_app(retriever: Retriever, llm_model: str, top_k: int, max_distance: float, include_references: bool) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE, chunk_count=retriever.collection.count())

    @app.post("/api/chat")
    def api_chat():
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not message:
            return jsonify({"error": "message is required"}), 400

        retrieval_query = build_retrieval_query(message, history)
        chunks = retriever.retrieve(retrieval_query, top_k, include_references)

        if not has_relevant_match(chunks, max_distance):
            return jsonify({"answer": NO_MATCH_MESSAGE, "sources": ""})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        messages.append({"role": "user", "content": build_user_turn(message, chunks)})
        try:
            answer = chat(llm_model, messages)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify({"answer": answer, "sources": source_summary(chunks)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat UI for the RAG assistant")
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
    parser.add_argument("--port", type=int, default=5000, help="Local port to serve the web UI on")
    args = parser.parse_args()

    print("Loading embedding model and vector index...")
    retriever = Retriever(args.persist_dir, args.collection, args.model)
    ensure_ollama_ready(args.llm_model)

    app = create_app(retriever, args.llm_model, args.top_k, args.max_distance, args.include_references)
    print(f"Ready ({retriever.collection.count()} chunks indexed). Open http://127.0.0.1:{args.port} in your browser.")
    app.run(port=args.port)


if __name__ == "__main__":
    main()
