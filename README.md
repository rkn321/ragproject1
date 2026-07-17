# Horology RAG Chatbot

A local, end-to-end RAG (retrieval-augmented generation) pipeline over Wikipedia articles about
watches, clocks, and timekeeping. Everything runs locally — the embedding model and the LLM both
run on your machine, so no API keys are needed.

The pipeline has four stages:

1. **Download** — grab Wikipedia articles as PDFs (`download_wikipedia_pdf.py`)
2. **Extract + chunk** — PDF → markdown → overlapping text chunks (`extract_and_chunk.py`)
3. **Index** — embed the chunks into a local vector store (`build_index.py`)
4. **Chat** — ask questions, grounded in retrieved excerpts (`rag_chat.py` or `app.py`)

## Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

You also need [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull llama3.2:3b
```

## Usage

### 1. Download articles (optional — `data/` already has some)

```bash
python download_wikipedia_pdf.py "https://en.wikipedia.org/wiki/Balance_spring"
```

Saves to `data/<Article_Title>.pdf` by default, or pass `--output <path>`.

### 2. Extract and chunk

```bash
python extract_and_chunk.py
```

Converts every PDF in `data/` to markdown via [marker](https://github.com/datalab-to/marker), then
splits it into ~1500-character chunks and writes `data_ext_vector/chunks.jsonl`.

This is the slow step — marker runs layout detection and OCR models, so expect several minutes per
PDF on CPU (the first run also downloads a few GB of models). It's **resumable**: if it's
interrupted, just run it again and it skips PDFs already present in the output file.

### 3. Build the vector index

```bash
python build_index.py
```

Embeds each chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) and stores it in a persistent
Chroma index at `vector_store/`. Re-running after new chunks are added only indexes the new ones.

### 4. Chat

Command line:

```bash
python rag_chat.py
```

Or the web UI:

```bash
python app.py
```

Then open http://127.0.0.1:5000.

## How retrieval works

Each question is embedded and matched against the indexed chunks. The top matches are passed to the
LLM as reference excerpts, with instructions to answer only from those excerpts and cite the source
article. A few details worth knowing:

- **Off-topic questions are rejected.** If the best match is farther than `--max-distance`, the bot
  says it has no relevant content instead of letting the LLM improvise an answer.
- **Follow-up questions work.** A vague follow-up ("what is it made of?") is combined with the
  previous turn before embedding, so the retrieval query still has enough context to match.
- **Reference sections are excluded** from retrieval by default, since citation lists are mostly
  URLs and add noise. Pass `--include-references` to index them anyway.

## Common options

All chat/index scripts accept these (see `--help` for the full list):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--llm-model` | `llama3.2:3b` | Ollama model tag used for generation |
| `--model` | `all-MiniLM-L6-v2` | Embedding model (must match between indexing and chat) |
| `--top-k` | `5` | Number of chunks retrieved per question |
| `--max-distance` | `1.7` | Reject retrieval if the best match is farther than this |
| `--persist-dir` | `vector_store` | Where the Chroma index lives |
