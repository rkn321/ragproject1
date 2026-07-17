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
ollama pull qwen2.5:3b
```

`qwen2.5:3b` (~2 GB) is the default: it fits on a CPU-only machine with ~16 GB
RAM and follows the "answer only from the excerpts, cite the source" system
prompt well. If you have headroom to spare, `qwen2.5:7b` (~4.7 GB) reasons
noticeably better at the cost of a few seconds per answer.

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

Converts every PDF in `data/` to markdown, splits it into ~1000-character chunks, and writes
`data_ext_vector/chunks.jsonl`. The whole corpus takes about 20 seconds.

These are born-digital PDFs, so their text and font metadata are already in the file. The extractor
reads that text layer directly with [pdftext](https://github.com/datalab-to/pdftext) and infers
heading levels from relative font size, rather than running OCR and layout models to rediscover
structure the file already carries.

It's **resumable**: if interrupted, run it again and it skips PDFs already in the output file. You
can also pass a single PDF instead of a folder:

```bash
python extract_and_chunk.py data/Tourbillon.pdf --output data_ext_vector/tourbillon.jsonl
```

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
  says it has no relevant content instead of letting the LLM improvise an answer. On this corpus
  on-topic questions score 0.46-0.94 and unrelated ones 1.34-1.69, so the 1.2 default sits in the
  gap. It isn't a perfect split: a question using a synonym the article doesn't ("hairspring" where
  the text says "balance spring") can be refused even though the answer is there.
- **Follow-up questions work.** A follow-up that leans on a pronoun ("what is it made of?") is
  combined with the previous turn before embedding, so the retrieval query can resolve it. Questions
  that carry their own subject are left alone — enriching those would drag the old topic into every
  later question.
- **Reference sections are excluded** from retrieval by default, since citation lists are mostly
  bibliographic noise. Pass `--include-references` to search them anyway.
- **Link URLs are dropped** from chunk text. The anchor text carries the meaning, while hrefs are
  semantically empty and were eating ~44% of every chunk's tokens — enough to push most chunks past
  the embedding model's 256-token limit, where the overflow is silently truncated.

### Chunk size and the embedding model

`all-MiniLM-L6-v2` truncates at **256 tokens**, silently: a longer chunk is still stored and shown
to the LLM in full, but only its opening is searchable. The `--chunk-size 1000` default keeps the
median body chunk near 196 tokens. What still overruns is mostly reference lists, which tokenize
badly and are excluded from retrieval anyway. If you raise `--chunk-size`, switch to an embedding
model with a longer limit (e.g. `BAAI/bge-small-en-v1.5` at 512) or you'll silently lose the tails.

## Common options

All chat/index scripts accept these (see `--help` for the full list):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--llm-model` | `qwen2.5:3b` | Ollama model tag used for generation |
| `--model` | `all-MiniLM-L6-v2` | Embedding model (must match between indexing and chat) |
| `--top-k` | `5` | Number of chunks retrieved per question |
| `--max-distance` | `1.2` | Reject retrieval if the best match is farther than this |
| `--chunk-size` | `1000` | Max characters per chunk (see note above before raising) |
| `--persist-dir` | `vector_store` | Where the Chroma index lives |
