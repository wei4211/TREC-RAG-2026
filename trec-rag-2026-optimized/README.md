# TREC RAG 2026 standalone iterative agentic RAG runner

This repository contains a **standalone iterative agentic RAG runner** / CLI for the TREC RAG 2026 baseline. The runtime is the TypeScript CLI and does not require a chat-tool runtime.


## Pipeline

```text
topic narrative
-> raw query BM25 search top1000
-> read first 5 docs
-> NCHC LLM judge: enough? + follow-up queries
-> if enough: generate cited sentence-level answer
-> if not enough:
   -> check generated follow-up queries
   -> BM25 each follow-up query top1000
   -> weighted RRF fusion
      raw query weight = 1.0
      follow-up query weight = 0.25
   -> read 4 new unseen docs
   -> repeat judge loop
-> stop on enough / budget / no valid follow-up / judge fallback
-> generate TREC RAG 2026 JSONL output
```

## Runtime capabilities

- Pyserini REST `search(query, top1000)` against `climbmix-400b`
- Pyserini REST `doc(docid)` document reads
- NCHC LLM judge JSON request
- query checker for generated follow-up queries
- weighted RRF fusion
- NCHC LLM answer JSON request
- TREC JSONL validation, traces, and retrieval metrics

## Defaults

- `initial_docs = 5`
- `docs_per_iteration = 4`
- `max_documents_read = 12`
- `max_iterations = 3`
- `document_read_limit = 200`
- `pyserini_index = climbmix-400b`
- `llm_model = gpt-oss-120b`
- raw query weight `1.0`
- follow-up query weight `0.25`

## Environment

Copy the example file and fill in local credentials:

```bash
cp .env.example .env.local
```

Variables:

- `NCHC_API_KEY` — required for NCHC LLM calls
- `NCHC_BASE_URL` — default `https://portal.genai.nchc.org.tw/api/v1`
- `PYSERINI_API_TOKEN` — set if the Pyserini REST API requires a token
- `PYSERINI_INDEX` — default `climbmix-400b`
- optional `PYSERINI_BASE_URL` — default `http://api.castorini.uwaterloo.ca`
- optional `TOPICS_PATH` and `QRELS_DIR` for scripts

Default development-data paths on the shared server:

```text
TOPICS_PATH=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv
QRELS_DIR=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels
```

If your checkout is not on that server, override these paths with your local copies of the development topics and qrels.

Never commit or share `.env.local`.

## Quick start for teammates

1. Extract the tarball:

   ```bash
   tar -xzf trec-rag-2026-agentic-rag-baseline-<timestamp>.tar.gz
   cd trec-rag-2026-agentic-rag-baseline-<timestamp>
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create local environment file:

   ```bash
   cp .env.example .env.local
   ```

4. Edit `.env.local` and fill in `NCHC_API_KEY`; set `PYSERINI_API_TOKEN` only if required.

5. Typecheck:

   ```bash
   npm run typecheck -- --pretty false
   ```

6. Run a one-topic smoke test with local data paths:

   ```bash
   TOPICS_PATH=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
   QRELS_DIR=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels \
   scripts/run_dev_smoke.sh
   ```

7. Run full dev22:

   ```bash
   TOPICS_PATH=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
   QRELS_DIR=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels \
   scripts/run_dev22.sh
   ```

## Install

```bash
npm install
npm run typecheck -- --pretty false
```

Node.js 20+ is recommended.

## Run smoke test

The smoke test runs one topic only. It checks the environment, API keys, Pyserini, NCHC LLM, and JSONL output path. It is **not** an official experiment result.

```bash
scripts/run_dev_smoke.sh
```

The script prints `OUTPUT_DIR=...` and writes under:

```text
runs/trec-rag-2026/agentic-rag/<timestamped-smoke-run-id>/
```

## Run full dev22

```bash
scripts/run_dev22.sh
```

You may override data paths:

```bash
TOPICS_PATH=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
QRELS_DIR=/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels \
scripts/run_dev22.sh
```

## Expected outputs

Each run directory contains, among others:

- `rag_output_trec_rag_2026.jsonl` — final RAG JSONL
- `candidate_pool_top1000.trec` and `retrieval.internal.trec-run.tsv`
- `validation.json`
- `metrics.json` and `per_topic_metrics.json`
- `iteration_trace.jsonl`, `retrieval_trace.jsonl`, `llm_trace.jsonl`
- per-topic traces under `topics/`

Check for secret leaks before sharing an output directory:

```bash
scripts/check_no_secret_leak.sh <output_dir>
```


## Known limitations

- Depends on availability/rate limits of Pyserini REST and NCHC LLM.
- Follow-up query quality is constrained by a strict checker.
- Only up to 12 documents are read per topic by default.
- Retrieval is BM25 + weighted RRF only; no reranker is used.
- Dev metrics require local qrels files.
