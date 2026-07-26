#!/usr/bin/env bash
set -euo pipefail

# Full dev22 run using the current final defaults.
# Uses .env.local and writes to runs/trec-rag-2026/agentic-rag/<timestamped-run-id>.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
else
  echo "Missing .env.local. Copy .env.example to .env.local and fill credentials." >&2
  exit 1
fi

: "${NCHC_API_KEY:?NCHC_API_KEY is required in .env.local}"
export NCHC_BASE_URL="${NCHC_BASE_URL:-https://portal.genai.nchc.org.tw/api/v1}"
export PYSERINI_INDEX="${PYSERINI_INDEX:-climbmix-400b}"

TOPICS_PATH="${TOPICS_PATH:-/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv}"
QRELS_DIR="${QRELS_DIR:-/tmp2/trec-rag26/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels}"

if [[ ! -f "$TOPICS_PATH" ]]; then
  echo "Topics file not found: $TOPICS_PATH" >&2
  echo "Override by setting TOPICS_PATH to your local rag25-topics-dev.tsv file." >&2
  exit 1
fi
if [[ ! -d "$QRELS_DIR" ]]; then
  echo "Qrels dir not found: $QRELS_DIR" >&2
  echo "Override by setting QRELS_DIR to your local rag25-dev-umbrela-qrels directory." >&2
  exit 1
fi

LLM_MODEL="${LLM_MODEL:-gpt-oss-120b}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="iterative-agentic-w025-v4port-dev22-${TS}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/trec-rag-2026/agentic-rag/$RUN_ID}"

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LLM_MODEL=$LLM_MODEL"

npx tsx src/trec-rag-2026/agentic-rag/run_iterative_entry.ts \
  --run-id "$RUN_ID" \
  --team-id "pi-serini" \
  --output-dir "$OUTPUT_DIR" \
  --topics "$TOPICS_PATH" \
  --qrels-dir "$QRELS_DIR" \
  --initial-docs "${INITIAL_DOCS:-5}" \
  --docs-per-iteration "${DOCS_PER_ITER:-4}" \
  --max-documents-read "${MAX_DOCS:-12}" \
  --max-iterations "${MAX_ITERS:-3}" \
  --document-read-limit 200 \
  --pyserini-index "$PYSERINI_INDEX" \
  --llm-model "$LLM_MODEL" \
  --force

node -e 'const fs=require("fs"); const p=process.argv[1]; const v=JSON.parse(fs.readFileSync(p,"utf8")); if(!v.ok) process.exit(1); console.log("validation ok", JSON.stringify(v));' "$OUTPUT_DIR/validation.json"

echo "DEV22_OUTPUT_DIR=$OUTPUT_DIR"
