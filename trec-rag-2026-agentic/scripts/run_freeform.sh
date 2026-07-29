#!/usr/bin/env bash
set -euo pipefail

# Route B — free-form tool-calling agent. Same corpus, model and output contract as the
# pipeline runs; the difference is that the agent decides each step for itself.
# Set LIMIT_TOPICS=1 for a smoke test.

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

TOPICS_PATH="${TOPICS_PATH:?Set TOPICS_PATH to rag25-topics-dev.tsv}"
QRELS_DIR="${QRELS_DIR:?Set QRELS_DIR to the rag25-dev-umbrela-qrels directory}"
[[ -f "$TOPICS_PATH" ]] || { echo "Topics file not found: $TOPICS_PATH" >&2; exit 1; }
[[ -d "$QRELS_DIR" ]] || { echo "Qrels dir not found: $QRELS_DIR" >&2; exit 1; }

LLM_MODEL="${LLM_MODEL:-Llama-3.3-70B-Instruct}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SUFFIX="dev22"
[[ -n "${LIMIT_TOPICS:-}" ]] && SUFFIX="smoke"
RUN_ID="freeform-agent-${SUFFIX}-${TS}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/trec-rag-2026/agentic-rag/$RUN_ID}"

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LLM_MODEL=$LLM_MODEL"

npx tsx src/trec-rag-2026/agentic-rag/run_freeform_entry.ts \
  --run-id "$RUN_ID" \
  --team-id "pi-serini" \
  --output-dir "$OUTPUT_DIR" \
  --topics "$TOPICS_PATH" \
  --qrels-dir "$QRELS_DIR" \
  --pyserini-index "$PYSERINI_INDEX" \
  --llm-model "$LLM_MODEL" \
  ${LIMIT_TOPICS:+--limit-topics "$LIMIT_TOPICS"} \
  --force

node -e 'const fs=require("fs"); const v=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); if(!v.ok) process.exit(1); console.log("validation ok", JSON.stringify(v));' "$OUTPUT_DIR/validation.json"

echo "FREEFORM_OUTPUT_DIR=$OUTPUT_DIR"
