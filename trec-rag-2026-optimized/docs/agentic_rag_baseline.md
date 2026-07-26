# Agentic RAG baseline details

This is a **standalone iterative agentic RAG runner**: a code-controlled loop that calls retrieval APIs and LLM JSON endpoints. It is not a Pi/Codex free-form tool-calling agent.

## Raw query

The raw query is the original topic narrative string from the TREC RAG topic TSV. The runner uses the narrative directly, without LLM rewriting, as the raw query for the first retrieval call.

## Raw query BM25 top1000

The runner sends the raw query to the Pyserini REST search API against `climbmix-400b` and requests the top 1000 BM25 hits. This top1000 pool is used as the raw-query retrieval run in later fusion.

## Why only read the first 5 docs first?

The LLM does not inspect all top1000 retrieved documents. Reading all top1000 would be expensive, slow, and far beyond the answer prompt budget. Instead, the runner reads the first 5 documents from the initial BM25 ranking and asks the LLM to judge whether this small evidence set is sufficient. If not, the LLM proposes focused follow-up queries and the runner expands the evidence set iteratively.

## Follow-up query generation

At each iteration, the NCHC LLM receives:

- the topic narrative,
- previous queries,
- the currently read evidence snippets.

It must return JSON indicating whether evidence is sufficient. If insufficient, it returns 1 to 3 follow-up BM25 keyword queries. The runner, not the LLM, decides whether to execute them.

## Check generated queries

Generated follow-up queries are accepted only if they pass all rules:

- value is a string,
- trim result is non-empty,
- 4 to 12 English/digit tokens,
- not a duplicate of previous queries,
- no malformed/code-like tokens, including `{`, `}`, `?`, `;`, `:`, `` ``` ``, `def`, `function`,
- no broad/instruction-like words, including `obtain`, `find`, `source`, `sources`, `detail`, `detailed`, `comprehensive`, `concrete`, `examples`, `evidence`, `information`, `overview`, `history`, `impact`, `provide`, `explain`.

The implementation also removes duplicates within the same returned list and limits accepted follow-ups to 3.

## Weighted RRF

After valid follow-up queries are accepted:

- raw query BM25 top1000 is kept as the raw-query run with weight `1.0`,
- each follow-up query BM25 top1000 is added with weight `0.25`,
- weighted reciprocal rank fusion uses `rrf_k = 60`,
- fused output depth is 1000.

This keeps the original narrative query dominant while allowing focused follow-up queries to surface additional documents.

## Read budget

- Initial read: first 5 documents from raw-query BM25.
- Each loop: read 4 new unseen documents from the fused ranking.
- Maximum documents read per topic: 12.
- Default document read limit: 200 lines per document.

## Stop reasons

A topic stops when one of these occurs:

- `enough` — judge says evidence is sufficient,
- `max_iterations` — reached 3 judge/retrieval iterations,
- `max_documents_read` — reached 12 successfully read documents,
- `no_valid_followup_query` — judge returned no legal follow-up query,
- judge fallback condition, such as `judge_empty_assistant_message`, `judge_rate_limit`, `judge_server_error`, `judge_transient_request_failed`, or `judge_json_parse_failed`.

After stopping, the runner still generates the final cited sentence-level answer from the read evidence.

## Output artifacts

Run-level artifacts include:

- `config.json`,
- `rag_output_trec_rag_2026.jsonl`,
- `candidate_pool_top1000.trec`,
- `retrieval.internal.trec-run.tsv`,
- `validation.json`,
- `metrics.json`, `per_topic_metrics.json`,
- `iteration_trace.jsonl`,
- `retrieval_trace.jsonl`,
- `final_read_docs_trace.jsonl`,
- `llm_trace.jsonl`,
- `run-summary.internal.json`,
- per-topic trace and validation files under `topics/`.

## Standalone runner vs. free-form tool-calling agent

This baseline is a standalone code-controlled agentic RAG pipeline. The TypeScript runner owns the loop, budgets, query validation, retrieval calls, fusion, stopping logic, and artifact writing.

A free-form ReAct-style tool-calling agent would let the model decide when and how to call tools during runtime. This baseline does not do that. The LLM only answers constrained JSON requests for evidence sufficiency/follow-up queries and final answer drafting.

Therefore, the current system is **not** a free-form tool-calling agent and should not be described as one.

## Sharing exclusions

Do not share:

- `.env.local`,
- real API keys / tokens,
- `node_modules/`,
- `tmp/`,
- large failed experiment `runs/`,
- private server absolute paths only artifacts,
- the full ClimbMix corpus.
