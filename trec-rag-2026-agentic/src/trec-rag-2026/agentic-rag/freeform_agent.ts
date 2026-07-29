// Route B — free-form tool-calling agent.
//
// Where the A/B pipeline hard-codes the sequence (retrieve, judge, decompose into aspects, write per
// aspect, fill gaps), this version hands the decisions to the model: at every step it sees what it has
// searched and read so far and chooses the next action itself. The point is an architectural comparison,
// not a parameter sweep — the two share a corpus, a model, and an output contract, and differ in who
// decides what happens next.
//
// Tool calls are expressed as JSON rather than the provider's function-calling API, because the NCHC
// client exposes only messages plus json_object. That also keeps the agent portable across models.
//
// Deliberately self-contained: it duplicates small search/read helpers rather than importing them from
// iterative_runner, so that changing one pipeline can never perturb a run of the other.

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseTrecRag2026TopicsTsv } from "../retrieval/topics";
import { createLlmClient } from "../../llm/create";
import { normalizeLlmClientConfig, type RawLlmClientConfig } from "../../llm/config";
import type { LlmClient } from "../../llm/types";
import { AGENTIC_RAG_BASELINE_PROMPT_VERSION, CLIMBMIX_DOCID_RE, type AgenticRagOutputObject } from "../agentic-rag-baseline/contracts";
import { evaluateRankings, type Qrels, type Rankings } from "../../evaluation/retrieval_metrics";

export type FreeformOptions = {
  runId: string; teamId: string; outputDir: string; topicsPath: string; qrelsDir: string;
  pyseriniBaseUrl: string; pyseriniIndex: string; pyseriniTokenEnv: string;
  limitTopics?: number; llm: RawLlmClientConfig; force?: boolean; env?: NodeJS.ProcessEnv;
};

const POLICY = {
  architecture: "freeform-tool-calling-agent",
  max_steps: 24,           // hard ceiling on agent turns per topic
  max_reads: 40,           // hard ceiling on documents read per topic
  search_hits: 20,         // hits returned to the agent per search
  snippet_chars: 300,      // per-hit snippet in a search observation
  read_chars: 2500,        // document text returned by a read
  recent_observations: 6,  // full observations kept in context; older ones collapse to one line
  output_depth: 1000,      // ranked list depth for the R task
  rrf_k: 60,
  max_answer_words: 1000,
  step_max_tokens: 1200,
  answer_max_tokens: 4096,
} as const;

type Hit = { docid: string; score: number; snippet: string };
type Step = { n: number; action: string; detail: string; observation: string; ok: boolean };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ── Corpus access ────────────────────────────────────────────────────────────
async function search(o: FreeformOptions, query: string, hits: number, env: NodeJS.ProcessEnv): Promise<Hit[]> {
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/search?${new URLSearchParams({ query, hits: String(hits) })}`;
  for (let attempt = 1; attempt <= 6; attempt++) {
    let r: Response;
    try { r = await fetch(url, { headers: token ? { authorization: `Bearer ${token}` } : {} }); }
    catch { if (attempt === 6) return []; await sleep(600 * 2 ** attempt); continue; }
    if (r.ok) {
      const v = (await r.json()) as any;
      return (v.candidates ?? [])
        .filter((c: any) => typeof c.docid === "string")
        .slice(0, hits)
        .map((c: any) => ({ docid: c.docid, score: Number(c.score) || 0, snippet: docText(c.doc).slice(0, POLICY.snippet_chars) }));
    }
    if (![429, 500, 502, 503, 504].includes(r.status)) return [];
    await sleep(600 * 2 ** attempt);
  }
  return [];
}

async function readDoc(o: FreeformOptions, docid: string, env: NodeJS.ProcessEnv): Promise<string | null> {
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/doc/${encodeURIComponent(docid)}`;
  for (let attempt = 1; attempt <= 6; attempt++) {
    let r: Response;
    try { r = await fetch(url, { headers: token ? { authorization: `Bearer ${token}` } : {} }); }
    catch { if (attempt === 6) return null; await sleep(600 * 2 ** attempt); continue; }
    if (r.status === 404) return null;
    if (r.ok) return docText(((await r.json()) as any).doc);
    if (![429, 500, 502, 503, 504].includes(r.status)) return null;
    await sleep(r.status === 429 ? Math.min(30000, 2000 * 2 ** attempt) : 600 * 2 ** attempt);
  }
  return null;
}

function docText(doc: any): string {
  if (typeof doc === "string") return doc;
  if (doc && typeof doc.text === "string") return doc.text;
  if (doc && typeof doc.contents === "string") return doc.contents;
  return JSON.stringify(doc ?? "");
}

// ── Agent prompt ─────────────────────────────────────────────────────────────
function actionPrompt(narrative: string, steps: Step[], known: Map<string, Hit>, readIds: Set<string>, budget: { steps: number; reads: number }): string {
  const recent = steps.slice(-POLICY.recent_observations);
  const older = steps.slice(0, Math.max(0, steps.length - POLICY.recent_observations));
  const history = [
    ...older.map((s) => `Step ${s.n}: ${s.action}(${s.detail}) -> ${s.ok ? "ok" : "failed"}`),
    ...recent.map((s) => `Step ${s.n}: ${s.action}(${s.detail})\nResult:\n${s.observation}`),
  ].join("\n\n");
  const unread = [...known.values()].filter((h) => !readIds.has(h.docid)).slice(0, 30);
  return [
    "You are a retrieval agent answering an information need from the ClimbMix corpus. You choose what to do next.",
    "",
    "Available actions, one per turn, as strict JSON and nothing else:",
    '  {"action":"search","query":"<keyword query>","why":"<short reason>"}',
    '  {"action":"read","docid":"shard_00000_00000","why":"<short reason>"}',
    '  {"action":"answer","why":"<short reason>"}   — only when the evidence is sufficient',
    "",
    "Guidance:",
    "- Search with short keyword queries; vary them to cover every distinct thing the narrative asks about.",
    "- Read a document only when its snippet suggests it carries specific facts you still need.",
    "- The final answer is scored on how many distinct required facts it states completely, so seek breadth of specific facts (figures, dates, named entities, mechanisms), not general background.",
    "- Choose \"answer\" once you can state specific facts covering every part of the narrative.",
    "",
    `Budget left: ${budget.steps} steps, ${budget.reads} reads. When steps run out you are forced to answer, so leave time to gather evidence.`,
    "",
    `Information need:\n${narrative}`,
    "",
    history ? `What you have done so far:\n${history}` : "You have not acted yet.",
    "",
    unread.length > 0 ? `Retrieved but unread documents:\n${unread.map((h) => `  ${h.docid}: ${h.snippet.slice(0, 160)}`).join("\n")}` : "No unread documents yet.",
    `Documents read so far: ${readIds.size}`,
    "",
    "Return only the JSON object for your next action.",
  ].join("\n");
}

function answerPrompt(narrative: string, docs: { docid: string; text: string }[]): string {
  return [
    "Write the final evidence-grounded answer using only the documents below.",
    "Break it into individual sentences. State exactly ONE fact per sentence, stated COMPLETELY, typically 15-25 words.",
    "Completeness is the priority: a fact stated only partially scores nothing. If it is a list, name every item; if it involves a quantity, give the number with its unit and period; name entities in full.",
    "Cover every distinct thing the narrative asks about. Do not write generic background sentences.",
    `Keep the whole answer at or below ${POLICY.max_answer_words} words.`,
    "Every sentence must cite at least one and at most three documents, by docid, ordered strongest support first.",
    "Return only strict JSON:",
    '{"answer":[{"text":"A complete factual sentence.","citations":["shard_00000_00000"]}]}',
    "",
    `Information need:\n${narrative}`,
    "",
    "Documents:",
    ...docs.map((d) => `[${d.docid}]\n${d.text}`),
  ].join("\n");
}

// ── JSON helpers ─────────────────────────────────────────────────────────────
function extractObject(text: string): any | null {
  const cleaned = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = cleaned.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < cleaned.length; i++) {
    if (cleaned[i] === "{") depth++;
    else if (cleaned[i] === "}") { depth--; if (depth === 0) { try { return JSON.parse(cleaned.slice(start, i + 1)); } catch { return null; } } }
  }
  return null;
}

// ── One topic ────────────────────────────────────────────────────────────────
async function runTopic(a: { topic: { qid: string; narrative: string }; o: FreeformOptions; llm: LlmClient; env: NodeJS.ProcessEnv }) {
  const known = new Map<string, Hit>();          // every hit the agent has seen
  const searchRuns: string[][] = [];             // docid order per search, for RRF
  const readDocs = new Map<string, string>();    // docid -> text
  const steps: Step[] = [];
  let queries: string[] = [];
  let decidedToAnswer = false;

  for (let n = 1; n <= POLICY.max_steps; n++) {
    const budget = { steps: POLICY.max_steps - n + 1, reads: POLICY.max_reads - readDocs.size };
    let action: any = null;
    try {
      const r = await a.llm.generate({
        messages: [{ role: "user", content: actionPrompt(a.topic.narrative, steps, known, new Set(readDocs.keys()), budget) }],
        temperature: 0, maxTokens: POLICY.step_max_tokens, responseFormat: "json_object",
      });
      action = extractObject(r.text);
    } catch { /* fall through to the malformed-action branch */ }

    if (!action || typeof action.action !== "string") {
      steps.push({ n, action: "invalid", detail: "", observation: "Could not parse an action. Reply with exactly one JSON action object.", ok: false });
      continue;
    }

    if (action.action === "answer") { decidedToAnswer = true; steps.push({ n, action: "answer", detail: String(action.why ?? ""), observation: "Proceeding to write the answer.", ok: true }); break; }

    if (action.action === "search") {
      const query = String(action.query ?? "").trim();
      if (!query) { steps.push({ n, action: "search", detail: "", observation: "Empty query.", ok: false }); continue; }
      const hits = await search(a.o, query, POLICY.search_hits, a.env);
      for (const h of hits) if (!known.has(h.docid)) known.set(h.docid, h);
      searchRuns.push(hits.map((h) => h.docid));
      queries.push(query);
      steps.push({ n, action: "search", detail: query, observation: hits.length === 0 ? "No results." : hits.slice(0, 10).map((h) => `  ${h.docid}: ${h.snippet.slice(0, 200)}`).join("\n"), ok: hits.length > 0 });
      await sleep(200);
      continue;
    }

    if (action.action === "read") {
      const docid = String(action.docid ?? "").trim();
      if (!CLIMBMIX_DOCID_RE.test(docid)) { steps.push({ n, action: "read", detail: docid, observation: "Not a valid ClimbMix docid.", ok: false }); continue; }
      if (readDocs.has(docid)) { steps.push({ n, action: "read", detail: docid, observation: "Already read; choose a different document or answer.", ok: false }); continue; }
      if (readDocs.size >= POLICY.max_reads) { steps.push({ n, action: "read", detail: docid, observation: "Read budget exhausted; you must answer now.", ok: false }); continue; }
      const text = await readDoc(a.o, docid, a.env);
      if (text === null) { steps.push({ n, action: "read", detail: docid, observation: "Document not found.", ok: false }); continue; }
      readDocs.set(docid, text);
      steps.push({ n, action: "read", detail: docid, observation: text.slice(0, POLICY.read_chars), ok: true });
      await sleep(200);
      continue;
    }

    steps.push({ n, action: String(action.action), detail: "", observation: 'Unknown action. Use "search", "read", or "answer".', ok: false });
  }

  // Final answer over everything the agent chose to read.
  const docs = [...readDocs.entries()].map(([docid, text]) => ({ docid, text: text.slice(0, 2000) }));
  let sentences: { text: string; docids: string[] }[] = [];
  if (docs.length > 0) {
    try {
      const r = await a.llm.generate({ messages: [{ role: "user", content: answerPrompt(a.topic.narrative, docs) }], temperature: 0, maxTokens: POLICY.answer_max_tokens, responseFormat: "json_object" });
      const parsed = extractObject(r.text);
      const arr: any[] = Array.isArray(parsed?.answer) ? parsed.answer : [];
      let words = 0;
      for (const s of arr) {
        if (typeof s?.text !== "string" || !s.text.trim()) continue;
        const docids = [...new Set((Array.isArray(s.citations) ? s.citations : []).map((c: any) => String(c)).filter((d: string) => readDocs.has(d)))].slice(0, 3) as string[];
        if (docids.length === 0) continue;
        const w = s.text.split(/\s+/).filter(Boolean).length;
        if (words + w > POLICY.max_answer_words && sentences.length > 0) break;
        sentences.push({ text: s.text.trim(), docids });
        words += w;
      }
    } catch { /* leave sentences empty; the fallback below handles it */ }
  }
  // A topic must still produce a valid object, so fall back to the first read documents.
  if (sentences.length === 0 && docs.length > 0) {
    sentences = docs.slice(0, 5).map((d) => ({ text: d.text.split(/(?<=[.!?])\s+/).find((s) => s.split(/\s+/).length >= 8)?.slice(0, 400) ?? d.text.slice(0, 300), docids: [d.docid] }));
  }

  const references: string[] = [];
  const idx = new Map<string, number>();
  for (const s of sentences) for (const d of s.docids) if (!idx.has(d)) { idx.set(d, references.length); references.push(d); }

  const ragObject: AgenticRagOutputObject = {
    metadata: {
      team_id: a.o.teamId, run_id: a.o.runId, type: "automatic", narrative_id: a.topic.qid,
      title: "", narrative: a.topic.narrative, prompt: AGENTIC_RAG_BASELINE_PROMPT_VERSION,
      run_desc: POLICY.architecture, generator: a.llm.model, retrieval_depth: POLICY.output_depth,
    },
    references,
    answer: sentences.map((s) => ({ text: s.text, citations: s.docids.map((d) => idx.get(d)!) })),
  };

  // R-task ranking: RRF over every search the agent chose to run, so its query strategy is what is scored.
  const fused = new Map<string, number>();
  for (const run of searchRuns) run.forEach((docid, rank) => fused.set(docid, (fused.get(docid) ?? 0) + 1 / (POLICY.rrf_k + rank + 1)));
  const ranking = [...fused.entries()].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0])).slice(0, POLICY.output_depth)
    .map(([docid, score], i) => ({ docid, rank: i + 1, score }));

  return { ragObject, ranking, steps, queries, decidedToAnswer, readCount: readDocs.size };
}

// ── Driver ───────────────────────────────────────────────────────────────────
export async function runFreeformAgenticRag(o: FreeformOptions) {
  const env = o.env ?? process.env;
  const out = resolve(o.outputDir);
  if (o.force) rmSync(out, { recursive: true, force: true });
  mkdirSync(join(out, "topics"), { recursive: true });
  const llm = createLlmClient(normalizeLlmClientConfig(o.llm), env);
  const topics = parseTrecRag2026TopicsTsv(readFileSync(resolve(o.topicsPath), "utf8"))
    .map((t) => ({ qid: t.topicId, narrative: t.narrative }))
    .slice(0, o.limitTopics ?? Infinity);

  const rags: AgenticRagOutputObject[] = [];
  const runLines: string[] = [];
  let failed = 0;

  for (const [i, topic] of topics.entries()) {
    try {
      const r = await runTopic({ topic, o, llm, env });
      writeFileSync(join(out, "topics", `${topic.qid}.agent_trace.json`), JSON.stringify({ topic_id: topic.qid, steps: r.steps, queries: r.queries, decided_to_answer: r.decidedToAnswer, documents_read: r.readCount }, null, 2));
      rags.push(r.ragObject);
      for (const e of r.ranking) runLines.push(`${topic.qid} Q0 ${e.docid} ${e.rank} ${e.score.toFixed(8)} ${o.runId}`);
      console.error(`${i + 1}/${topics.length} ${topic.qid} steps=${r.steps.length} read=${r.readCount} sentences=${r.ragObject.answer.length} ${r.decidedToAnswer ? "self-stopped" : "budget-stopped"}`);
    } catch (e) {
      failed++;
      console.error(`${i + 1}/${topics.length} ${topic.qid} FAILED ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  writeFileSync(join(out, "rag_output_trec_rag_2026.jsonl"), rags.map((x) => JSON.stringify(x)).join("\n") + (rags.length ? "\n" : ""));
  writeFileSync(join(out, "retrieval.internal.trec-run.tsv"), runLines.join("\n") + (runLines.length ? "\n" : ""));
  const validation = { ok: failed === 0 && rags.length === topics.length, output_count: rags.length, expected_count: topics.length };
  writeFileSync(join(out, "validation.json"), JSON.stringify(validation, null, 2));
  writeFileSync(join(out, "policy.json"), JSON.stringify(POLICY, null, 2));

  // Same three dev qrels and the same cutoffs the pipeline runs use, so R is directly comparable.
  const rankings: Rankings = new Map();
  for (const line of runLines) { const [qid, , docid, rank, score] = line.split(/\s+/); const list = rankings.get(qid) ?? []; list.push({ docid, rank: Number(rank), score: Number(score) }); rankings.set(qid, list); }
  const qrelsFiles = [
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
    "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
    "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
  ].map((n) => join(resolve(o.qrelsDir), n));
  const qids = topics.map((t) => t.qid);
  const rows = qrelsFiles.filter((p) => existsSync(p)).map((p) => {
    const q: Qrels = new Map();
    for (const line of readFileSync(p, "utf8").split(/\r?\n/)) {
      if (!line.trim()) continue;
      const [qid, , docid, rel] = line.split(/\s+/);
      if (!qids.includes(qid)) continue;
      const m = q.get(qid) ?? new Map<string, number>();
      m.set(docid, Number(rel) || 0); q.set(qid, m);
    }
    const res = evaluateRankings(q, rankings, qids, { recallCutoffs: [10, 20, 50, 100, 500, 1000], ndcgCutoffs: [10, 20, 100, 1000], mrrCutoffs: [1000] }, { recallRelevantThreshold: 2, binaryRelevantThreshold: 2, ndcgGainMode: "linear" });
    return { ndcg_10: res.ndcgByCutoff.get(10) ?? 0, ndcg_20: res.ndcgByCutoff.get(20) ?? 0, ndcg_100: res.ndcgByCutoff.get(100) ?? 0, recall_100: res.macroRecallByCutoff.get(100) ?? 0, recall_1000: res.macroRecallByCutoff.get(1000) ?? 0, map: res.map, mrr: res.mrrByCutoff.get(1000) ?? 0 };
  });
  if (rows.length > 0) {
    const keys = Object.keys(rows[0]) as (keyof (typeof rows)[0])[];
    const mean = Object.fromEntries(keys.map((k) => [k, rows.reduce((s, r) => s + r[k], 0) / rows.length]));
    writeFileSync(join(out, "metrics.json"), JSON.stringify({ qrels: rows, arithmetic_mean_across_qrels: mean }, null, 2));
  }
  return { outputDir: out, validation };
}
