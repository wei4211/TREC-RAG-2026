// Sweep candidate confidence thresholds (tau) over already-computed CrossEncoder scores.
// Does NOT call Pyserini or the LLM again -- reads the ce_scores.json artifacts written by
// a prior run (see rerankTopOfRanking in iterative_runner.ts) and re-evaluates instantly.
//
// For each tau, the "confident-only" submission per topic is: docs with ce_calibrated >= tau,
// sorted by ce_calibrated descending. This models the official "choose the submitted depth k
// separately for each narrative / do not pad to a conventional cutoff" rule -- depth varies
// per topic and is never forced to a fixed number.
//
// Usage:
//   npx tsx scripts/sweep_ce_threshold.ts \
//     --run-dir runs/trec-rag-2026/agentic-rag/<run-id> \
//     --topics /path/to/rag25-topics-dev.tsv \
//     --qrels-dir /path/to/rag25-dev-umbrela-qrels \
//     --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7

import { readFileSync, existsSync } from "node:fs";
import { join, basename, resolve } from "node:path";
import { parseTrecRag2026TopicsTsv } from "../src/trec-rag-2026/retrieval/topics";
import { readQrels, evaluateRankings, type Rankings } from "../src/evaluation/retrieval_metrics";

function arg(name: string, def?: string): string {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || !process.argv[i + 1]) {
    if (def !== undefined) return def;
    throw new Error(`Missing --${name}`);
  }
  return process.argv[i + 1];
}

const runDir = resolve(arg("run-dir"));
const topicsPath = resolve(arg("topics"));
const qrelsDir = resolve(arg("qrels-dir"));
const thresholds = arg("thresholds", "0.1,0.2,0.3,0.4,0.5,0.6,0.7")
  .split(",")
  .map((s) => Number.parseFloat(s.trim()));

const QRELS_FILES = [
  "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
  "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
  "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
];

const topics = parseTrecRag2026TopicsTsv(readFileSync(topicsPath, "utf8"));
const qids = topics.map((t) => t.topicId);

type CeScoreRow = { docid: string; ce_logit: number; ce_calibrated: number };
const ceScoresByQid = new Map<string, CeScoreRow[]>();
for (const qid of qids) {
  const p = join(runDir, "topics", `${qid}.ce_scores.json`);
  ceScoresByQid.set(qid, existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : []);
}

function buildRankingsForThreshold(tau: number): { rankings: Rankings; avgDepth: number } {
  const rankings: Rankings = new Map();
  let totalDepth = 0;
  for (const qid of qids) {
    const rows = (ceScoresByQid.get(qid) ?? [])
      .filter((r) => r.ce_calibrated >= tau)
      .sort((a, b) => b.ce_calibrated - a.ce_calibrated);
    rankings.set(
      qid,
      rows.map((r, i) => ({ docid: r.docid, rank: i + 1, score: r.ce_calibrated })),
    );
    totalDepth += rows.length;
  }
  return { rankings, avgDepth: totalDepth / qids.length };
}

const qrelsPaths = QRELS_FILES.map((n) => join(qrelsDir, n));
const cutoffs = { recallCutoffs: [100, 1000], ndcgCutoffs: [10, 20], mrrCutoffs: [1000] };

console.log(
  ["tau", "avg_depth", "ndcg_10", "ndcg_20", "recall_100", "recall_1000", "map"].join("\t"),
);
for (const tau of thresholds) {
  const { rankings, avgDepth } = buildRankingsForThreshold(tau);
  const perQrels = qrelsPaths.map((p) => {
    const qrels = readQrels(p);
    return evaluateRankings(qrels, rankings, qids, cutoffs, {
      recallRelevantThreshold: 2,
      ndcgGainMode: "linear",
    });
  });
  const mean = (get: (r: (typeof perQrels)[number]) => number) =>
    perQrels.reduce((s, r) => s + get(r), 0) / perQrels.length;
  const row = [
    tau.toFixed(2),
    avgDepth.toFixed(1),
    mean((r) => r.ndcgByCutoff.get(10) ?? 0).toFixed(4),
    mean((r) => r.ndcgByCutoff.get(20) ?? 0).toFixed(4),
    mean((r) => r.macroRecallByCutoff.get(100) ?? 0).toFixed(4),
    mean((r) => r.macroRecallByCutoff.get(1000) ?? 0).toFixed(4),
    mean((r) => r.map).toFixed(4),
  ];
  console.log(row.join("\t"));
}
