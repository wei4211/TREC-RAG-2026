// Recompute R-task metrics from a run's own retrieval.internal.trec-run.tsv, using the exact same
// qrels files, cutoffs, and relevance thresholds as iterative_runner.ts's evalAll. Needed when a run's
// own metrics.json was written before all topics were present (e.g. Route B's dev22: metrics.json was
// written at 21/22, before topic 161 was manually appended after a retry) -- run this against the
// corrected file instead of trusting the stale metrics.json.
//
// Usage: npx tsx scripts/recompute_pool_metrics.ts <run-dir> <qrels-dir>

import { readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { evaluateRankings, type Qrels, type Rankings } from "../src/evaluation/retrieval_metrics";

const CUTS = [10, 20, 50, 100, 500, 1000];
const NDCG = [10, 20, 100, 1000];
const QRELS_FILES = [
  "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
  "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
  "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
];

function parseQrels(path: string, qids: Set<string>): Qrels {
  const q: Qrels = new Map();
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [qid, , docid, rel] = line.split(/\s+/);
    if (!qids.has(qid)) continue;
    const m = q.get(qid) ?? new Map<string, number>();
    m.set(docid, Number(rel) || 0);
    q.set(qid, m);
  }
  return q;
}

function readRankings(path: string): Rankings {
  const r: Rankings = new Map();
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [qid, , docid, rank, score] = line.split(/\s+/);
    const list = r.get(qid) ?? [];
    list.push({ docid, rank: Number(rank), score: Number(score) });
    r.set(qid, list);
  }
  return r;
}

function main() {
  const [runDirArg, qrelsDirArg] = process.argv.slice(2);
  if (!runDirArg || !qrelsDirArg) {
    console.error("usage: npx tsx scripts/recompute_pool_metrics.ts <run-dir> <qrels-dir>");
    process.exit(1);
  }
  const runDir = resolve(runDirArg);
  const rankings = readRankings(join(runDir, "retrieval.internal.trec-run.tsv"));
  const qids = new Set(rankings.keys());
  console.log(`recomputing over ${qids.size} topics`);

  const rows = QRELS_FILES.map((n) => join(resolve(qrelsDirArg), n)).map((p) => {
    const qrels = parseQrels(p, qids);
    const res = evaluateRankings(qrels, rankings, [...qids], { recallCutoffs: CUTS, ndcgCutoffs: NDCG, mrrCutoffs: [1000] }, { recallRelevantThreshold: 2, binaryRelevantThreshold: 2, ndcgGainMode: "linear" });
    return {
      qrels_filename: basename(p),
      ndcg_10: res.ndcgByCutoff.get(10) ?? 0,
      ndcg_20: res.ndcgByCutoff.get(20) ?? 0,
      ndcg_100: res.ndcgByCutoff.get(100) ?? 0,
      ndcg_1000: res.ndcgByCutoff.get(1000) ?? 0,
      recall_100: res.macroRecallByCutoff.get(100) ?? 0,
      recall_1000: res.macroRecallByCutoff.get(1000) ?? 0,
      map: res.map,
      mrr: res.mrrByCutoff.get(1000) ?? 0,
    };
  });

  const keys = Object.keys(rows[0]).filter((k) => k !== "qrels_filename") as (keyof (typeof rows)[0])[];
  const mean = Object.fromEntries(keys.map((k) => [k, rows.reduce((s, r) => s + (r[k] as number), 0) / rows.length]));
  console.log("per-qrels:");
  console.log(JSON.stringify(rows, null, 2));
  console.log("arithmetic_mean_across_qrels:");
  console.log(JSON.stringify(mean, null, 2));
}

main();
