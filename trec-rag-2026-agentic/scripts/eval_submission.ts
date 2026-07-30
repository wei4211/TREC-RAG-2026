// Evaluate the variable-depth Retrieval submission, not the internal candidate pool.
//
// metrics.json scores `<qid>.runfile.trec`, which is the full reranked pool. What would actually be
// submitted is `submission_variable_k.trec`: per the 2026 variable-depth rule, each narrative carries its
// own k and must not be padded to a fixed cutoff. Those are different files, so nDCG@10 over the pool
// says nothing about how good the submitted list is -- for a topic where k < 10 the submitted list does
// not even have ten rows.
//
// This scores the submission on its own terms (precision/recall/nDCG at each topic's own k) and, for
// comparison, what fixed depths would have produced from the same ranking. That is what tells us whether
// the confidence threshold and the [min,max] clamp are set sensibly.
//
// Usage: npx tsx scripts/eval_submission.ts <run-dir> <qrels-dir>

import { existsSync, readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const QRELS_FILES = [
  "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
  "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
  "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
];
const REL_THRESHOLD = 2;            // matches the recall/binary threshold used for metrics.json
const FIXED_DEPTHS = [5, 10, 15, 20, 50];

type Ranked = Map<string, string[]>;                 // qid -> docids in rank order
type Qrels = Map<string, Map<string, number>>;       // qid -> docid -> relevance

function readRanked(path: string): Ranked {
  const out: Ranked = new Map();
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [qid, , docid] = line.split(/\s+/);
    const list = out.get(qid) ?? [];
    list.push(docid);
    out.set(qid, list);
  }
  return out;
}

function readQrels(path: string): Qrels {
  const out: Qrels = new Map();
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [qid, , docid, rel] = line.split(/\s+/);
    const m = out.get(qid) ?? new Map<string, number>();
    m.set(docid, Number(rel) || 0);
    out.set(qid, m);
  }
  return out;
}

/** Linear-gain nDCG over the first `k` entries, against the ideal ranking of that topic's judged docs. */
function ndcgAt(docids: string[], judged: Map<string, number>, k: number): number {
  const dcg = docids.slice(0, k).reduce((sum, docid, i) => sum + (judged.get(docid) ?? 0) / Math.log2(i + 2), 0);
  const ideal = [...judged.values()].filter((r) => r > 0).sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((sum, rel, i) => sum + rel / Math.log2(i + 2), 0);
  return idcg > 0 ? dcg / idcg : 0;
}

function scoreAtDepth(docids: string[], judged: Map<string, number>, k: number) {
  const head = docids.slice(0, k);
  const hits = head.filter((d) => (judged.get(d) ?? 0) >= REL_THRESHOLD).length;
  const totalRelevant = [...judged.values()].filter((r) => r >= REL_THRESHOLD).length;
  const unjudged = head.filter((d) => !judged.has(d)).length;
  return {
    submitted: head.length,
    relevant: hits,
    precision: head.length > 0 ? hits / head.length : 0,
    recall: totalRelevant > 0 ? hits / totalRelevant : 0,
    ndcg: ndcgAt(docids, judged, k),
    unjudgedRate: head.length > 0 ? unjudged / head.length : 0,
  };
}

const mean = (xs: number[]) => (xs.length > 0 ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

function main() {
  const [runDirArg, qrelsDirArg] = process.argv.slice(2);
  if (!runDirArg || !qrelsDirArg) {
    console.error("usage: npx tsx scripts/eval_submission.ts <run-dir> <qrels-dir>");
    process.exit(1);
  }
  const runDir = resolve(runDirArg);
  const qrelsDir = resolve(qrelsDirArg);

  const submissionPath = join(runDir, "submission_variable_k.trec");
  const poolPath = join(runDir, "retrieval.internal.trec-run.tsv");
  if (!existsSync(submissionPath)) {
    console.error(`missing ${submissionPath}`);
    process.exit(1);
  }
  const submission = readRanked(submissionPath);
  const pool = existsSync(poolPath) ? readRanked(poolPath) : new Map<string, string[]>();

  const qrelsPaths = QRELS_FILES.map((n) => join(qrelsDir, n)).filter((p) => existsSync(p));
  if (qrelsPaths.length === 0) {
    console.error(`no qrels found in ${qrelsDir}`);
    process.exit(1);
  }

  const qids = [...submission.keys()].sort((a, b) => Number(a) - Number(b));
  const ks = qids.map((q) => (submission.get(q) ?? []).length);
  console.log(`submission: ${qids.length} topics, k min/median/max = ${Math.min(...ks)}/${[...ks].sort((a, b) => a - b)[Math.floor(ks.length / 2)]}/${Math.max(...ks)}`);
  console.log(`qrels: ${qrelsPaths.length} files, relevance threshold >= ${REL_THRESHOLD}\n`);

  // Per-qrels aggregate for the submission as-is.
  const perQrels = qrelsPaths.map((p) => {
    const qrels = readQrels(p);
    const rows = qids.map((qid) => scoreAtDepth(submission.get(qid) ?? [], qrels.get(qid) ?? new Map(), Infinity));
    return {
      name: basename(p).replace(/^rag25-climbmix-umbrela-/, "").replace(/\.qrels$/, ""),
      precision: mean(rows.map((r) => r.precision)),
      recall: mean(rows.map((r) => r.recall)),
      ndcg: mean(rows.map((r) => r.ndcg)),
      unjudged: mean(rows.map((r) => r.unjudgedRate)),
    };
  });

  console.log("=== 提交檔(每題自己的 k)===");
  console.log("qrels".padEnd(42) + "P@k".padStart(8) + "R@k".padStart(8) + "nDCG@k".padStart(9) + "未判定率".padStart(10));
  for (const r of perQrels) {
    console.log(r.name.slice(0, 40).padEnd(42) + r.precision.toFixed(4).padStart(8) + r.recall.toFixed(4).padStart(8) + r.ndcg.toFixed(4).padStart(9) + (r.unjudged * 100).toFixed(1).padStart(9) + "%");
  }
  console.log("平均".padEnd(42) + mean(perQrels.map((r) => r.precision)).toFixed(4).padStart(8) + mean(perQrels.map((r) => r.recall)).toFixed(4).padStart(8) + mean(perQrels.map((r) => r.ndcg)).toFixed(4).padStart(9) + (mean(perQrels.map((r) => r.unjudged)) * 100).toFixed(1).padStart(9) + "%");

  // What fixed depths over the same pool would have given, as the comparison the variable-depth rule invites.
  if (pool.size > 0) {
    console.log("\n=== 對照:同一排序改用固定深度 ===");
    console.log("深度".padEnd(10) + "P@k".padStart(8) + "R@k".padStart(8) + "nDCG@k".padStart(9));
    for (const depth of FIXED_DEPTHS) {
      const acc = qrelsPaths.map((p) => {
        const qrels = readQrels(p);
        const rows = qids.map((qid) => scoreAtDepth(pool.get(qid) ?? [], qrels.get(qid) ?? new Map(), depth));
        return { precision: mean(rows.map((r) => r.precision)), recall: mean(rows.map((r) => r.recall)), ndcg: mean(rows.map((r) => r.ndcg)) };
      });
      console.log(`top-${depth}`.padEnd(10) + mean(acc.map((a) => a.precision)).toFixed(4).padStart(8) + mean(acc.map((a) => a.recall)).toFixed(4).padStart(8) + mean(acc.map((a) => a.ndcg)).toFixed(4).padStart(9));
    }
  }

  // Per-topic view: which topics submit too little or too much.
  const qrels0 = readQrels(qrelsPaths[0]);
  console.log("\n=== 逐題(以第一份 qrels 為例,依 P@k 排序)===");
  console.log("題號".padEnd(8) + "k".padStart(4) + "命中".padStart(6) + "該題相關總數".padStart(14) + "P@k".padStart(8) + "R@k".padStart(8));
  const perTopic = qids.map((qid) => {
    const judged = qrels0.get(qid) ?? new Map();
    const s = scoreAtDepth(submission.get(qid) ?? [], judged, Infinity);
    return { qid, k: s.submitted, hits: s.relevant, total: [...judged.values()].filter((r) => r >= REL_THRESHOLD).length, p: s.precision, r: s.recall };
  }).sort((a, b) => a.p - b.p);
  for (const t of perTopic) {
    console.log(t.qid.padEnd(8) + String(t.k).padStart(4) + String(t.hits).padStart(6) + String(t.total).padStart(14) + t.p.toFixed(3).padStart(8) + t.r.toFixed(3).padStart(8));
  }
}

main();
