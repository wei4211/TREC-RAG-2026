#!/usr/bin/env python3
"""
Self-contained RAG self-evaluation (nugget coverage + citation support).

Faithful inlining of RAGDoll's NUGGET_ASSIGNER + SUPPORT_EVAL prompts and the
scoring logic from evaluate_official.py, with ZERO ragdoll import so it runs
anywhere with only the Python standard library. Judge = NCHC gpt-oss-120b (the official judge).
This is a SELF-CHECK, not the organizer's evaluation.

Env vars (all overridable):
  NCHC_API_KEY  (or NCHC_GENAI_API_KEY)   -- required, judge LLM
  PYSERINI_API_TOKEN                        -- required, fetch cited doc text
  RAG_OUTPUT_FILE   default: rag_output_trec_rag_2026.jsonl
  NUGGETS_FILE      default: rag25-dev-nuggets.jsonl
  REPORT_FILE       default: official-eval-report.json
  DOC_CACHE_FILE    default: eval_doc_cache.json
  EVAL_CACHE_DIR    default: eval_cache
  LLM_BASE_URL      default: https://portal.genai.nchc.org.tw/api/v1/
  MODEL             default: gpt-oss-120b
  PYSERINI_BASE_URL default: http://api.castorini.uwaterloo.ca
  PYSERINI_INDEX    default: climbmix-400b
"""
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

LLM_API_KEY = os.environ.get("NCHC_API_KEY", "") or os.environ.get("NCHC_GENAI_API_KEY", "")
PYSERINI_API_TOKEN = os.environ.get("PYSERINI_API_TOKEN", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1/")
MODEL = os.environ.get("MODEL", "gpt-oss-120b")
PYSERINI_BASE_URL = os.environ.get("PYSERINI_BASE_URL", "http://api.castorini.uwaterloo.ca")
PYSERINI_INDEX = os.environ.get("PYSERINI_INDEX", "climbmix-400b")

RAG_OUTPUT_FILE = os.environ.get("RAG_OUTPUT_FILE", "rag_output_trec_rag_2026.jsonl")
NUGGETS_FILE    = os.environ.get("NUGGETS_FILE", "rag25-dev-nuggets.jsonl")
REPORT_FILE     = os.environ.get("REPORT_FILE", "official-eval-report.json")
DOC_CACHE_FILE  = os.environ.get("DOC_CACHE_FILE", "eval_doc_cache.json")
EVAL_CACHE_DIR  = os.environ.get("EVAL_CACHE_DIR", "eval_cache")
MAX_RETRIES = 6

# ── inlined RAGDoll prompts ──────────────────────────────────────────────────
NUGGET_ASSIGNER_SYSTEM = ("You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of "
                          "atomic nuggets based on if they are captured by a given passage.")
NUGGET_ASSIGNER_USER = """\
Based on the query and passage, label each of the {num_nuggets} nuggets either as \
support, partial_support, or not_support using the following criteria. \
A nugget that is fully captured in the passage should be labeled as support. \
A nugget that is partially captured in the passage should be labeled as partial_support. \
If the nugget is not captured at all, label it as not_support. Return the list of labels in a \
Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. \
Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {context}
Nugget List: {nuggets}
Only return the list of labels (List[str]). Do not explain.
Labels:"""
SUPPORT_EVAL_PROMPT = """In this task, you will evaluate whether each statement is supported by its corresponding citations. Note
that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are
not easy to discern at first glance. Pay close attention to the text.

You will be provided with a statement and its corresponding passage which the statement cites. It may be
helpful to ask yourself whether it is accurate to say "according to the citation ..." with the statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

- Full Support: All of the information in the statement is supported in the citation.
- Partial Support: Some parts of the information are supported in the citation, but other parts are missing.
- No Support: The citation does not support any part of the statement.

Please provide your response based on the information in the citation. If you are unsure, use your best
judgment. Respond as either "Full Support", "Partial Support", or "No Support" with no additional
information.
Statement: {statement}
Citation: {citation}
"""

def render_assign_prompt(query, context, nuggets):
    return NUGGET_ASSIGNER_USER.format(query=query, context=context, nuggets=nuggets, num_nuggets=len(nuggets))

def render_support_prompt(statement, citation):
    return SUPPORT_EVAL_PROMPT.format(statement=statement, citation=citation)

def parse_support_label(text):
    low = text.lower()
    if "full support" in low: return "FS"
    if "partial support" in low: return "PS"
    if "no support" in low: return "NS"
    return None

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}
def clean_response(text): return text.replace("```python", "").replace("```", "").strip()
def parse_label_list(text):
    cleaned = clean_response(text)
    o, c = cleaned.find("["), cleaned.rfind("]")
    if o == -1 or c == -1 or c <= o: return None
    body = cleaned[o+1:c]; items = []; i, n = 0, len(body)
    while i < n:
        while i < n and (body[i] == "," or body[i].isspace()): i += 1
        if i >= n: break
        ch = body[i]
        if ch in ("'", '"'):
            q = ch; i += 1; buf = []; closed = False
            while i < n:
                cur = body[i]
                if cur == "\\" and i+1 < n: buf.append(_ESCAPES.get(body[i+1], body[i+1])); i += 2; continue
                if cur == q: i += 1; closed = True; break
                buf.append(cur); i += 1
            if not closed: return None
            items.append("".join(buf))
        else:
            buf = []
            while i < n and body[i] != ",": buf.append(body[i]); i += 1
            t = "".join(buf).strip()
            if t: items.append(t)
    return items

# ── doc fetch + LLM call ─────────────────────────────────────────────────────
def load_doc_cache():
    if Path(DOC_CACHE_FILE).exists():
        with open(DOC_CACHE_FILE, encoding="utf-8") as f: return json.load(f)
    return {}
def save_doc_cache(cache):
    Path(DOC_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_CACHE_FILE, "w", encoding="utf-8") as f: json.dump(cache, f, ensure_ascii=False)
def fetch_doc_text(docid, cache):
    if docid in cache: return cache[docid]
    url = f"{PYSERINI_BASE_URL}/v1/{PYSERINI_INDEX}/doc/{urllib.parse.quote(docid)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {PYSERINI_API_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: data = json.loads(r.read())
        doc = data.get("doc", "")
        text = (doc.get("text") or doc.get("contents") or json.dumps(doc)) if isinstance(doc, dict) else str(doc)
    except Exception as exc:
        text = ""; print(f"    [WARN] failed to fetch doc {docid}: {exc}")
    cache[docid] = text; return text
class _Client:
    """Minimal OpenAI-compatible chat client over urllib (no openai/pydantic dependency)."""
    def __init__(self, api_key, base_url):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}", "x-api-key": api_key}

def call_llm(client, system, user, retries=MAX_RETRIES):
    payload = {"model": MODEL, "max_tokens": 2048,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(client.url, data=data, headers=client.headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read())
            return body["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            wait = min(5 * (2 ** attempt), 60)
            print(f"    [RETRY {attempt+1}/{retries}] {exc} (waiting {wait}s)"); time.sleep(wait)
    return ""

# ── scoring ──────────────────────────────────────────────────────────────────
def calculate_nugget_scores(assignments):
    if not assignments: return {"strict_vital_score": 0.0, "vital_score": 0.0}
    strict = sum(1.0 for a in assignments if a == "support")
    lenient = sum(1.0 if a == "support" else 0.5 if a == "partial_support" else 0.0 for a in assignments)
    n = len(assignments)
    return {"strict_vital_score": strict/n, "vital_score": lenient/n}

NUGGET_CHUNK_SIZE = 15
def _assign_chunk(client, query, answer_text, nuggets):
    raw = call_llm(client, NUGGET_ASSIGNER_SYSTEM, render_assign_prompt(query, answer_text, nuggets))
    if not raw: print("    [FAIL] nugget assign chunk: retries exhausted"); return None
    labels = parse_label_list(raw)
    if labels is None or len(labels) != len(nuggets):
        print(f"    [FAIL] parse mismatch (got {len(labels) if labels else 0}, want {len(nuggets)})"); return None
    out = []
    for label in labels:
        low = label.strip().lower()
        if "partial" in low: out.append("partial_support")
        elif "not" in low or "no_support" in low: out.append("not_support")
        elif "support" in low: out.append("support")
        else: out.append("not_support")
    return out
def assign_nuggets(client, query, answer_text, nuggets):
    if not nuggets: return []
    alll = []
    for start in range(0, len(nuggets), NUGGET_CHUNK_SIZE):
        chunk = nuggets[start:start+NUGGET_CHUNK_SIZE]
        if len(nuggets) > NUGGET_CHUNK_SIZE:
            print(f"    assigning chunk {start//NUGGET_CHUNK_SIZE+1} ({len(chunk)} nuggets)...")
        labels = _assign_chunk(client, query, answer_text, chunk)
        if labels is None: return None
        alll.extend(labels)
    return alll

SUPPORT_SCORE_MAP = {"FS": 2, "PS": 1, "NS": 0}
WEIGHTED = {2: 1.0, 1: 0.5, 0: 0.0}
HARD = {2: 1.0, 1: 0.0, 0: 0.0}
def judge_citation_support(client, statement, citation_text):
    return parse_support_label(call_llm(client, "", render_support_prompt(statement, citation_text[:4000])))
def compute_support_metrics(scores):
    w, h, n = 0.0, 0.0, 0; total = len(scores)
    for s in scores:
        if s > -1: w += WEIGHTED[s]; h += HARD[s]; n += 1
        else: total -= 1
    return {"weighted_precision_first_citation": w/n if n else 0.0,
            "weighted_recall_first_citation": w/total if total else 0.0,
            "hard_precision": h/n if n else 0.0,
            "hard_recall": h/total if total else 0.0,
            "num_judged": n, "num_missing": len(scores)-n}

def main():
    if not LLM_API_KEY: raise SystemExit("ERROR: set NCHC_API_KEY (or NCHC_GENAI_API_KEY).")
    if not PYSERINI_API_TOKEN: raise SystemExit("ERROR: set PYSERINI_API_TOKEN.")
    nuggets_by_qid = {}
    with open(NUGGETS_FILE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            nuggets_by_qid[obj["qid"]] = [n["text"] for n in obj.get("nuggets", []) if n.get("importance") == "vital"]
    rag_outputs = [json.loads(l) for l in open(RAG_OUTPUT_FILE, encoding="utf-8")]
    client = _Client(LLM_API_KEY, LLM_BASE_URL)
    doc_cache = load_doc_cache(); Path(EVAL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    report = []
    for i, obj in enumerate(rag_outputs):
        qid = obj["metadata"]["narrative_id"]; narrative = obj["metadata"]["narrative"]
        answer = obj["answer"]; references = obj["references"]
        answer_text = " ".join(s["text"] for s in answer)
        cache_file = Path(EVAL_CACHE_DIR) / f"topic_{qid}.json"
        if cache_file.exists():
            print(f"[{i+1}/{len(rag_outputs)}] Topic {qid} (cached)")
            report.append(json.load(open(cache_file, encoding="utf-8"))); continue
        print(f"[{i+1}/{len(rag_outputs)}] Topic {qid}")
        vital = nuggets_by_qid.get(qid, [])
        print(f"    assigning {len(vital)} vital nuggets...")
        assignments = assign_nuggets(client, narrative, answer_text, vital)
        if assignments is None: print(f"    [SKIP] topic {qid} assignment failed"); continue
        nscore = calculate_nugget_scores(assignments)
        print(f"    fetching {len(references)} cited documents...")
        for docid in references: fetch_doc_text(docid, doc_cache)
        save_doc_cache(doc_cache)
        fscores = []
        for sent in answer:
            if not sent["citations"]: fscores.append(-1); continue
            docid = references[sent["citations"][0]]; dt = doc_cache.get(docid, "")
            if not dt: fscores.append(-1); continue
            label = judge_citation_support(client, sent["text"], dt)
            fscores.append(SUPPORT_SCORE_MAP[label] if label else -1)
        sscore = compute_support_metrics(fscores)
        print(f"    vital={nscore['vital_score']:.2f} strict={nscore['strict_vital_score']:.2f} "
              f"w_prec={sscore['weighted_precision_first_citation']:.2f} "
              f"hard_prec={sscore['hard_precision']:.2f} missing={sscore['num_missing']}")
        entry = {"narrative_id": qid, "num_vital_nuggets": len(vital),
                 "nugget_assignments": list(zip(vital, assignments)), "nugget_scores": nscore,
                 "num_sentences": len(answer), "support_scores": sscore}
        json.dump(entry, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        report.append(entry); time.sleep(0.3)
    Path(REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(REPORT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    mean = lambda xs: sum(xs)/len(xs) if xs else 0.0
    ov = mean([r["nugget_scores"]["vital_score"] for r in report])
    os_ = mean([r["nugget_scores"]["strict_vital_score"] for r in report])
    op = mean([r["support_scores"]["weighted_precision_first_citation"] for r in report])
    oh = mean([r["support_scores"]["hard_precision"] for r in report])
    orr = mean([r["support_scores"]["weighted_recall_first_citation"] for r in report])
    print(f"\n{'='*70}\nReport -> {REPORT_FILE}  (topics scored: {len(report)})")
    print(f"Vital nugget coverage (lenient):   {ov:.1%}")
    print(f"Vital nugget coverage (strict):    {os_:.1%}")
    print(f"Citation weighted precision:       {op:.1%}")
    print(f"Citation HARD precision (full):    {oh:.1%}")
    print(f"Citation weighted recall:          {orr:.1%}")
    print("\nPer-topic (lowest coverage first):")
    for r in sorted(report, key=lambda r: r["nugget_scores"]["vital_score"]):
        print(f"  {r['narrative_id']:>6}  vital={r['nugget_scores']['vital_score']:.0%}  "
              f"strict={r['nugget_scores']['strict_vital_score']:.0%}  "
              f"w_prec={r['support_scores']['weighted_precision_first_citation']:.0%}  "
              f"hard_prec={r['support_scores']['hard_precision']:.0%}")

if __name__ == "__main__":
    main()
