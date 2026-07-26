# TREC RAG 2026 — Agentic RAG Pipelines

**CFDA Lab · Academia Sinica · TREC RAG 2026 track**

在學姊的 agentic RAG baseline 上,針對 **R(檢索)** 與 **RAG(生成)** 兩個任務各新增模組。
TypeScript / Node.js;LLM 走 NCHC GenAI Portal(Llama-4-Maverick),檢索走 Pyserini REST(ClimbMix)。

---

## 架構

共用檢索前端(BM25 + 加權 RRF + agentic 判斷迴圈,**沿用 baseline**)→ 分流成 R 與 RAG 兩個輸出。
在檢索與生成兩端各新增 2 個模組,共 4 個:

| 模組 | 任務 | 作用 |
|---|---|---|
| **① Query2Doc 假文件擴展** | R / RAG | LLM 先生成假設答案(含具名實體/日期)擴展查詢 → 撈到字面不符但語意相關的文件,提 **recall** |
| **② 三軌融合重排(受控)** | R | top-100 內把 BM25 名次 + CrossEncoder 名次 + dense(bge-m3)名次用 RRF 融合;集合固定只換順序 → 提 **nDCG**、recall 不變 |
| **③ per-aspect 生成** | RAG | 把題目拆成面向,每面向各自多輪檢索 + reflection 補漏 + 要求寫具體列舉式句子 → 提 **coverage** |
| **④ re-attribution 引用驗證** | RAG | 用 BGE-Reranker 幫每句找出最支持它的證據、把引用改指過去(不砍句子)→ 提 **support**、不犧牲 coverage |

---

## 版本資料夾

| 資料夾 | 內容 |
|---|---|
| `trec-rag-2026-agentic-rag-baseline-minimal-*` | 學姊 baseline(原封未改,作為基準) |
| `trec-rag-2026-no-ce` | baseline 方法 + Llama-4-Maverick(乾淨對照組) |
| `trec-rag-2026-optimized` | 早期 CrossEncoder 重排實驗 |
| `trec-rag-2026-q2d` | + ① Query2Doc |
| `trec-rag-2026-q2d-gen` | + ③ per-aspect 生成 |
| `trec-rag-2026-q2d-fusion` | + ② 三軌融合重排(R-only 快速測試) |
| `trec-rag-2026-q2d-max` | **最強版:四模組整合(① + ② + ③ + ④)** |

---

## 結果(dev 22 題,同一模型 Llama-4-Maverick)

### R 任務(三份官方 dev qrels 取平均)

| 配置 | nDCG@10 | Recall@1000 | MRR |
|---|---|---|---|
| 學姊 baseline | 0.654 | 0.269 | 0.915 |
| + Query2Doc | 0.701 | 0.298 | 0.951 |
| **+ 三軌融合重排(最終)** | **0.763** | **0.302** | **0.978** |

nDCG@10 **+0.109**、Recall@1000 同步上升;受控重排使 Recall@100 依設計不變。

### RAG 任務(RAGDoll 官方 prompt × gpt-oss 判官)

| 配置 | 涵蓋(寬) | 加權引用支持 | 完全引用支持 |
|---|---|---|---|
| 學姊 baseline | 16.7% | 75.5% | 68.4% |
| + per-aspect 生成 | 33.4% | 79.2% | 64.4% |
| **+ re-attribution(最終)** | 30.1% | **81.0%** | 66.5% |

涵蓋率**翻倍**、加權支持 **+5.5pp**;完全支持持平(答案變長的偏涵蓋取捨,由 re-attribution 救回)。

---

## 執行

在任一版本資料夾:

\`\`\`bash
npm install
# 建立 .env.local,填入 NCHC_API_KEY 與 PYSERINI_API_TOKEN
TOPICS_PATH=<rag25-topics-dev.tsv> QRELS_DIR=<rag25-dev-umbrela-qrels/> \
LLM_MODEL="Llama-4-Maverick-17B-128E-Instruct-FP8" bash scripts/run_dev22.sh
\`\`\`

R 任務快速測試(跳過生成)可加 \`R_ONLY=1\`。

> **注意**:\`.env.local\`(金鑰)與 \`runs/\`(大型實驗輸出)不納入版控。
