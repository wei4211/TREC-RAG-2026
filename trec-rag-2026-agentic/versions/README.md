# 版本快照

每個資料夾是該版本的 `src/` 和 `scripts/` 實際檔案,從對應的 git 標籤匯出。
**主資料夾(上一層)永遠是最新版**;這裡是為了方便直接翻閱和比較舊版。

| 資料夾 | 標籤 | 內容 |
|---|---|---|
| `A-official/` | `version-A-official` | 官方對齊:Llama-3.3-70B、Top-5000 候選池、變深度提交、④LLM 改寫、v0.6.0 metadata |
| `B1-breadth/` | `version-B1-breadth` | + 廣度輪流分配預算、面向 8-12、一句一事實短句、禁止刪句、`run_desc` |
| `B2-nugget-loop/` | `version-B2-nugget-loop` | + 完整度優先(15-25 字)、自評補洞迴圈(迭代式 nugget 預測 → 批改 → 定向補洞) |
| (主資料夾) | `version-C-freeform-agent` | + Route B:自由 tool-calling agent(`freeform_agent.ts`) |

## dev22 實測

| 版本 | nDCG@10 | 涵蓋(寬) | 涵蓋(嚴) | 加權支持 | hard 支持 |
|---|---|---|---|---|---|
| 學姊 baseline | 0.654 | 16.7% | — | 75.5% | 68.4% |
| A | 0.7475 | 21.9% | 16.0% | 83.1% | 70.3% |
| B1 | 0.7515 | 26.1% | 16.7% | 93.0% | 90.5% |
| B2 | 跑測中 | | | | |
| Route B | 未跑 | | | | |

## 改進路徑

每一步都由前一步的數據指出,不是憑空加模組:

```
A    官方對齊,建立基準
 ↓   診斷:句數(28)< 每題 vital nugget 數(41),是結構性天花板;
     且預算被前段面向吃光,後段面向被整段截斷
B1   廣度輪流分配 + 一句一事實
 ↓   結果:hard 支持 70.3% → 90.5%,涵蓋 21.9% → 26.1%
     診斷:漏掉的變成「部分命中」而非「完全命中」(partial 11.8%→18.8%,
     support 只從 16.0%→16.7%),句子 9.6 字太短講不完整
B2   完整度優先(15-25 字)+ 自評補洞迴圈
 ↓
C    Route B:改由模型自己決定控制流程,作為架構對照
```

## 重新產生快照

```bash
git archive <tag> trec-rag-2026-agentic/src trec-rag-2026-agentic/scripts \
  | tar -x -C trec-rag-2026-agentic/versions/<name> --strip-components=1
```
