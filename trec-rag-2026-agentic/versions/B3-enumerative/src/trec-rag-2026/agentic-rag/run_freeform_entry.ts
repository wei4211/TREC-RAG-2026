import { pathToFileURL } from "node:url";
import { runFreeformAgenticRag, type FreeformOptions } from "./freeform_agent";

function str(v: unknown) { return typeof v === "string" ? v.trim() : ""; }
function req(v: unknown, n: string) { const s = str(v); if (!s) throw new Error(`Missing ${n}`); return s; }

function parse(argv: string[], env: NodeJS.ProcessEnv = process.env): FreeformOptions {
  const r: Record<string, string | boolean | undefined> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--force") { r.force = true; continue; }
    if (!a.startsWith("--")) throw new Error(`Unexpected ${a}`);
    const v = argv[++i];
    if (!v || v.startsWith("--")) throw new Error(`Missing value for ${a}`);
    r[a.slice(2)] = v;
  }
  return {
    runId: req(r["run-id"], "--run-id"),
    teamId: str(r["team-id"]) || "pi-serini",
    outputDir: req(r["output-dir"], "--output-dir"),
    topicsPath: req(r.topics, "--topics"),
    qrelsDir: req(r["qrels-dir"], "--qrels-dir"),
    pyseriniBaseUrl: str(r["pyserini-base-url"]) || env.PYSERINI_BASE_URL || "http://api.castorini.uwaterloo.ca",
    pyseriniIndex: str(r["pyserini-index"]) || env.PYSERINI_INDEX || "climbmix-400b",
    pyseriniTokenEnv: str(r["pyserini-token-env"]) || "PYSERINI_API_TOKEN",
    limitTopics: r["limit-topics"] ? Number(r["limit-topics"]) : undefined,
    llm: {
      provider: "nchc_llm",
      model: str(r["llm-model"]) || "Llama-3.3-70B-Instruct",
      apiKeyEnv: str(r["llm-api-key-env"]) || "NCHC_API_KEY",
      baseUrl: env.NCHC_BASE_URL || "https://portal.genai.nchc.org.tw/api/v1",
      temperature: 0,
      maxTokens: 2048,
    },
    force: r.force === true,
    env,
  };
}

async function main() { console.log(JSON.stringify(await runFreeformAgenticRag(parse(process.argv.slice(2))), null, 2)); }
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e instanceof Error ? e.message : String(e)); process.exitCode = 1; });
}
