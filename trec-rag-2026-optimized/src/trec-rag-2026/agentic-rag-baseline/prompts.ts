import { AGENTIC_RAG_BASELINE_PROMPT_VERSION, type TopicIdentity } from "./contracts";

export type EvidenceDocumentForPrompt = {
  docid: string;
  text: string;
};

export type JudgePromptInput = {
  topic: TopicIdentity;
  iteration: number;
  maxIterations: number;
  previousQueries: string[];
  documents: EvidenceDocumentForPrompt[];
};

export type FollowupQueryPromptInput = {
  topic: TopicIdentity;
  previousQueries: string[];
  missingAspects: string[];
  recommendedFollowupFocus?: string;
};

export type AnswerGenerationPromptInput = {
  topic: TopicIdentity;
  documents: EvidenceDocumentForPrompt[];
};

export function buildJudgePrompt(input: JudgePromptInput): string {
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
    "You are an evidence sufficiency judge for a TREC RAG baseline.",
    "Use only the provided topic and evidence documents. Do not use outside knowledge.",
    "Decide whether the current evidence is sufficient to answer all important aspects of the narrative.",
    "Do not write the final answer.",
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"enough":false,"confidence":0.0,"covered_aspects":["..."],"missing_aspects":["..."],"needs_followup":true,"recommended_followup_focus":"...","rationale":"..."}',
    "",
    formatTopic(input.topic),
    `Iteration: ${input.iteration} of ${input.maxIterations}`,
    `Previous queries: ${JSON.stringify(input.previousQueries)}`,
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export function buildFollowupQueryPrompt(input: FollowupQueryPromptInput): string {
  return [
    "Return exactly one strict JSON object and nothing else:",
    '{"subquery":"..."}',
    "The subquery must be a concise BM25 keyword query, 3-160 characters, no Markdown, no code, no braces in the value.",
    "Use only the missing aspects and recommended focus. Do not explain.",
    `Previous queries: ${JSON.stringify(input.previousQueries)}`,
    `Missing aspects: ${JSON.stringify(input.missingAspects)}`,
    `Recommended focus: ${input.recommendedFollowupFocus ?? ""}`,
  ].join("\n");
}

export function buildAnswerGenerationPrompt(input: AnswerGenerationPromptInput): string {
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
    "You generate an evidence-grounded TREC RAG answer.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    "Break the answer into individual factual sentences.",
    "Every answer sentence must have citations.",
    "Citations are zero-indexed positions into the references array, not docids.",
    "Each sentence may cite at most three references.",
    "References must be unique ClimbMix docids and every reference must be cited at least once.",
    "Do not include uncited retrieved documents in references.",
    "Keep the complete answer under 1024 words.",
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export function buildCompactAnswerGenerationPrompt(input: AnswerGenerationPromptInput): string {
  return [
    "Return only this JSON shape, with no Markdown or explanation:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "Use only the provided docs. Every sentence needs citations. Citation numbers are references array indexes. Use only cited docids in references.",
    formatTopic(input.topic),
    "Docs:",
    formatDocuments(input.documents.slice(0, 5).map((doc) => ({ ...doc, text: doc.text.slice(0, 800) }))),
  ].join("\n");
}

function formatTopic(topic: TopicIdentity): string {
  return [
    "Topic:",
    `narrative_id: ${topic.qid}`,
    `title: ${topic.title}`,
    "narrative:",
    topic.narrative,
  ].join("\n");
}

function formatDocuments(documents: EvidenceDocumentForPrompt[]): string {
  if (documents.length === 0) return "(none)";
  return documents
    .map((document, index) => [
      `Document ${index}:`,
      `docid: ${document.docid}`,
      "text:",
      document.text,
    ].join("\n"))
    .join("\n\n");
}
