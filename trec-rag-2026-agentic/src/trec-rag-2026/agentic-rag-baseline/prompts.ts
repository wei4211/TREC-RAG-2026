import { AGENTIC_RAG_BASELINE_PROMPT_VERSION, type TopicIdentity } from "./contracts";

export type EvidenceDocumentForPrompt = {
  docid: string;
  text: string;
};

// ④ LLM grounded revision (support upgrade): the model reads each answer sentence together with the
// exact documents it cites, verifies support, rewrites over-claims to match the evidence, weakens or
// drops unsupported claims. Citations are docid strings (v0.6.0 allows this). One call per topic.
export function buildGroundedRevisionPrompt(
  topic: TopicIdentity,
  sentences: { text: string; docids: string[] }[],
  citedDocs: { docid: string; text: string }[],
): string {
  return [
    "You revise a draft answer so every sentence is FULLY supported by the documents it cites.",
    "For each sentence: check whether the cited documents actually state its claim.",
    "  - If the sentence over-claims (says more than the evidence), rewrite it to state only what the cited documents support.",
    "  - If a cited document does not support the sentence, drop that citation. Prefer citing a document that genuinely supports the claim.",
    "  - If no cited document supports the sentence at all, weaken it to the strongest statement the cited documents DO support.",
    "NEVER delete a sentence. Return exactly as many sentences as you were given, in the same order. A weakened sentence is always better than a missing one.",
    "Keep the answer informative and specific; do not add facts that are not in the cited documents.",
    "Citations are ClimbMix docid strings. Each sentence may cite at most three documents.",
    "Return ONLY strict JSON, no prose:",
    '{"answer":[{"text":"...","citations":["shard_00000_00000"]}]}',
    "",
    `Narrative: ${topic.narrative}`,
    "",
    "Cited documents (docid → text):",
    citedDocs.map((d) => `[${d.docid}]\n${d.text}`).join("\n\n"),
    "",
    "Draft answer (each sentence with the docids it currently cites):",
    sentences.map((s, i) => `${i + 1}. ${s.text}  — cites: ${s.docids.join(", ")}`).join("\n"),
  ].join("\n");
}

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

// v4-style comprehensive generation: decompose the narrative into aspects and
// require the answer to cover EVERY aspect, one factual sentence at a time.
// This drives nugget coverage up (short single-pass answers cover ~16%; aspect-
// driven multi-sentence answers cover far more) while staying within the
// official 1024-word / <=3-citations-per-sentence limits.
export function buildComprehensiveAnswerPrompt(input: AnswerGenerationPromptInput & { aspects?: string[] }): string {
  const aspectBlock = input.aspects && input.aspects.length > 0
    ? ["Aspects to cover (write at least one grounded sentence for EACH; do not drop any):", ...input.aspects.map((a, i) => `  ${i + 1}. ${a}`)].join("\n")
    : "First identify every distinct aspect the narrative asks about, then cover each of them.";
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
    "You generate an evidence-grounded TREC RAG answer that comprehensively covers the narrative.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    aspectBlock,
    "Write a thorough answer: aim to cover every aspect above with specific facts, figures, named entities, and details found in the evidence.",
    "Break the answer into many individual factual sentences (typically 12-25 sentences for a multi-aspect narrative).",
    "Prefer more specific, information-dense sentences over few vague ones. Each sentence should add a distinct fact.",
    "Every answer sentence must have citations. Citations are zero-indexed positions into the references array, not docids.",
    "Each sentence may cite at most three references. Only cite a document that genuinely supports that exact sentence.",
    "References must be unique ClimbMix docids and every reference must be cited at least once. Do not include uncited documents.",
    "Keep the complete answer at or below 1024 words.",
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

// Per-aspect (v4-style) sub-answer: given ONE aspect and the documents retrieved
// specifically for it, write a few grounded sentences answering just that aspect.
// Merging these per-aspect sub-answers is what drives high nugget coverage
// (each aspect gets its own targeted evidence instead of sharing one pool).
export function buildAspectAnswerPrompt(input: { topic: TopicIdentity; aspect: string; documents: EvidenceDocumentForPrompt[]; alreadyWritten?: string[] }): string {
  const already = input.alreadyWritten && input.alreadyWritten.length > 0
    ? ["Already written for this aspect (do NOT repeat these; only add NEW distinct facts not covered here):", ...input.alreadyWritten.map((s) => `  - ${s}`), ""].join("\n")
    : "";
  return [
    "You write a few evidence-grounded sentences answering ONE specific aspect of a TREC RAG narrative.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    `Aspect to answer: ${input.aspect}`,
    already,
    "Write 2-5 factual sentences that specifically answer this aspect.",
    "CRITICAL — each sentence must state a SPECIFIC, checkable claim: a concrete fact, statistic, date, named entity, mechanism, or an explicit enumerated list taken from the evidence.",
    "Do NOT write generic overview sentences that only describe the topic in the abstract (e.g. 'This is an important and dynamic field', 'Sports are part of culture', 'This raises many questions'). Such sentences are worthless and must be omitted.",
    "Write exactly ONE fact per sentence, and state that fact COMPLETELY. Typical length is 15-25 words.",
    "COMPLETENESS IS THE PRIORITY. A fact stated only partially scores nothing at all, so never compress a fact down to a fragment. It is better to spend more words and state one fact fully than to state two facts halfway.",
    "Concretely, when you state a fact: if it is a list, name EVERY item in it; if it involves a quantity, give the number with its unit and its time period; if it names an entity, give the full name; if it is a mechanism, say what causes what.",
    "Do NOT combine several distinct facts into one compound sentence — two facts means two sentences, each complete.",
    "Good (one complete fact): 'Major nuclear accidents include Chernobyl in 1986, Three Mile Island in 1979, and Fukushima in 2011.'",
    "Bad (same fact left incomplete): 'Chernobyl was a major nuclear accident.'",
    "Bad (two facts crammed together, neither complete): 'Nuclear accidents have occurred and they affect public opinion and policy.'",
    "Every sentence must have citations. Citations are zero-indexed positions into the references array, not docids.",
    "Each sentence may cite at most three references. Only cite a document that genuinely supports that exact sentence.",
    "References must be unique ClimbMix docids; every reference must be cited at least once; do not include uncited documents.",
    "If the evidence does not cover this aspect at all, return an empty answer array.",
    "Return only strict JSON. No Markdown fences, no prose.",
    "",
    "Required JSON shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

// v4-style reflection: after a draft answer is assembled, ask the model which
// aspects of the narrative are still missing or thinly covered, so we can run
// extra targeted retrieval+generation passes for them.
export function buildReflectionPrompt(topic: TopicIdentity, answerText: string): string {
  return [
    "You review a draft answer against a narrative and identify what is still MISSING or only thinly covered.",
    "Return ONLY strict JSON, no prose:",
    '{"gaps":["missing aspect one","missing aspect two"]}',
    "Rules: 0-2 gaps; each a short noun phrase (3-10 words) naming a distinct aspect the narrative asks about but the draft does not adequately cover; if the draft already covers everything well, return an empty array.",
    "",
    formatTopic(topic),
    "Draft answer:",
    answerText,
  ].join("\n");
}

// One cheap LLM call to list the distinct aspects a narrative asks about.
export function buildAspectDecompositionPrompt(topic: TopicIdentity): string {
  return [
    "List every distinct aspect the following narrative asks about, so that together they cover the whole question.",
    "Return ONLY a strict JSON object, no prose:",
    '{"aspects":["aspect one","aspect two"]}',
    "Rules: 8-12 aspects; each a short noun phrase (3-10 words); split the narrative FINELY so each aspect maps to a distinct sub-question a good answer must address; cover every distinct thing asked; do not overlap; do not add aspects the narrative does not ask about.",
    "Prefer MORE, NARROWER aspects over few broad ones: the answer is scored on how many distinct required facts it covers, so a fine split gives each fact its own targeted evidence.",
    "",
    formatTopic(topic),
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
