import {
  CLIMBMIX_DOCID_RE,
  type AgenticRagBaselineConfig,
  type AgenticRagOutputObject,
  type RagReferenceNormalizationResult,
  type TopicIdentity,
  type ValidationIssue,
  type ValidationResult,
} from "./contracts";

export class AgenticRagValidationError extends Error {
  readonly issues: ValidationIssue[];

  constructor(message: string, issues: ValidationIssue[]) {
    super(message);
    this.name = "AgenticRagValidationError";
    this.issues = issues;
  }
}

export function parseRagOutputObjectJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new AgenticRagValidationError("RAG output is not valid JSON.", [
      { code: "INVALID_JSON", message: error instanceof Error ? error.message : String(error) },
    ]);
  }
}

export function normalizeRagOutputObjectReferences(
  value: unknown,
  args: { config: AgenticRagBaselineConfig; topic: TopicIdentity; readDocids: Set<string> },
): RagReferenceNormalizationResult {
  const draft = validateRagOutputObject(value, args, { allowUncitedReferences: true });
  const citedReferenceIndices = new Set<number>();
  for (const sentence of draft.answer) for (const citation of sentence.citations) citedReferenceIndices.add(citation);

  const oldToNew = new Map<number, number>();
  const references: string[] = [];
  draft.references.forEach((reference, oldIndex) => {
    if (!citedReferenceIndices.has(oldIndex)) return;
    oldToNew.set(oldIndex, references.length);
    references.push(reference);
  });

  const normalized: AgenticRagOutputObject = {
    metadata: draft.metadata,
    references,
    answer: draft.answer.map((sentence) => ({
      text: sentence.text,
      citations: sentence.citations.map((citation) => {
        const remapped = oldToNew.get(citation);
        if (remapped === undefined) {
          throw new AgenticRagValidationError("Citation could not be remapped after reference normalization.", [
            { code: "CITATION_REMAP_FAILED", message: `citation ${citation} has no retained reference` },
          ]);
        }
        return remapped;
      }),
    })),
  };

  const strictlyValidated = validateRagOutputObject(normalized, args);
  return { ragObject: strictlyValidated, removedReferenceCount: draft.references.length - references.length };
}

export function validateRagOutputObjectStrict(
  value: unknown,
  args: { config: AgenticRagBaselineConfig; topic: TopicIdentity; readDocids: Set<string> },
): ValidationResult {
  const issues = collectValidationIssues(value, args, { allowUncitedReferences: false });
  return { ok: issues.length === 0, issues };
}

export function validateRagOutputObject(
  value: unknown,
  args: { config: AgenticRagBaselineConfig; topic: TopicIdentity; readDocids: Set<string> },
  options: { allowUncitedReferences?: boolean } = {},
): AgenticRagOutputObject {
  const issues = collectValidationIssues(value, args, options);
  if (issues.length > 0) {
    throw new AgenticRagValidationError("Invalid RAG output object.", issues);
  }
  return value as AgenticRagOutputObject;
}

function collectValidationIssues(
  value: unknown,
  args: { config: AgenticRagBaselineConfig; topic: TopicIdentity; readDocids: Set<string> },
  options: { allowUncitedReferences?: boolean },
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (!isRecord(value)) return [{ code: "INVALID_SHAPE", message: "RAG output must be an object." }];
  requireExactKeys(value, ["metadata", "references", "answer"], "top-level", issues);

  const metadata = value.metadata;
  if (!isRecord(metadata)) {
    issues.push({ code: "INVALID_METADATA", message: "metadata must be an object." });
  } else {
    requireExactKeys(metadata, ["team_id", "run_id", "type", "narrative_id", "title", "narrative", "prompt", "generator", "retrieval_depth"], "metadata", issues);
    if (typeof metadata.team_id !== "string" || metadata.team_id.trim() === "") {
      issues.push({ code: "INVALID_TEAM_ID", message: "metadata.team_id must be non-empty." });
    }
    if (metadata.run_id !== args.config.runId) {
      issues.push({ code: "RUN_ID_MISMATCH", message: "metadata.run_id must match config run_id." });
    }
    if (metadata.type !== "automatic") {
      issues.push({ code: "INVALID_TYPE", message: 'metadata.type must equal "automatic".' });
    }
    if (metadata.narrative_id !== args.topic.qid) {
      issues.push({ code: "NARRATIVE_ID_MISMATCH", message: "metadata.narrative_id must preserve qid exactly." });
    }
    if (metadata.title !== "") {
      issues.push({ code: "INVALID_DEV_TITLE", message: 'metadata.title must equal "" in dev mode.' });
    }
    if (metadata.narrative !== args.topic.narrative) {
      issues.push({ code: "NARRATIVE_MISMATCH", message: "metadata.narrative must preserve narrative exactly." });
    }
    if (metadata.prompt !== args.config.promptVersion) {
      issues.push({ code: "PROMPT_VERSION_MISMATCH", message: "metadata.prompt must match configured prompt version." });
    }
  }

  const references = value.references;
  if (!Array.isArray(references)) {
    issues.push({ code: "INVALID_REFERENCES", message: "references must be an array." });
  }
  const referenceStrings = Array.isArray(references) ? validateReferences(references, args.readDocids, issues) : [];

  const citedReferenceIndices = new Set<number>();
  const answer = value.answer;
  if (!Array.isArray(answer) || answer.length === 0) {
    issues.push({ code: "INVALID_ANSWER", message: "answer must be a non-empty array." });
  } else {
    answer.forEach((sentence, sentenceIndex) => {
      if (!isRecord(sentence)) {
        issues.push({ code: "INVALID_ANSWER_SENTENCE", message: `answer[${sentenceIndex}] must be an object.` });
        return;
      }
      requireExactKeys(sentence, ["text", "citations"], `answer[${sentenceIndex}]`, issues);
      if (typeof sentence.text !== "string" || sentence.text.trim() === "") {
        issues.push({ code: "INVALID_SENTENCE_TEXT", message: `answer[${sentenceIndex}].text must be non-empty.` });
      }
      if (!Array.isArray(sentence.citations) || sentence.citations.length === 0) {
        issues.push({ code: "MISSING_CITATIONS", message: `answer[${sentenceIndex}].citations must be non-empty.` });
        return;
      }
      const seenInSentence = new Set<number>();
      sentence.citations.forEach((citation, citationIndex) => {
        if (!Number.isInteger(citation)) {
          issues.push({ code: "INVALID_CITATION", message: `answer[${sentenceIndex}].citations[${citationIndex}] must be an integer.` });
          return;
        }
        if (citation < 0 || citation >= referenceStrings.length) {
          issues.push({ code: "CITATION_OUT_OF_RANGE", message: `answer[${sentenceIndex}].citations[${citationIndex}] is out of range.` });
          return;
        }
        if (seenInSentence.has(citation)) {
          issues.push({ code: "DUPLICATE_SENTENCE_CITATION", message: `answer[${sentenceIndex}] has duplicate citation ${citation}.` });
        }
        seenInSentence.add(citation);
        citedReferenceIndices.add(citation);
      });
    });
  }

  if (!options.allowUncitedReferences) {
    referenceStrings.forEach((_reference, index) => {
      if (!citedReferenceIndices.has(index)) {
        issues.push({ code: "UNCITED_REFERENCE", message: `references[${index}] is not cited by any answer sentence.` });
      }
    });
  }

  return issues;
}

function validateReferences(references: unknown[], readDocids: Set<string>, issues: ValidationIssue[]): string[] {
  const referenceStrings: string[] = [];
  const seen = new Set<string>();
  references.forEach((reference, index) => {
    if (typeof reference !== "string" || reference.trim() === "") {
      issues.push({ code: "INVALID_REFERENCE", message: `references[${index}] must be a non-empty string.` });
      return;
    }
    if (reference !== reference.trim()) {
      issues.push({ code: "INVALID_REFERENCE", message: `references[${index}] must not contain leading/trailing whitespace.` });
    }
    if (!CLIMBMIX_DOCID_RE.test(reference)) {
      issues.push({ code: "INVALID_CLIMBMIX_DOCID", message: `references[${index}] is not a ClimbMix docid.` });
    }
    if (seen.has(reference)) issues.push({ code: "DUPLICATE_REFERENCE", message: `Duplicate reference: ${reference}` });
    seen.add(reference);
    if (!readDocids.has(reference)) {
      issues.push({ code: "REFERENCE_NOT_READ", message: `Cited reference was not in read docids: ${reference}` });
    }
    referenceStrings.push(reference);
  });
  return referenceStrings;
}

function requireExactKeys(value: Record<string, unknown>, expectedKeys: string[], label: string, issues: ValidationIssue[]): void {
  const expected = new Set(expectedKeys);
  for (const key of Object.keys(value)) {
    if (!expected.has(key)) issues.push({ code: "EXTRA_FIELD", message: `Unexpected field in ${label}: ${key}` });
  }
  for (const key of expectedKeys) {
    if (!Object.hasOwn(value, key)) issues.push({ code: "MISSING_FIELD", message: `Missing field in ${label}: ${key}` });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
