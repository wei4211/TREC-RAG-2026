export const AGENTIC_RAG_BASELINE_PROMPT_VERSION = "agentic_rag_baseline_v1";
export const CLIMBMIX_DOCID_RE = /^shard_\d+_\d+$/;

export type AgenticRagBaselineMode = "dev";

export type AgenticRagOutputMetadata = {
  team_id: string;
  run_id: string;
  type: "automatic";
  narrative_id: string;
  title: string;
  narrative: string;
  prompt: string;
};

export type AgenticRagAnswerSentence = {
  text: string;
  citations: number[];
};

export type AgenticRagOutputObject = {
  metadata: AgenticRagOutputMetadata;
  references: string[];
  answer: AgenticRagAnswerSentence[];
};

export type AgenticRagBaselineConfig = {
  runId: string;
  teamId: string;
  mode: AgenticRagBaselineMode;
  promptVersion: string;
};

export type TopicIdentity = {
  qid: string;
  title: string;
  narrative: string;
};

export type ValidationIssue = {
  code: string;
  message: string;
  details?: Record<string, unknown>;
};

export type ValidationResult = {
  ok: boolean;
  issues: ValidationIssue[];
};

export type RagReferenceNormalizationResult = {
  ragObject: AgenticRagOutputObject;
  removedReferenceCount: number;
};
