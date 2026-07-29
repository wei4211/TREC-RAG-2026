import { LlmConfigError } from "./errors";
import type { LlmProviderName } from "./types";

export type NchcLlmConfig = {
  provider: "nchc_llm";
  model: string;
  baseUrl: string;
  apiKeyEnv: string;
  temperature: number;
  maxTokens: number;
};

export type LlmClientConfig = NchcLlmConfig;

export type RawLlmClientConfig = {
  provider?: unknown;
  model?: unknown;
  base_url?: unknown;
  baseUrl?: unknown;
  api_key_env?: unknown;
  apiKeyEnv?: unknown;
  temperature?: unknown;
  max_tokens?: unknown;
  maxTokens?: unknown;
};

export const DEFAULT_NCHC_BASE_URL = "https://portal.genai.nchc.org.tw/api/v1";
export const DEFAULT_NCHC_MODEL = "gpt-oss-120b";
export const DEFAULT_NCHC_API_KEY_ENV = "NCHC_API_KEY";
export const DEFAULT_LLM_TEMPERATURE = 0;
export const DEFAULT_LLM_MAX_TOKENS = 2048;

export function normalizeLlmClientConfig(raw: RawLlmClientConfig): LlmClientConfig {
  const provider = normalizeProvider(raw.provider ?? "nchc_llm");
  if (provider !== "nchc_llm") throw new LlmConfigError(`Unsupported LLM provider: ${provider}`);
  return {
    provider,
    model: normalizeNonEmptyString(raw.model ?? DEFAULT_NCHC_MODEL, "llm.model"),
    baseUrl: normalizeNonEmptyString(raw.baseUrl ?? raw.base_url ?? DEFAULT_NCHC_BASE_URL, "llm.base_url"),
    apiKeyEnv: normalizeNonEmptyString(raw.apiKeyEnv ?? raw.api_key_env ?? DEFAULT_NCHC_API_KEY_ENV, "llm.api_key_env"),
    temperature: normalizeFiniteNumber(raw.temperature ?? DEFAULT_LLM_TEMPERATURE, "llm.temperature"),
    maxTokens: normalizePositiveInteger(raw.maxTokens ?? raw.max_tokens ?? DEFAULT_LLM_MAX_TOKENS, "llm.max_tokens"),
  };
}

export function safeLlmConfigForArtifacts(config: LlmClientConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    model: config.model,
    base_url: config.baseUrl,
    api_key_env: config.apiKeyEnv,
    temperature: config.temperature,
    max_tokens: config.maxTokens,
  };
}

function normalizeProvider(value: unknown): LlmProviderName {
  if (value !== "nchc_llm") throw new LlmConfigError("llm.provider must be nchc_llm in Phase 1.");
  return value;
}

function normalizeNonEmptyString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new LlmConfigError(`${fieldName} must be a non-empty string.`);
  }
  return value.trim();
}

function normalizeFiniteNumber(value: unknown, fieldName: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new LlmConfigError(`${fieldName} must be a finite number.`);
  }
  return value;
}

function normalizePositiveInteger(value: unknown, fieldName: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new LlmConfigError(`${fieldName} must be a positive integer.`);
  }
  return value;
}
