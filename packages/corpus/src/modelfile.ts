import { basename } from "node:path"

import { SYSTEM_PROMPT } from "@rocky/prompt"

import type { ModelSpec } from "./model-spec.js"
import { promptHash } from "./trainer-export.js"

export const MODELFILE_GENERATED_MARKER = "GENERATED — do not edit by hand"

export interface GenerateModelfileOptions {
  readonly spec: ModelSpec
  readonly systemPrompt?: string
  readonly ggufPath?: string
}

export class ModelfileError extends Error {
  readonly reason: string

  constructor(reason: string) {
    super(reason)
    this.name = "ModelfileError"
    this.reason = reason
  }
}

export const assertCanonicalSystemPrompt = (systemPrompt: string): void => {
  if (systemPrompt !== SYSTEM_PROMPT) {
    throw new ModelfileError("system prompt must match @rocky/prompt SYSTEM_PROMPT")
  }
}

export const resolveModelfileGgufFrom = (ggufPath: string): string => `./${basename(ggufPath)}`

const resolveModelfileRendererParser = (
  chatTemplate: string,
): { readonly renderer: string; readonly parser: string } => {
  const normalized = chatTemplate.trim().toLowerCase()
  if (normalized === "gemma4" || normalized.includes("gemma4")) {
    return { renderer: "gemma4", parser: "gemma4" }
  }
  if (normalized === "gemma") {
    return { renderer: "gemma", parser: "gemma" }
  }
  return { renderer: normalized, parser: normalized }
}

export const generateModelfile = ({
  spec,
  systemPrompt = SYSTEM_PROMPT,
  ggufPath = spec.artifacts.gguf_path,
}: GenerateModelfileOptions): string => {
  assertCanonicalSystemPrompt(systemPrompt)

  const lines = [
    `# ${MODELFILE_GENERATED_MARKER}`,
    `# spec: ${spec.id}`,
    `# prompt_hash: ${promptHash()}`,
    `from ${resolveModelfileGgufFrom(ggufPath)}`,
    `parameter temperature ${spec.inference.temperature}`,
    `parameter top_p ${spec.inference.top_p}`,
    `parameter num_ctx ${spec.inference.num_ctx}`,
  ]

  for (const stop of spec.inference.stop) {
    lines.push(`parameter stop ${JSON.stringify(stop)}`)
  }

  const { renderer, parser } = resolveModelfileRendererParser(spec.chat_template)
  lines.push(`renderer ${renderer}`)
  lines.push(`parser ${parser}`)
  lines.push(`system """${systemPrompt}"""`)

  return `${lines.join("\n")}\n`
}
