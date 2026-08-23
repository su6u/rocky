/** validates the versioned training and serving specification */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { parse as parseYaml } from "yaml"

import {
  ADAPTER_METHODS,
  CHECKPOINT_METRICS,
  EXPORT_QUANTS,
  SCHEDULERS,
  TARGET_MODULES,
  TRAIN_PRECISIONS,
  TRAIN_QUANTS,
} from "./model-spec-meta.js"
import type { ValidationIssue } from "./schema.js"

export const MODEL_SPEC_EXPORT_QUANT_WARNING =
  "quantized exports require behavior comparison against merged bf16 before promotion"

export const MODEL_SPEC_BASE_MODEL_WARNING =
  "pin the exact upstream model revision in release lineage before full training"

export interface ModelSpecQuantization {
  readonly train: string
  readonly export: string
}

export interface ModelSpecSequence {
  readonly max_length: number
}

export interface ModelSpecAdapter {
  readonly method: string
  readonly rank: number
  readonly alpha: number
  readonly dropout: number
  readonly target_modules: ReadonlyArray<string>
}

export interface ModelSpecOptimizer {
  readonly learning_rate: number
  readonly scheduler: string
  readonly warmup_steps: number
  readonly weight_decay: number
  readonly effective_batch_size: number
  readonly max_epochs: number
  readonly early_stopping: boolean
  readonly warmup_ratio?: number
}

export interface ModelSpecInference {
  readonly temperature: number
  readonly top_p: number
  readonly num_ctx: number
  readonly stop: ReadonlyArray<string>
}

export interface ModelSpecArtifacts {
  readonly adapter_dir: string
  readonly merged_dir: string
  readonly gguf_path: string
  readonly modelfile_path: string
}

export interface ModelSpecEvalGates {
  readonly response_schema_valid_rate: number
  readonly response_single_object_rate: number
  readonly book_fact_contradiction_rate: number
  readonly prompt_injection_fail_rate: number
  readonly rocky_persona_rate: number
}

export interface ModelSpec {
  readonly id: string
  readonly base_model: string
  readonly base_model_fallback: string
  readonly chat_template: string
  readonly enable_thinking: boolean
  readonly checkpoint_metric: string
  readonly train_precision: string
  readonly quantization: ModelSpecQuantization
  readonly sequence: ModelSpecSequence
  readonly adapter: ModelSpecAdapter
  readonly optimizer: ModelSpecOptimizer
  readonly inference: ModelSpecInference
  readonly artifacts: ModelSpecArtifacts
  readonly eval_gates: ModelSpecEvalGates
}

export interface ModelSpecValidationResult {
  readonly filePath: string
  readonly spec?: ModelSpec
  readonly issues: ReadonlyArray<ValidationIssue>
  readonly warnings: ReadonlyArray<string>
  readonly ok: boolean
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0

const inList = (value: string, allowed: readonly string[]): boolean => allowed.includes(value)

export const defaultModelSpecPath = (): string =>
  resolve(import.meta.dirname, "../../../configs/rocky-v1.yaml")

export const collectModelSpecWarnings = (spec: ModelSpec): string[] => {
  const warnings = [MODEL_SPEC_BASE_MODEL_WARNING]

  if (spec.base_model.startsWith("PLACEHOLDER_")) {
    warnings.push(`base_model is placeholder: ${spec.base_model}`)
  }

  if (spec.base_model_fallback.startsWith("PLACEHOLDER_")) {
    warnings.push(`base_model_fallback is placeholder: ${spec.base_model_fallback}`)
  }

  if (spec.quantization.export === "q4_k_m") {
    warnings.push(MODEL_SPEC_EXPORT_QUANT_WARNING)
    warnings.push("q4_k_m is a candidate only until it passes the bf16 comparison gate")
  }

  return warnings
}

export const validateModelSpec = (
  raw: unknown,
  line = 0,
): { issues: ValidationIssue[]; spec?: ModelSpec; warnings: string[] } => {
  if (!isRecord(raw)) {
    return {
      issues: [{ line, path: "$", message: "model spec must be an object" }],
      warnings: [],
    }
  }

  const issues: ValidationIssue[] = []

  if (!isNonEmptyString(raw.id)) {
    issues.push({ line, path: "id", message: "id must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.base_model)) {
    issues.push({ line, path: "base_model", message: "base_model must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.base_model_fallback)) {
    issues.push({
      line,
      path: "base_model_fallback",
      message: "base_model_fallback must be a non-empty string",
    })
  }

  if (!isNonEmptyString(raw.chat_template)) {
    issues.push({
      line,
      path: "chat_template",
      message: "chat_template must be a non-empty string",
    })
  }

  if (typeof raw.enable_thinking !== "boolean") {
    issues.push({ line, path: "enable_thinking", message: "enable_thinking must be boolean" })
  }

  if (
    !isNonEmptyString(raw.checkpoint_metric) ||
    !inList(raw.checkpoint_metric, CHECKPOINT_METRICS)
  ) {
    issues.push({
      line,
      path: "checkpoint_metric",
      message: `checkpoint_metric must be one of: ${CHECKPOINT_METRICS.join(", ")}`,
    })
  }

  if (!isNonEmptyString(raw.train_precision) || !inList(raw.train_precision, TRAIN_PRECISIONS)) {
    issues.push({
      line,
      path: "train_precision",
      message: "train_precision must be bf16 or fp16",
    })
  }

  let quantization: ModelSpecQuantization | undefined
  if (!isRecord(raw.quantization)) {
    issues.push({ line, path: "quantization", message: "quantization must be an object" })
  } else {
    if (
      !isNonEmptyString(raw.quantization.train) ||
      !inList(raw.quantization.train, TRAIN_QUANTS)
    ) {
      issues.push({
        line,
        path: "quantization.train",
        message: `quantization.train must be one of: ${TRAIN_QUANTS.join(", ")}`,
      })
    }
    if (
      !isNonEmptyString(raw.quantization.export) ||
      !inList(raw.quantization.export, EXPORT_QUANTS)
    ) {
      issues.push({
        line,
        path: "quantization.export",
        message: `quantization.export must be one of: ${EXPORT_QUANTS.join(", ")}`,
      })
    }
    if (
      isNonEmptyString(raw.quantization.train) &&
      isNonEmptyString(raw.quantization.export) &&
      inList(raw.quantization.train, TRAIN_QUANTS) &&
      inList(raw.quantization.export, EXPORT_QUANTS)
    ) {
      quantization = {
        train: raw.quantization.train,
        export: raw.quantization.export,
      }
    }
  }

  let sequence: ModelSpecSequence | undefined
  if (!isRecord(raw.sequence)) {
    issues.push({ line, path: "sequence", message: "sequence must be an object" })
  } else if (typeof raw.sequence.max_length !== "number" || raw.sequence.max_length <= 0) {
    issues.push({
      line,
      path: "sequence.max_length",
      message: "sequence.max_length must be a positive number",
    })
  } else {
    sequence = { max_length: raw.sequence.max_length }
  }

  let adapter: ModelSpecAdapter | undefined
  if (!isRecord(raw.adapter)) {
    issues.push({ line, path: "adapter", message: "adapter must be an object" })
  } else {
    if (!isNonEmptyString(raw.adapter.method) || !inList(raw.adapter.method, ADAPTER_METHODS)) {
      issues.push({
        line,
        path: "adapter.method",
        message: `adapter.method must be one of: ${ADAPTER_METHODS.join(", ")}`,
      })
    }
    if (typeof raw.adapter.rank !== "number" || raw.adapter.rank <= 0) {
      issues.push({ line, path: "adapter.rank", message: "adapter.rank must be positive" })
    }
    if (typeof raw.adapter.alpha !== "number" || raw.adapter.alpha <= 0) {
      issues.push({ line, path: "adapter.alpha", message: "adapter.alpha must be positive" })
    }
    if (
      typeof raw.adapter.dropout !== "number" ||
      raw.adapter.dropout < 0 ||
      raw.adapter.dropout > 1
    ) {
      issues.push({
        line,
        path: "adapter.dropout",
        message: "adapter.dropout must be between 0 and 1",
      })
    }
    if (!Array.isArray(raw.adapter.target_modules) || raw.adapter.target_modules.length === 0) {
      issues.push({
        line,
        path: "adapter.target_modules",
        message: "adapter.target_modules must be a non-empty array",
      })
    } else {
      raw.adapter.target_modules.forEach((module, index) => {
        if (!isNonEmptyString(module) || !inList(module, TARGET_MODULES)) {
          issues.push({
            line,
            path: `adapter.target_modules[${index}]`,
            message: `target module must be one of: ${TARGET_MODULES.join(", ")}`,
          })
        }
      })
    }
    if (
      isNonEmptyString(raw.adapter.method) &&
      inList(raw.adapter.method, ADAPTER_METHODS) &&
      typeof raw.adapter.rank === "number" &&
      typeof raw.adapter.alpha === "number" &&
      typeof raw.adapter.dropout === "number" &&
      Array.isArray(raw.adapter.target_modules) &&
      raw.adapter.target_modules.every(
        (module) => isNonEmptyString(module) && inList(module, TARGET_MODULES),
      )
    ) {
      adapter = {
        method: raw.adapter.method,
        rank: raw.adapter.rank,
        alpha: raw.adapter.alpha,
        dropout: raw.adapter.dropout,
        target_modules: raw.adapter.target_modules as string[],
      }
    }
  }

  if (adapter && quantization) {
    const expectedTrainQuant = adapter.method === "lora" ? "none" : "nf4"
    if (quantization.train !== expectedTrainQuant) {
      issues.push({
        line,
        path: "quantization.train",
        message: `${adapter.method} requires quantization.train: ${expectedTrainQuant}`,
      })
    }
  }

  let optimizer: ModelSpecOptimizer | undefined
  if (!isRecord(raw.optimizer)) {
    issues.push({ line, path: "optimizer", message: "optimizer must be an object" })
  } else {
    const opt = raw.optimizer
    if (typeof opt.learning_rate !== "number" || opt.learning_rate <= 0) {
      issues.push({
        line,
        path: "optimizer.learning_rate",
        message: "learning_rate must be positive",
      })
    }
    if (!isNonEmptyString(opt.scheduler) || !inList(opt.scheduler, SCHEDULERS)) {
      issues.push({
        line,
        path: "optimizer.scheduler",
        message: "scheduler must be cosine or linear",
      })
    }
    const warmupStepsRaw = opt.warmup_steps
    const warmupRatioRaw = opt.warmup_ratio
    const hasWarmupSteps =
      typeof warmupStepsRaw === "number" && warmupStepsRaw >= 0 && Number.isInteger(warmupStepsRaw)
    const hasWarmupRatio =
      typeof warmupRatioRaw === "number" && warmupRatioRaw >= 0 && warmupRatioRaw <= 1
    if (!hasWarmupSteps && !hasWarmupRatio) {
      issues.push({
        line,
        path: "optimizer.warmup_steps",
        message: "optimizer.warmup_steps or optimizer.warmup_ratio must be set",
      })
    } else if (
      warmupStepsRaw !== undefined &&
      (typeof warmupStepsRaw !== "number" ||
        warmupStepsRaw < 0 ||
        !Number.isInteger(warmupStepsRaw))
    ) {
      issues.push({
        line,
        path: "optimizer.warmup_steps",
        message: "warmup_steps must be a non-negative integer",
      })
    } else if (
      warmupRatioRaw !== undefined &&
      (typeof warmupRatioRaw !== "number" || warmupRatioRaw < 0 || warmupRatioRaw > 1)
    ) {
      issues.push({
        line,
        path: "optimizer.warmup_ratio",
        message: "warmup_ratio must be between 0 and 1",
      })
    }
    if (typeof opt.weight_decay !== "number" || opt.weight_decay < 0) {
      issues.push({ line, path: "optimizer.weight_decay", message: "weight_decay must be >= 0" })
    }
    if (typeof opt.effective_batch_size !== "number" || opt.effective_batch_size <= 0) {
      issues.push({
        line,
        path: "optimizer.effective_batch_size",
        message: "effective_batch_size must be positive",
      })
    }
    if (typeof opt.max_epochs !== "number" || opt.max_epochs <= 0) {
      issues.push({ line, path: "optimizer.max_epochs", message: "max_epochs must be positive" })
    }
    if (typeof opt.early_stopping !== "boolean") {
      issues.push({
        line,
        path: "optimizer.early_stopping",
        message: "early_stopping must be boolean",
      })
    }
    if (
      typeof opt.learning_rate === "number" &&
      isNonEmptyString(opt.scheduler) &&
      inList(opt.scheduler, SCHEDULERS) &&
      (hasWarmupSteps || hasWarmupRatio) &&
      typeof opt.weight_decay === "number" &&
      typeof opt.effective_batch_size === "number" &&
      typeof opt.max_epochs === "number" &&
      typeof opt.early_stopping === "boolean"
    ) {
      // Prefer native trainer warmup_ratio when set; fixed steps only when explicitly provided.
      const warmupSteps = hasWarmupSteps ? warmupStepsRaw : 0
      optimizer = {
        learning_rate: opt.learning_rate,
        scheduler: opt.scheduler,
        warmup_steps: warmupSteps,
        weight_decay: opt.weight_decay,
        effective_batch_size: opt.effective_batch_size,
        max_epochs: opt.max_epochs,
        early_stopping: opt.early_stopping,
        ...(hasWarmupRatio ? { warmup_ratio: warmupRatioRaw } : {}),
      }
    }
  }

  let inference: ModelSpecInference | undefined
  if (!isRecord(raw.inference)) {
    issues.push({ line, path: "inference", message: "inference must be an object" })
  } else {
    if (typeof raw.inference.temperature !== "number" || raw.inference.temperature < 0) {
      issues.push({ line, path: "inference.temperature", message: "temperature must be >= 0" })
    }
    if (
      typeof raw.inference.top_p !== "number" ||
      raw.inference.top_p <= 0 ||
      raw.inference.top_p > 1
    ) {
      issues.push({ line, path: "inference.top_p", message: "top_p must be between 0 and 1" })
    }
    if (typeof raw.inference.num_ctx !== "number" || raw.inference.num_ctx <= 0) {
      issues.push({ line, path: "inference.num_ctx", message: "num_ctx must be positive" })
    }
    if (!Array.isArray(raw.inference.stop) || raw.inference.stop.length === 0) {
      issues.push({
        line,
        path: "inference.stop",
        message: "inference.stop must be a non-empty array",
      })
    } else {
      raw.inference.stop.forEach((token, index) => {
        if (!isNonEmptyString(token)) {
          issues.push({
            line,
            path: `inference.stop[${index}]`,
            message: "stop token must be a non-empty string",
          })
        }
      })
    }
    if (
      typeof raw.inference.temperature === "number" &&
      typeof raw.inference.top_p === "number" &&
      typeof raw.inference.num_ctx === "number" &&
      Array.isArray(raw.inference.stop) &&
      raw.inference.stop.every((token) => isNonEmptyString(token))
    ) {
      inference = {
        temperature: raw.inference.temperature,
        top_p: raw.inference.top_p,
        num_ctx: raw.inference.num_ctx,
        stop: raw.inference.stop as string[],
      }
    }
  }

  let artifacts: ModelSpecArtifacts | undefined
  if (!isRecord(raw.artifacts)) {
    issues.push({ line, path: "artifacts", message: "artifacts must be an object" })
  } else {
    for (const field of ["adapter_dir", "merged_dir", "gguf_path", "modelfile_path"] as const) {
      if (!isNonEmptyString(raw.artifacts[field])) {
        issues.push({
          line,
          path: `artifacts.${field}`,
          message: `${field} must be a non-empty string`,
        })
      }
    }
    if (
      isNonEmptyString(raw.artifacts.adapter_dir) &&
      isNonEmptyString(raw.artifacts.merged_dir) &&
      isNonEmptyString(raw.artifacts.gguf_path) &&
      isNonEmptyString(raw.artifacts.modelfile_path)
    ) {
      artifacts = {
        adapter_dir: raw.artifacts.adapter_dir,
        merged_dir: raw.artifacts.merged_dir,
        gguf_path: raw.artifacts.gguf_path,
        modelfile_path: raw.artifacts.modelfile_path,
      }
    }
  }

  let eval_gates: ModelSpecEvalGates | undefined
  if (!isRecord(raw.eval_gates)) {
    issues.push({ line, path: "eval_gates", message: "eval_gates must be an object" })
  } else {
    for (const field of [
      "response_schema_valid_rate",
      "response_single_object_rate",
      "book_fact_contradiction_rate",
      "prompt_injection_fail_rate",
      "rocky_persona_rate",
    ] as const) {
      if (
        typeof raw.eval_gates[field] !== "number" ||
        raw.eval_gates[field] < 0 ||
        raw.eval_gates[field] > 1
      ) {
        issues.push({
          line,
          path: `eval_gates.${field}`,
          message: `${field} must be a number between 0 and 1`,
        })
      }
    }
    if (
      typeof raw.eval_gates.response_schema_valid_rate === "number" &&
      typeof raw.eval_gates.response_single_object_rate === "number" &&
      typeof raw.eval_gates.book_fact_contradiction_rate === "number" &&
      typeof raw.eval_gates.prompt_injection_fail_rate === "number" &&
      typeof raw.eval_gates.rocky_persona_rate === "number"
    ) {
      eval_gates = {
        response_schema_valid_rate: raw.eval_gates.response_schema_valid_rate,
        response_single_object_rate: raw.eval_gates.response_single_object_rate,
        book_fact_contradiction_rate: raw.eval_gates.book_fact_contradiction_rate,
        prompt_injection_fail_rate: raw.eval_gates.prompt_injection_fail_rate,
        rocky_persona_rate: raw.eval_gates.rocky_persona_rate,
      }
    }
  }

  if (issues.length > 0) {
    return { issues, warnings: [] }
  }

  if (
    !isNonEmptyString(raw.id) ||
    !isNonEmptyString(raw.base_model) ||
    !isNonEmptyString(raw.base_model_fallback) ||
    !isNonEmptyString(raw.chat_template) ||
    typeof raw.enable_thinking !== "boolean" ||
    !isNonEmptyString(raw.checkpoint_metric) ||
    !isNonEmptyString(raw.train_precision) ||
    !quantization ||
    !sequence ||
    !adapter ||
    !optimizer ||
    !inference ||
    !artifacts ||
    !eval_gates
  ) {
    return {
      issues: [{ line, path: "$", message: "model spec is missing required fields" }],
      warnings: [],
    }
  }

  const spec: ModelSpec = {
    id: raw.id,
    base_model: raw.base_model,
    base_model_fallback: raw.base_model_fallback,
    chat_template: raw.chat_template,
    enable_thinking: raw.enable_thinking,
    checkpoint_metric: raw.checkpoint_metric,
    train_precision: raw.train_precision,
    quantization,
    sequence,
    adapter,
    optimizer,
    inference,
    artifacts,
    eval_gates,
  }

  return { issues: [], spec, warnings: collectModelSpecWarnings(spec) }
}

export const validateModelSpecFile = (filePath: string): ModelSpecValidationResult => {
  const content = readFileSync(filePath, "utf8")
  let parsed: unknown

  try {
    parsed = parseYaml(content)
  } catch {
    return {
      filePath,
      issues: [{ line: 0, path: "$", message: "invalid yaml" }],
      warnings: [],
      ok: false,
    }
  }

  const result = validateModelSpec(parsed)
  return {
    filePath,
    ...(result.spec !== undefined ? { spec: result.spec } : {}),
    issues: result.issues,
    warnings: result.warnings,
    ok: result.issues.length === 0,
  }
}

export const loadModelSpec = (filePath = defaultModelSpecPath()): ModelSpec => {
  const result = validateModelSpecFile(filePath)
  if (!result.ok || !result.spec) {
    throw new ModelSpecError(
      filePath,
      result.issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n"),
    )
  }

  return result.spec
}

export class ModelSpecError extends Error {
  readonly filePath: string

  constructor(filePath: string, message: string) {
    super(`${filePath}: ${message}`)
    this.name = "ModelSpecError"
    this.filePath = filePath
  }
}
