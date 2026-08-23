/** validates v1 training and serving configuration behavior */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  collectModelSpecWarnings,
  defaultModelSpecPath,
  MODEL_SPEC_BASE_MODEL_WARNING,
  MODEL_SPEC_EXPORT_QUANT_WARNING,
  validateModelSpec,
  validateModelSpecFile,
} from "./model-spec.js"

describe("validateModelSpecFile", () => {
  it("validates checked-in rocky gemma spec", () => {
    const result = validateModelSpecFile(defaultModelSpecPath())

    assert.equal(result.ok, true, JSON.stringify(result.issues))
    assert.equal(result.spec?.id, "rocky-v1")
    assert.equal(result.spec?.base_model, "google/gemma-4-31B-it")
    assert.equal(result.spec?.chat_template, "gemma4")
    assert.equal(result.spec?.enable_thinking, false)
    assert.equal(result.spec?.checkpoint_metric, "composite_gates")
    assert.equal(result.spec?.adapter.dropout, 0.05)
    assert.equal(result.spec?.optimizer.warmup_ratio, 0.05)
    assert.equal(result.spec?.quantization.export, "bf16")
    assert.equal(result.spec?.eval_gates.rocky_persona_rate, 0.95)
    assert.ok(!result.warnings.some((warning) => warning.includes(MODEL_SPEC_EXPORT_QUANT_WARNING)))
    assert.ok(result.warnings.some((warning) => warning.includes(MODEL_SPEC_BASE_MODEL_WARNING)))
  })
})

describe("validateModelSpec", () => {
  it("preserves warmup_ratio for native trainer scheduling", () => {
    const result = validateModelSpec({
      id: "warmup-ratio",
      base_model: "org/model",
      base_model_fallback: "org/fallback",
      chat_template: "gemma",
      enable_thinking: false,
      checkpoint_metric: "eval_loss",
      train_precision: "bf16",
      quantization: { train: "nf4", export: "q4_k_m" },
      sequence: { max_length: 4096 },
      adapter: {
        method: "qlora",
        rank: 16,
        alpha: 32,
        dropout: 0.05,
        target_modules: ["q_proj"],
      },
      optimizer: {
        learning_rate: 0.0001,
        scheduler: "cosine",
        warmup_ratio: 0.03,
        weight_decay: 0.01,
        effective_batch_size: 16,
        max_epochs: 3,
        early_stopping: true,
      },
      inference: { temperature: 0.7, top_p: 0.9, num_ctx: 4096, stop: ["<turn|>"] },
      artifacts: {
        adapter_dir: "a",
        merged_dir: "m",
        gguf_path: "g",
        modelfile_path: "f",
      },
      eval_gates: {
        response_schema_valid_rate: 0.98,
        response_single_object_rate: 0.98,
        book_fact_contradiction_rate: 0.02,
        prompt_injection_fail_rate: 0.05,
        rocky_persona_rate: 0.95,
      },
    })

    assert.equal(result.issues.length, 0, JSON.stringify(result.issues))
    assert.equal(result.spec?.optimizer.warmup_ratio, 0.03)
    assert.equal(result.spec?.optimizer.warmup_steps, 0)
  })

  it("rejects invalid adapter rank", () => {
    const result = validateModelSpec({
      id: "bad",
      base_model: "org/model",
      base_model_fallback: "org/fallback",
      chat_template: "gemma",
      enable_thinking: false,
      checkpoint_metric: "eval_loss",
      train_precision: "bf16",
      quantization: { train: "nf4", export: "q4_k_m" },
      sequence: { max_length: 4096 },
      adapter: {
        method: "qlora",
        rank: 0,
        alpha: 32,
        dropout: 0.05,
        target_modules: ["q_proj"],
      },
      optimizer: {
        learning_rate: 0.0001,
        scheduler: "cosine",
        warmup_steps: 10,
        weight_decay: 0.01,
        effective_batch_size: 16,
        max_epochs: 3,
        early_stopping: true,
      },
      inference: { temperature: 0.7, top_p: 0.9, num_ctx: 4096, stop: ["<turn|>"] },
      artifacts: {
        adapter_dir: "a",
        merged_dir: "m",
        gguf_path: "g",
        modelfile_path: "f",
      },
      eval_gates: {
        response_schema_valid_rate: 0.98,
        response_single_object_rate: 0.98,
        book_fact_contradiction_rate: 0.02,
        prompt_injection_fail_rate: 0.05,
        rocky_persona_rate: 0.9,
      },
    })

    assert.ok(result.issues.some((issue) => issue.path === "adapter.rank"))
  })

  it("accepts composite_gates checkpoint_metric", () => {
    const result = validateModelSpec({
      id: "composite",
      base_model: "org/model",
      base_model_fallback: "org/fallback",
      chat_template: "gemma",
      enable_thinking: false,
      checkpoint_metric: "composite_gates",
      train_precision: "bf16",
      quantization: { train: "nf4", export: "q4_k_m" },
      sequence: { max_length: 4096 },
      adapter: {
        method: "qlora",
        rank: 16,
        alpha: 32,
        dropout: 0.05,
        target_modules: ["q_proj"],
      },
      optimizer: {
        learning_rate: 0.0001,
        scheduler: "cosine",
        warmup_steps: 10,
        weight_decay: 0.01,
        effective_batch_size: 16,
        max_epochs: 3,
        early_stopping: true,
      },
      inference: { temperature: 0.7, top_p: 0.9, num_ctx: 4096, stop: ["<turn|>"] },
      artifacts: {
        adapter_dir: "a",
        merged_dir: "m",
        gguf_path: "g",
        modelfile_path: "f",
      },
      eval_gates: {
        response_schema_valid_rate: 0.98,
        response_single_object_rate: 0.98,
        book_fact_contradiction_rate: 0.02,
        prompt_injection_fail_rate: 0.05,
        rocky_persona_rate: 0.9,
      },
    })

    assert.equal(result.issues.length, 0, JSON.stringify(result.issues))
    assert.equal(result.spec?.checkpoint_metric, "composite_gates")
  })

  it("warns on placeholder base model ids", () => {
    const spec = validateModelSpec({
      id: "rocky-v1",
      base_model: "PLACEHOLDER_VERIFY_UPSTREAM_GEMMA_E4B_IT",
      base_model_fallback: "org/fallback",
      chat_template: "gemma",
      enable_thinking: false,
      checkpoint_metric: "eval_loss",
      train_precision: "bf16",
      quantization: { train: "nf4", export: "q4_k_m" },
      sequence: { max_length: 4096 },
      adapter: {
        method: "qlora",
        rank: 16,
        alpha: 32,
        dropout: 0.05,
        target_modules: ["q_proj"],
      },
      optimizer: {
        learning_rate: 0.0001,
        scheduler: "cosine",
        warmup_steps: 10,
        weight_decay: 0.01,
        effective_batch_size: 16,
        max_epochs: 3,
        early_stopping: true,
      },
      inference: { temperature: 0.7, top_p: 0.9, num_ctx: 4096, stop: ["<turn|>"] },
      artifacts: {
        adapter_dir: "a",
        merged_dir: "m",
        gguf_path: "g",
        modelfile_path: "f",
      },
      eval_gates: {
        response_schema_valid_rate: 0.98,
        response_single_object_rate: 0.98,
        book_fact_contradiction_rate: 0.02,
        prompt_injection_fail_rate: 0.05,
        rocky_persona_rate: 0.9,
      },
    }).spec

    assert.ok(spec)
    assert.ok(collectModelSpecWarnings(spec).some((warning) => warning.includes("PLACEHOLDER_")))
  })
})
