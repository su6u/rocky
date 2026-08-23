/** verifies serving configuration renders directly from ModelSpec */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { SYSTEM_PROMPT } from "@rocky/prompt"

import { loadModelSpec } from "./model-spec.js"
import {
  assertCanonicalSystemPrompt,
  generateModelfile,
  MODELFILE_GENERATED_MARKER,
  ModelfileError,
  resolveModelfileGgufFrom,
} from "./modelfile.js"
import { promptHash } from "./trainer-export.js"

describe("generateModelfile", () => {
  it("generates modelfile from spec and canonical prompt", () => {
    const spec = loadModelSpec()
    const output = generateModelfile({ spec })

    assert.match(output, new RegExp(`^# ${MODELFILE_GENERATED_MARKER}`))
    assert.match(output, /# prompt_hash: [a-f0-9]{64}/)
    assert.ok(output.includes(`from ${resolveModelfileGgufFrom(spec.artifacts.gguf_path)}`))
    assert.match(output, /parameter temperature 1/)
    assert.match(output, /parameter top_p 0\.95/)
    assert.match(output, /parameter num_ctx 16384/)
    assert.ok(output.includes('parameter stop "<turn|>"'))
    assert.ok(output.includes("renderer gemma4"))
    assert.ok(output.includes("parser gemma4"))
    assert.ok(output.includes(`system """${SYSTEM_PROMPT}"""`))
    assert.ok(output.includes(`# prompt_hash: ${promptHash()}`))
  })

  it("rejects non-canonical system prompt", () => {
    const spec = loadModelSpec()

    assert.throws(
      () => generateModelfile({ spec, systemPrompt: "wrong prompt" }),
      (error: unknown) => error instanceof ModelfileError,
    )
    assert.throws(() => assertCanonicalSystemPrompt("wrong prompt"), ModelfileError)
  })

  it("locks every user turn to Grace", () => {
    assert.match(SYSTEM_PROMPT, /The user is always Grace/)
    assert.doesNotMatch(SYSTEM_PROMPT, /not automatically Grace/)
    assert.match(SYSTEM_PROMPT, /Never call yourself AI, software, model, bot/)
    assert.match(SYSTEM_PROMPT, /Harmless companionship and play stay inside/)
  })

  it("uses gemma renderer for gemma chat template", () => {
    const spec = loadModelSpec()
    const output = generateModelfile({
      spec: { ...spec, chat_template: "gemma" },
    })

    assert.ok(output.includes("renderer gemma"))
    assert.ok(output.includes("parser gemma"))
    assert.ok(!output.includes("renderer gemma4"))
  })
})
