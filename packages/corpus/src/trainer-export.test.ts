/** tests trainer rendering and split behavior with local fixtures */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { SYSTEM_PROMPT } from "@rocky/prompt"

import type { TrainingExample } from "./schema.js"
import { buildSplitRegistry } from "./split.js"
import {
  buildTrainerExport,
  convertTrainingExampleToTrainerRows,
  countResponseObjects,
  decodeAssistantResponse,
  extractResponseObject,
  formatAssistantResponse,
  promptHash,
} from "./trainer-export.js"

const trainingRows: ReadonlyArray<TrainingExample> = [
  {
    id: "sft-v1-casual-1",
    source: "reviewed-v1",
    scenarioFamily: "emotional_friendship",
    messages: [
      { role: "user", content: "I finished the repair" },
      {
        role: "assistant",
        content: "Good good good. Grace repairs machine; Rocky celebrates Grace.",
        metadata: { emotion: "happy", intensity: 0.7, gesture: "jazz_hands" },
      },
    ],
  },
  {
    id: "sft-v1-engineering-1",
    source: "reviewed-v1",
    scenarioFamily: "engineering_reasoning",
    messages: [
      { role: "user", content: "Pump pressure is falling" },
      {
        role: "assistant",
        content: "Measure inlet and outlet. Then leak has location.",
        metadata: { emotion: "curious", intensity: 0.5, gesture: "none" },
      },
    ],
  },
]

describe("formatAssistantResponse", () => {
  it("creates exactly one response object", () => {
    const content = formatAssistantResponse("Pressure bad bad bad", {
      emotion: "alarmed",
      intensity: 0.9,
      gesture: "hunker_carapace",
    })

    assert.equal(countResponseObjects(content), 1)
    assert.match(extractResponseObject(content) ?? "", /hunker_carapace/)
  })
})

describe("convertTrainingExampleToTrainerRows", () => {
  it("uses canonical system prompt and preserves every dialogue turn", () => {
    const rows = convertTrainingExampleToTrainerRows({
      id: "example-1",
      source: "seed",
      messages: [
        { role: "system", content: "Scene note from book" },
        { role: "user", content: "First user line" },
        {
          role: "assistant",
          content: "Intermediate reply",
          metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
        },
        { role: "user", content: "Second user line" },
        {
          role: "assistant",
          content: "Final reply",
          metadata: { emotion: "curious", intensity: 0.6, gesture: "cock_carapace" },
        },
      ],
    })

    assert.equal(rows.length, 1)
    assert.equal(rows[0]?.messages[0]?.content, SYSTEM_PROMPT)
    assert.match(rows[0]?.messages[1]?.content ?? "", /Scene note from book/)
    assert.equal(rows[0]?.messages[2]?.content, "First user line")
    assert.match(rows[0]?.messages[3]?.content ?? "", /Intermediate reply/)
    assert.equal(rows[0]?.messages[4]?.content, "Second user line")
    assert.match(rows[0]?.messages[5]?.content ?? "", /Final reply/)
    assert.equal(countResponseObjects(rows[0]?.messages[3]?.content ?? ""), 1)
    assert.equal(countResponseObjects(rows[0]?.messages[5]?.content ?? ""), 1)
  })

  it("injects groundingNotes via prepareTurnMessages before the user turn", () => {
    const rows = convertTrainingExampleToTrainerRows({
      id: "grounded-1",
      source: "seed",
      groundingNotes: "Nova Motors recalled 4000 rover batteries.",
      messages: [
        { role: "user", content: "What should I know?" },
        {
          role: "assistant",
          content: "Notes say recall happened.",
          metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
        },
      ],
    })

    assert.equal(rows[0]?.messages[0]?.role, "system")
    assert.match(rows[0]?.messages[1]?.content ?? "", /Grounding notes/)
    assert.match(rows[0]?.messages[1]?.content ?? "", /Nova Motors/)
    assert.equal(rows[0]?.messages[2]?.content, "What should I know?")
    assert.match(rows[0]?.messages[3]?.content ?? "", /Notes say recall/)
  })
})

describe("buildTrainerExport", () => {
  it("exports train split rows with manifest metadata", () => {
    const registry = buildSplitRegistry({
      trainingRows,
      goldenEvalIds: ["eval-v1-held-out"],
      seed: 42,
      holdoutFraction: 0,
    })

    const result = buildTrainerExport({
      trainingRows,
      goldenEvalIds: ["eval-v1-held-out"],
      splitRegistry: registry,
      exportedAt: "2026-01-01T00:00:00.000Z",
    })

    assert.ok(result.rows.length > 0)
    assert.equal(result.manifest.promptHash, promptHash())
    assert.equal(result.manifest.domainVersion.length, 16)
    assert.equal(typeof result.manifest.holdoutFraction, "number")
    assert.deepEqual(result.manifest.sourceIds, ["reviewed-v1"])
    assert.equal(result.manifest.rowCount, result.rows.length)
    const firstAssistant = result.rows[0]?.messages.find((message) => message.role === "assistant")
    assert.ok(firstAssistant)

    const decoded = decodeAssistantResponse(firstAssistant.content)
    assert.ok(decoded)
    assert.ok(decoded.spoken.length > 0)
    assert.doesNotThrow(() => JSON.parse(decoded.responseJson))
  })

  it("rejects incomplete custom split registries", () => {
    const registry = buildSplitRegistry({
      trainingRows: trainingRows.slice(0, 1),
      goldenEvalIds: [],
    })

    assert.throws(() =>
      buildTrainerExport({
        trainingRows,
        goldenEvalIds: [],
        splitRegistry: registry,
        exportedAt: "2026-01-01T00:00:00.000Z",
      }),
    )
  })
})
