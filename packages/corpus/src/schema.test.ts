/** tests record validation independently of any unpublished dataset */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  validateCorpusMessage,
  validateGoldenEvalPrompt,
  validateMetadata,
  validateTrainingExample,
} from "./schema.js"

describe("validateMetadata", () => {
  it("accepts valid metadata", () => {
    assert.deepEqual(
      validateMetadata({ emotion: "neutral", intensity: 0.5, gesture: "none" }, "metadata"),
      [],
    )
  })

  it("reports invalid emotion path", () => {
    const issues = validateMetadata(
      { emotion: "angry", intensity: 0.5, gesture: "none" },
      "metadata",
    )
    assert.equal(issues.length, 1)
    assert.equal(issues[0]?.path, "metadata.emotion")
  })
})

describe("validateCorpusMessage", () => {
  it("rejects metadata on user messages", () => {
    const result = validateCorpusMessage(
      {
        role: "user",
        content: "Hello",
        metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
      },
      "messages[0]",
    )

    assert.equal(result.issues[0]?.path, "messages[0].metadata")
  })

  it("requires content", () => {
    const result = validateCorpusMessage({ role: "user", content: "" }, "messages[0]")
    assert.equal(result.issues[0]?.path, "messages[0].content")
  })

  it("rejects invalid role", () => {
    const result = validateCorpusMessage({ role: "tool", content: "Hello" }, "messages[0]")
    assert.equal(result.issues[0]?.path, "messages[0].role")
  })
})

describe("validateTrainingExample", () => {
  it("accepts a valid training row", () => {
    const result = validateTrainingExample(
      {
        id: "seed-001",
        source: "seed",
        messages: [
          { role: "user", content: "Pressure drop fast fast fast" },
          {
            role: "assistant",
            content: "Seal bulkhead now now now",
            metadata: { emotion: "alarmed", intensity: 0.9, gesture: "hunker_carapace" },
          },
        ],
      },
      1,
    )

    assert.equal(result.issues.length, 0)
    assert.equal(result.value?.id, "seed-001")
  })

  it("reports invalid metadata on assistant turns", () => {
    const result = validateTrainingExample(
      {
        id: "seed-001",
        source: "seed",
        messages: [
          {
            role: "assistant",
            content: "Bad metadata",
            metadata: { emotion: "angry", intensity: 0.5, gesture: "none" },
          },
        ],
      },
      4,
    )

    assert.ok(result.issues.some((issue) => issue.path === "messages[0].metadata.emotion"))
    assert.equal(result.issues[0]?.line, 4)
  })

  it("requires scenarioFamily for hand-authored rows", () => {
    const result = validateTrainingExample(
      {
        id: "hand-001",
        source: "hand-authored",
        messages: [
          {
            role: "assistant",
            content: "Hello",
            metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
          },
        ],
      },
      1,
    )

    assert.ok(result.issues.some((issue) => issue.path === "scenarioFamily"))
  })

  it("reports invalid scenario family", () => {
    const result = validateTrainingExample(
      {
        id: "seed-001",
        source: "seed",
        scenarioFamily: "spaceship_chatter",
        messages: [
          {
            role: "assistant",
            content: "Hello",
            metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
          },
        ],
      },
      2,
    )

    assert.equal(result.issues[0]?.path, "scenarioFamily")
  })
})

describe("validateGoldenEvalPrompt", () => {
  it("accepts a valid golden row", () => {
    const result = validateGoldenEvalPrompt(
      {
        id: "eval-test",
        scenarioFamily: "repairing_machines",
        user: "Pump failed again",
        qualityFocus: "diagnose first",
      },
      1,
    )

    assert.equal(result.issues.length, 0)
  })

  it("preserves optional pattern fields", () => {
    const result = validateGoldenEvalPrompt(
      {
        id: "eval-patterns",
        scenarioFamily: "grounded_context",
        user: "What is sample status?",
        qualityFocus: "stay in notes",
        groundingNotes: "Sample pending",
        groundingPatterns: ["pending"],
        uncertaintyPatterns: ["maybe"],
        roleplayForbiddenPatterns: ["\\*\\*"],
        bookFactForbiddenPatterns: ["oxygen"],
        expectsStillness: true,
      },
      2,
    )

    assert.equal(result.issues.length, 0)
    assert.deepEqual(result.value?.groundingPatterns, ["pending"])
    assert.equal(result.value?.expectsStillness, true)
  })

  it("rejects invalid regex patterns", () => {
    const result = validateGoldenEvalPrompt(
      {
        id: "eval-bad-pattern",
        scenarioFamily: "grounded_context",
        user: "Test",
        qualityFocus: "test",
        groundingPatterns: ["("],
      },
      3,
    )

    assert.ok(result.issues.some((issue) => issue.path.startsWith("groundingPatterns")))
  })
})
