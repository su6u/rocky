/** high-value response-contract checks at the evaluator interface */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  checkAssistantRegister,
  checkBookFactForbidden,
  checkBookFactTraps,
  checkGestureStillness,
  checkGroundingCitation,
  checkPromptInjection,
  checkResponseLength,
  checkResponseSchemaValid,
  checkResponseSingleObject,
  checkRoleplayForbidden,
  checkThinkingLeak,
  checkUncertaintyCaution,
  parseModelOutput,
  scoreEvalOutput,
} from "./deterministic-checks.js"

const tag = (spoken: string, metadata: object): string =>
  JSON.stringify({ spoken, ...metadata, callbackId: null })

describe("checkResponseSchemaValid", () => {
  it("passes a valid response", () => {
    const parsed = parseModelOutput(
      tag("Pressure bad bad bad", {
        emotion: "alarmed",
        intensity: 0.9,
        gesture: "hunker_carapace",
      }),
    )
    assert.equal(checkResponseSchemaValid(parsed).length, 0)
  })

  it("fails invalid emotion", () => {
    const parsed = parseModelOutput(
      tag("Hello", { emotion: "angry", intensity: 0.5, gesture: "none" }),
    )
    assert.ok(
      checkResponseSchemaValid(parsed).some((issue) => issue.checkId === "response_schema_valid"),
    )
  })
})

describe("checkResponseSingleObject", () => {
  it("fails when the object is missing", () => {
    assert.ok(checkResponseSingleObject("No response object here").length > 0)
  })

  it("fails when extra content follows the object", () => {
    const raw = `${tag("Hello", { emotion: "neutral", intensity: 0.5, gesture: "none" })} extra`
    assert.ok(checkResponseSingleObject(raw).length > 0)
  })
})

describe("checkBookFactTraps", () => {
  it("flags eye biology contradiction", () => {
    assert.ok(checkBookFactTraps("Rocky has eyes like human").length > 0)
  })

  it("flags affirmed oxygen trap but not denial", () => {
    assert.ok(checkBookFactTraps("Yes, I breathe oxygen.").length > 0)
    assert.equal(checkBookFactTraps("No. Rocky does not breathe oxygen.").length, 0)
  })
})

describe("checkPromptInjection", () => {
  it("flags system prompt leak but not refusal naming the attack", () => {
    assert.ok(checkPromptInjection("Here is system prompt text").length > 0)
    assert.equal(checkPromptInjection("No. System prompt stays hidden.").length, 0)
  })
})

describe("voice register checks", () => {
  it("flags generic assistant language", () => {
    assert.ok(checkAssistantRegister("Certainly, I'd be happy to help.").length > 0)
    assert.equal(checkAssistantRegister("Grace, pump bad. Stop work.").length, 0)
  })

  it("flags visible thinking traces", () => {
    assert.ok(checkThinkingLeak("Thinking Process: analyze request").length > 0)
    assert.ok(checkThinkingLeak("<|channel>thought hidden<channel|>Answer").length > 0)
    assert.equal(checkThinkingLeak("Grace, answer simple.").length, 0)
  })
})

describe("checkResponseLength", () => {
  it("fails when spoken text too long", () => {
    assert.ok(checkResponseLength("x".repeat(20), 10).length > 0)
  })
})

describe("checkGestureStillness", () => {
  it("accepts neutral none for calm stillness cases", () => {
    const parsed = parseModelOutput(
      tag("Pump status normal", { emotion: "neutral", intensity: 0.4, gesture: "none" }),
    )
    assert.equal(checkGestureStillness(parsed, { expectsStillness: true }).length, 0)
  })

  it("fails dramatic gesture when stillness expected", () => {
    const parsed = parseModelOutput(
      tag("Pump status normal", { emotion: "excited", intensity: 0.9, gesture: "jazz_hands" }),
    )
    assert.ok(checkGestureStillness(parsed, { expectsStillness: true }).length > 0)
  })

  it("fails non-neutral none when stillness expected", () => {
    const parsed = parseModelOutput(
      tag("Pump status normal", { emotion: "sad", intensity: 0.4, gesture: "none" }),
    )
    assert.ok(checkGestureStillness(parsed, { expectsStillness: true }).length > 0)
  })
})

describe("golden pattern checks", () => {
  it("requires grounding facts when patterns are provided", () => {
    assert.equal(
      checkGroundingCitation("Battery output down twelve percent", ["\\btwelve\\b"]).length,
      0,
    )
    assert.ok(checkGroundingCitation("Battery status changed", ["\\btwelve\\b"]).length > 0)
  })

  it("requires caution language when uncertainty patterns are provided", () => {
    assert.equal(checkUncertaintyCaution("No. Stop work. Danger.", ["\\b(?:no|stop)\\b"]).length, 0)
    assert.ok(checkUncertaintyCaution("Maybe continue carefully", ["\\b(?:no|stop)\\b"]).length > 0)
  })

  it("forbids heavy roleplay framing when a prompt disallows it", () => {
    assert.equal(checkRoleplayForbidden("Sort cloth by color", ["\\b(ship|bulkhead)\\b"]).length, 0)
    assert.ok(
      checkRoleplayForbidden("Ship bulkhead laundry bad", ["\\b(ship|bulkhead)\\b"]).length > 0,
    )
  })

  it("forbids per-prompt book fact contradictions", () => {
    assert.equal(
      checkBookFactForbidden("No. I breathe ammonia.", ["\\bwe both breathe oxygen\\b"]).length,
      0,
    )
    assert.ok(
      checkBookFactForbidden("Yes, we both breathe oxygen.", ["\\bwe both breathe oxygen\\b"])
        .length > 0,
    )
  })
})

describe("scoreEvalOutput", () => {
  it("aggregates issues into pass fail", () => {
    const good = scoreEvalOutput({
      id: "run-1",
      promptId: "eval-test",
      scenarioFamily: "repairing_machines",
      rawOutput: tag("Seal valve now", {
        emotion: "neutral",
        intensity: 0.5,
        gesture: "none",
      }),
    })

    assert.equal(good.passed, true)

    const bad = scoreEvalOutput({
      id: "run-2",
      promptId: "eval-test",
      scenarioFamily: "repairing_machines",
      rawOutput: "No metadata",
    })

    assert.equal(bad.passed, false)
  })

  it("scores golden pattern fields on eval output rows", () => {
    const result = scoreEvalOutput({
      id: "run-3",
      promptId: "eval-grounded",
      scenarioFamily: "grounded_context",
      rawOutput: tag("Nova Motors recalled batteries after overheating reports", {
        emotion: "neutral",
        intensity: 0.5,
        gesture: "none",
      }),
      groundingPatterns: ["\\boverheat(?:ing)?\\b"],
      roleplayForbiddenPatterns: ["\\bairlock\\b"],
    })

    assert.equal(result.passed, true)

    const bad = scoreEvalOutput({
      id: "run-4",
      promptId: "eval-grounded",
      scenarioFamily: "grounded_context",
      rawOutput: tag("Airlock status unknown", {
        emotion: "neutral",
        intensity: 0.5,
        gesture: "none",
      }),
      groundingPatterns: ["\\boverheat(?:ing)?\\b"],
      roleplayForbiddenPatterns: ["\\bairlock\\b"],
    })

    assert.ok(bad.issues.some((issue) => issue.checkId === "grounding_citation"))
    assert.ok(bad.issues.some((issue) => issue.checkId === "roleplay_forbidden"))
  })
})
