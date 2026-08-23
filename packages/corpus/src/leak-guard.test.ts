/** verifies exact normalized leakage rejection across all release splits */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  assertNoContentLeakage,
  ContentLeakError,
  findContentLeakage,
  normalizePromptText,
} from "./leak-guard.js"

describe("leak-guard", () => {
  it("normalizes case and whitespace", () => {
    assert.equal(normalizePromptText("  Hello   World  "), "hello world")
  })

  it("detects exact content overlap across buckets", () => {
    const issues = findContentLeakage([
      { name: "train", texts: ["Do aliens exist?", "Keep this"] },
      { name: "persona", texts: ["do aliens exist?"] },
    ])
    assert.equal(issues.length, 1)
    assert.equal(issues[0]?.leftBucket, "train")
    assert.equal(issues[0]?.rightBucket, "persona")
  })

  it("assertNoContentLeakage throws ContentLeakError", () => {
    assert.throws(
      () =>
        assertNoContentLeakage([
          { name: "train", texts: ["hi"] },
          { name: "golden", texts: ["HI"] },
        ]),
      ContentLeakError,
    )
  })

  it("passes when buckets are disjoint", () => {
    assert.doesNotThrow(() =>
      assertNoContentLeakage([
        { name: "train", texts: ["train only"] },
        { name: "golden", texts: ["golden only"] },
      ]),
    )
  })

  it("rejects train and holdout overlap", () => {
    assert.throws(
      () =>
        assertNoContentLeakage([
          { name: "train", texts: ["same prompt"] },
          { name: "holdout", texts: ["same prompt"] },
        ]),
      ContentLeakError,
    )
  })
})
