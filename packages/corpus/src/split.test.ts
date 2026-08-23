/** tests deterministic split properties with reviewed-style fixtures */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import type { TrainingExample } from "./schema.js"
import {
  assertGoldenEvalNotInTrainExport,
  assertSplitRegistry,
  assertTrainingRowsCoveredByRegistry,
  assignTrainingSplit,
  buildSplitRegistry,
  buildSplitReport,
  DEFAULT_SPLIT_SEED,
  findSplitLeakage,
  formatSplitReport,
  hashSplitBucket,
} from "./split.js"

const trainingRows: ReadonlyArray<TrainingExample> = [
  {
    id: "sft-v1-casual-1",
    source: "reviewed-v1",
    scenarioFamily: "emotional_friendship",
    messages: [
      { role: "user", content: "Repair is done" },
      {
        role: "assistant",
        content: "Good good good.",
        metadata: { emotion: "happy", intensity: 0.7, gesture: "jazz_hands" },
      },
    ],
  },
  {
    id: "sft-v1-engineering-1",
    source: "reviewed-v1",
    scenarioFamily: "engineering_reasoning",
    messages: [
      { role: "user", content: "Pump pressure falls" },
      {
        role: "assistant",
        content: "Measure both sides first.",
        metadata: { emotion: "curious", intensity: 0.5, gesture: "none" },
      },
    ],
  },
]

describe("assignTrainingSplit", () => {
  it("is deterministic for the same id and seed", () => {
    const first = assignTrainingSplit("seed-local-source-0001", DEFAULT_SPLIT_SEED, 0.1)
    const second = assignTrainingSplit("seed-local-source-0001", DEFAULT_SPLIT_SEED, 0.1)

    assert.equal(first, second)
  })

  it("changes when seed changes", () => {
    const buckets = new Set<string>()

    for (let seed = 0; seed < 20; seed += 1) {
      buckets.add(assignTrainingSplit("seed-local-source-0042", seed, 0.5))
    }

    assert.ok(buckets.size > 1)
  })
})

describe("buildSplitRegistry", () => {
  it("keeps golden eval ids out of train export", () => {
    const goldenIds = ["eval-v1-casual-1"]
    const registry = buildSplitRegistry({
      trainingRows,
      goldenEvalIds: goldenIds,
    })

    assertSplitRegistry(registry)
    assertGoldenEvalNotInTrainExport(registry, goldenIds)

    const evalIds = registry.entries
      .filter((entry) => entry.split === "eval")
      .map((entry) => entry.id)
    assert.equal(evalIds.length, goldenIds.length)
  })

  it("detects duplicate split assignments", () => {
    const registry = buildSplitRegistry({
      trainingRows: [
        {
          id: "row-a",
          source: "seed",
          messages: [
            {
              role: "assistant",
              content: "Hello",
              metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
            },
          ],
        },
      ],
      goldenEvalIds: [],
    })

    const broken = {
      ...registry,
      entries: [...registry.entries, { id: "row-a", split: "holdout" as const }],
    }

    const issues = findSplitLeakage(broken)
    assert.equal(issues.length, 1)
    assert.equal(issues[0]?.id, "row-a")
  })

  it("rejects golden eval ids assigned to holdout", () => {
    const registry = buildSplitRegistry({
      trainingRows: [],
      goldenEvalIds: ["eval-test"],
    })

    const broken = {
      ...registry,
      entries: registry.entries.map((entry) =>
        entry.id === "eval-test" ? { ...entry, split: "holdout" as const } : entry,
      ),
    }

    assert.throws(() => assertGoldenEvalNotInTrainExport(broken, ["eval-test"]))
  })

  it("requires every training row id in the registry", () => {
    const registry = buildSplitRegistry({
      trainingRows: [
        {
          id: "row-a",
          source: "seed",
          messages: [
            {
              role: "assistant",
              content: "Hello",
              metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
            },
          ],
        },
      ],
      goldenEvalIds: [],
    })

    assert.throws(() =>
      assertTrainingRowsCoveredByRegistry(registry, [
        {
          id: "row-a",
          source: "seed",
          messages: [
            {
              role: "assistant",
              content: "Hello",
              metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
            },
          ],
        },
        {
          id: "row-missing",
          source: "seed",
          messages: [
            {
              role: "assistant",
              content: "Missing",
              metadata: { emotion: "neutral", intensity: 0.5, gesture: "none" },
            },
          ],
        },
      ]),
    )
  })

  it("prints row counts by source and split", () => {
    const registry = buildSplitRegistry({
      trainingRows,
      goldenEvalIds: [],
      seed: 7,
      holdoutFraction: 0.25,
    })

    const report = formatSplitReport(buildSplitReport(registry, trainingRows))
    assert.match(report, /^split\tsource\tscenarioFamily\tcount/m)
    assert.match(report, /train/)
  })
})

describe("hashSplitBucket", () => {
  it("returns a stable fraction", () => {
    const first = hashSplitBucket("seed-local-source-0100", 42)
    const second = hashSplitBucket("seed-local-source-0100", 42)

    assert.equal(first, second)
    assert.ok(first >= 0 && first < 1)
  })
})
