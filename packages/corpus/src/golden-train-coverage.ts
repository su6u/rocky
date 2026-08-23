import type { TrainingExample, TrainingScenarioFamily } from "./schema.js"

export const GOLDEN_TRAIN_MIN_ROWS = 15

export const GOLDEN_TRAIN_FAMILY_MAP = {
  repairing_machines: ["engineering_reasoning", "danger_and_safety"],
  eridian_concepts: ["rocky_identity", "eridani_speak", "eridian_concepts"],
  human_concepts: ["emotional_friendship", "teaching"],
  grace_trust: ["emotional_friendship", "rocky_identity"],
  danger_problem_solving: ["danger_and_safety", "engineering_reasoning"],
  coding_debugging: ["coding_debugging", "metadata_contract"],
  rocky_hardware_software: [
    "motion_intent",
    "metadata_contract",
    "anti_overroleplay",
    "rocky_identity",
  ],
  emotional_recovery: ["emotional_friendship", "still_body"],
  casual_conversation: ["general_world_questions", "eridani_speak"],
  metadata_edge_cases: ["metadata_contract", "prompt_injection", "still_body", "motion_intent"],
  general_world_questions: ["general_world_questions", "anti_overroleplay"],
  grounded_context: ["groundingNotes"],
  uncertainty_caution: ["ambiguity_and_uncertainty", "prompt_injection"],
} as const satisfies Record<string, ReadonlyArray<string>>

export type GoldenEvalFamily = keyof typeof GOLDEN_TRAIN_FAMILY_MAP

export interface GoldenTrainCoverageEntry {
  readonly goldenFamily: GoldenEvalFamily
  readonly goldenPromptCount: number
  readonly trainFamilies: ReadonlyArray<string>
  readonly trainRowCount: number
  readonly meetsMinimum: boolean
}

const countGroundedTrainRows = (rows: ReadonlyArray<TrainingExample>): number =>
  rows.filter((row) => row.groundingNotes !== undefined && row.groundingNotes.length > 0).length

const countByFamilies = (
  rows: ReadonlyArray<TrainingExample>,
  families: ReadonlyArray<string>,
): number => {
  const familySet = new Set(families)
  return rows.filter((row) => row.scenarioFamily && familySet.has(row.scenarioFamily)).length
}

export const buildGoldenTrainCoverage = (
  trainRows: ReadonlyArray<TrainingExample>,
  goldenPromptCounts: Readonly<Record<string, number>>,
): ReadonlyArray<GoldenTrainCoverageEntry> =>
  (Object.keys(GOLDEN_TRAIN_FAMILY_MAP) as GoldenEvalFamily[])
    .sort((left, right) => left.localeCompare(right))
    .map((goldenFamily) => {
      const trainFamilies = GOLDEN_TRAIN_FAMILY_MAP[goldenFamily]
      const trainRowCount =
        goldenFamily === "grounded_context"
          ? countGroundedTrainRows(trainRows)
          : countByFamilies(trainRows, trainFamilies as ReadonlyArray<TrainingScenarioFamily>)

      return {
        goldenFamily,
        goldenPromptCount: goldenPromptCounts[goldenFamily] ?? 0,
        trainFamilies,
        trainRowCount,
        meetsMinimum: trainRowCount >= GOLDEN_TRAIN_MIN_ROWS,
      }
    })

export const countGoldenPromptFamilies = (
  goldenFamilies: ReadonlyArray<string>,
): Record<string, number> => {
  const counts: Record<string, number> = {}

  for (const family of goldenFamilies) {
    counts[family] = (counts[family] ?? 0) + 1
  }

  return counts
}
