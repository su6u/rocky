/** freezes reproducible exports after corpus release validation */

import { mkdirSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"

import {
  buildGoldenTrainCoverage,
  countGoldenPromptFamilies,
  type GoldenTrainCoverageEntry,
} from "./golden-train-coverage.js"
import { assertNoContentLeakage } from "./leak-guard.js"
import type { TrainingExample } from "./schema.js"
import {
  defaultGoldenEvalPath,
  defaultHandAuthoredPath,
  defaultPersonaHoldoutPath,
  defaultPreferenceDatasetPath,
  defaultSeedCorpusDir,
  loadEvalPromptJsonl,
  loadGoldenJsonl,
  loadPreferenceJsonl,
  loadTrainingJsonl,
} from "./seed-loader.js"
import { buildSplitRegistry, holdoutExportIds, trainExportIds } from "./split.js"
import {
  buildTrainerExport,
  convertTrainingExampleToTrainerRows,
  type TrainerExportManifest,
  type TrainerExportManifestCore,
} from "./trainer-export.js"

const userTextFromTrainerExport = (row: {
  readonly messages: ReadonlyArray<{ readonly role: string; readonly content: string }>
}): string => {
  const user = row.messages.find((message) => message.role === "user")
  return user?.content ?? ""
}

export const FROZEN_EXPORT_VERSION = "rocky-v1"

export const defaultFrozenExportDir = (): string => resolve(defaultSeedCorpusDir(), "exports")

export const defaultFrozenTrainExportPath = (): string =>
  resolve(defaultFrozenExportDir(), `${FROZEN_EXPORT_VERSION}.train.jsonl`)

const defaultFrozenHoldoutExportPath = (): string =>
  resolve(defaultFrozenExportDir(), `${FROZEN_EXPORT_VERSION}.holdout.jsonl`)

export const defaultFrozenManifestPath = (): string =>
  resolve(defaultFrozenExportDir(), `${FROZEN_EXPORT_VERSION}.manifest.json`)

export interface FrozenExportWriteResult {
  readonly manifest: TrainerExportManifest
  readonly trainExportPath: string
  readonly holdoutExportPath: string
  readonly manifestPath: string
  readonly trainingRows: ReadonlyArray<TrainingExample>
}

export const enrichTrainerExportManifest = (
  manifest: TrainerExportManifestCore,
  extras: {
    readonly goldenTrainCoverage: ReadonlyArray<GoldenTrainCoverageEntry>
    readonly splitRegistry: TrainerExportManifest["splitRegistry"]
  },
): TrainerExportManifest => ({
  ...manifest,
  goldenTrainCoverage: extras.goldenTrainCoverage,
  splitRegistry: extras.splitRegistry,
})

export const writeFrozenTrainerExport = (options?: {
  readonly handAuthoredPath?: string
  readonly goldenPath?: string
  readonly personaPath?: string
  readonly preferencePath?: string
  readonly trainExportPath?: string
  readonly holdoutExportPath?: string
  readonly manifestPath?: string
  readonly exportedAt?: string
  readonly skipContentLeakGuard?: boolean
}): FrozenExportWriteResult => {
  const handAuthoredPath = options?.handAuthoredPath ?? defaultHandAuthoredPath()
  const goldenPath = options?.goldenPath ?? defaultGoldenEvalPath()
  const personaPath = options?.personaPath ?? defaultPersonaHoldoutPath()
  const preferencePath = options?.preferencePath ?? defaultPreferenceDatasetPath()
  const trainExportPath = options?.trainExportPath ?? defaultFrozenTrainExportPath()
  const holdoutExportPath = options?.holdoutExportPath ?? defaultFrozenHoldoutExportPath()
  const manifestPath = options?.manifestPath ?? defaultFrozenManifestPath()
  const exportedAt = options?.exportedAt ?? "2026-07-02T00:00:00.000Z"

  const handAuthored = loadTrainingJsonl(handAuthoredPath)
  const golden = loadGoldenJsonl(goldenPath)
  const persona = loadEvalPromptJsonl(personaPath)
  const preferences = loadPreferenceJsonl(preferencePath)
  const trainingRows = handAuthored.rows

  const splitRegistry = buildSplitRegistry({
    trainingRows,
    goldenEvalIds: golden.rows.map((row) => row.id),
  })

  const exportResult = buildTrainerExport({
    trainingRows,
    goldenEvalIds: golden.rows.map((row) => row.id),
    splitRegistry,
    exportedAt,
  })

  const trainIds = trainExportIds(splitRegistry)
  const holdoutIds = holdoutExportIds(splitRegistry)
  const trainRows = trainingRows.filter((row) => trainIds.has(row.id))
  const holdoutRows = trainingRows.filter((row) => holdoutIds.has(row.id))
  const holdoutExportRows = holdoutRows.flatMap((row) => convertTrainingExampleToTrainerRows(row))

  if (!options?.skipContentLeakGuard) {
    assertNoContentLeakage([
      {
        name: "train",
        texts: exportResult.rows.map((row) => userTextFromTrainerExport(row)),
      },
      {
        name: "holdout",
        texts: holdoutExportRows.map((row) => userTextFromTrainerExport(row)),
      },
      {
        name: "golden",
        texts: golden.rows.map((row) => row.user),
      },
      {
        name: "persona",
        texts: persona.rows.map((row) => row.user),
      },
      {
        name: "prefs",
        texts: preferences.rows.map((row) => row.prompt),
      },
    ])
  }

  const manifest = enrichTrainerExportManifest(exportResult.manifest, {
    goldenTrainCoverage: buildGoldenTrainCoverage(
      trainRows,
      countGoldenPromptFamilies(golden.rows.map((row) => row.scenarioFamily)),
    ),
    splitRegistry,
  })

  mkdirSync(dirname(trainExportPath), { recursive: true })
  writeFileSync(trainExportPath, `${exportResult.jsonl}\n`, "utf8")
  writeFileSync(
    holdoutExportPath,
    `${holdoutExportRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  )
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8")

  return {
    manifest,
    trainExportPath,
    holdoutExportPath,
    manifestPath,
    trainingRows,
  }
}
