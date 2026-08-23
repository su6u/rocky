/** renders reviewed conversations into deterministic trainer input */

import { createHash } from "node:crypto"

import { Emotions, Gestures } from "@rocky/domain"
import { SYSTEM_PROMPT } from "@rocky/prompt"
import { prepareTurnMessages } from "@rocky/protocol"
import type { GoldenTrainCoverageEntry } from "./golden-train-coverage.js"
import type { TrainingExample } from "./schema.js"
import {
  assertGoldenEvalNotInTrainExport,
  assertSplitRegistry,
  assertTrainingRowsCoveredByRegistry,
  buildSplitRegistry,
  type SplitRegistry,
  trainExportIds,
} from "./split.js"

export interface TrainerExportMessage {
  readonly role: "system" | "user" | "assistant"
  readonly content: string
}

export interface TrainerExportRow {
  readonly id: string
  readonly messages: ReadonlyArray<TrainerExportMessage>
}

export interface TrainerExportManifestCore {
  readonly promptHash: string
  readonly domainVersion: string
  readonly rowCount: number
  readonly trainingRowCount: number
  readonly neutralNoneCount: number
  readonly sourceIds: ReadonlyArray<string>
  readonly sourceCounts: ReadonlyArray<{ readonly source: string; readonly count: number }>
  readonly scenarioFamilyCounts: ReadonlyArray<{
    readonly scenarioFamily: string
    readonly count: number
  }>
  readonly splitSeed: number
  readonly holdoutFraction: number
  readonly exportedAt: string
}

export interface TrainerExportManifest extends TrainerExportManifestCore {
  readonly goldenTrainCoverage: ReadonlyArray<GoldenTrainCoverageEntry>
  readonly splitRegistry: SplitRegistry
}

export interface TrainerExportResult {
  readonly rows: ReadonlyArray<TrainerExportRow>
  readonly manifest: TrainerExportManifestCore
  readonly jsonl: string
}

export const domainVersion = (): string =>
  createHash("sha256")
    .update(
      JSON.stringify({
        response: "json-object-v1",
        emotions: Emotions,
        gestures: Gestures,
      }),
      "utf8",
    )
    .digest("hex")
    .slice(0, 16)

const parseResponse = (content: string): { emotion?: string; gesture?: string } | undefined => {
  try {
    return JSON.parse(content) as { emotion?: string; gesture?: string }
  } catch {
    return undefined
  }
}

export const promptHash = (): string =>
  createHash("sha256").update(SYSTEM_PROMPT, "utf8").digest("hex")

export const formatAssistantResponse = (
  content: string,
  metadata: { emotion: string; intensity: number; gesture: string },
): string => JSON.stringify({ spoken: content, ...metadata, callbackId: null })

export const extractResponseObject = (content: string): string | undefined => {
  try {
    const parsed = JSON.parse(content) as Record<string, unknown>
    return typeof parsed.spoken === "string" ? content : undefined
  } catch {
    return undefined
  }
}

export const countResponseObjects = (content: string): number => {
  try {
    const parsed = JSON.parse(content)
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? 1 : 0
  } catch {
    return 0
  }
}

export const convertTrainingExampleToTrainerRows = (
  example: TrainingExample,
): ReadonlyArray<TrainerExportRow> => {
  const conversation = example.messages.filter((message) => message.role !== "system")
  const finalMessage = conversation.at(-1)
  if (!finalMessage || finalMessage.role !== "assistant") {
    return []
  }

  const sourceContext = example.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n")
  const sourceMessages: TrainerExportMessage[] = [
    ...(sourceContext.length > 0
      ? [{ role: "user" as const, content: `Scene context:\n${sourceContext}` }]
      : []),
    ...conversation.map((message): TrainerExportMessage => {
      if (message.role === "assistant") {
        return {
          role: "assistant",
          content: formatAssistantResponse(message.content, message.metadata),
        }
      }
      return { role: "user", content: message.content }
    }),
  ]
  const turnMessages = prepareTurnMessages({
    messages: sourceMessages,
    ...(example.groundingNotes !== undefined ? { groundingNotes: example.groundingNotes } : {}),
    ...(example.memoryFacts !== undefined ? { memoryFacts: example.memoryFacts } : {}),
  })

  const messages: TrainerExportMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    ...turnMessages.map(
      (message): TrainerExportMessage => ({
        role: message.role as "user" | "assistant",
        content: message.content,
      }),
    ),
  ]

  return [
    {
      id: example.id,
      messages,
    },
  ]
}

export const buildTrainerExport = (options: {
  readonly trainingRows: ReadonlyArray<TrainingExample>
  readonly goldenEvalIds: ReadonlyArray<string>
  readonly splitRegistry?: SplitRegistry
  readonly splitSeed?: number
  readonly exportedAt?: string
}): TrainerExportResult => {
  const splitRegistry =
    options.splitRegistry ??
    buildSplitRegistry({
      trainingRows: options.trainingRows,
      goldenEvalIds: options.goldenEvalIds,
      ...(options.splitSeed !== undefined ? { seed: options.splitSeed } : {}),
    })

  assertSplitRegistry(splitRegistry)
  assertGoldenEvalNotInTrainExport(splitRegistry, options.goldenEvalIds)
  assertTrainingRowsCoveredByRegistry(splitRegistry, options.trainingRows)

  const allowedTrainIds = trainExportIds(splitRegistry)
  const exportedTrainingRows = options.trainingRows.filter((row) => allowedTrainIds.has(row.id))
  const exportRows = exportedTrainingRows.flatMap((row) => convertTrainingExampleToTrainerRows(row))

  for (const row of exportedTrainingRows) {
    if (!row.scenarioFamily) {
      throw new TrainerExportError(`training row ${row.id} missing scenarioFamily`)
    }
  }

  let neutralNoneCount = 0

  for (const row of exportRows) {
    const assistant = row.messages.find((message) => message.role === "assistant")
    if (!assistant) {
      continue
    }

    const response = extractResponseObject(assistant.content)
    if (!response) {
      throw new TrainerExportError(`assistant row ${row.id} missing response object`)
    }

    if (countResponseObjects(assistant.content) !== 1) {
      throw new TrainerExportError(
        `assistant row ${row.id} must include exactly one response object`,
      )
    }
    const decoded = decodeAssistantResponse(assistant.content)
    if (decoded) {
      const metadata = parseResponse(decoded.responseJson)
      if (metadata?.emotion === "neutral" && metadata.gesture === "none") {
        neutralNoneCount += 1
      }
    }
  }

  const sourceCountsMap = new Map<string, number>()
  const scenarioFamilyCountsMap = new Map<string, number>()

  for (const row of exportedTrainingRows) {
    sourceCountsMap.set(row.source, (sourceCountsMap.get(row.source) ?? 0) + 1)
    if (row.scenarioFamily) {
      scenarioFamilyCountsMap.set(
        row.scenarioFamily,
        (scenarioFamilyCountsMap.get(row.scenarioFamily) ?? 0) + 1,
      )
    }
  }

  const sourceIds = [...new Set(exportedTrainingRows.map((row) => row.source))].sort()
  const exportedAt = options.exportedAt ?? new Date(0).toISOString()

  const manifest: TrainerExportManifestCore = {
    promptHash: promptHash(),
    domainVersion: domainVersion(),
    rowCount: exportRows.length,
    trainingRowCount: exportedTrainingRows.length,
    neutralNoneCount,
    sourceIds,
    sourceCounts: [...sourceCountsMap.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([source, count]) => ({ source, count })),
    scenarioFamilyCounts: [...scenarioFamilyCountsMap.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([scenarioFamily, count]) => ({ scenarioFamily, count })),
    splitSeed: splitRegistry.seed,
    holdoutFraction: splitRegistry.holdoutFraction,
    exportedAt,
  }

  const jsonl = exportRows.map((row) => JSON.stringify(row)).join("\n")

  return { rows: exportRows, manifest, jsonl }
}

export class TrainerExportError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "TrainerExportError"
  }
}

export const decodeAssistantResponse = (
  content: string,
): { spoken: string; responseJson: string } | undefined => {
  try {
    const parsed = JSON.parse(content) as Record<string, unknown>
    if (typeof parsed.spoken !== "string") return undefined
    return { spoken: parsed.spoken, responseJson: JSON.stringify(parsed) }
  } catch {
    return undefined
  }
}
