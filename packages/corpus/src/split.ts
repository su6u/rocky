import type { TrainingExample } from "./schema.js"

export const SplitNames = ["train", "holdout", "eval"] as const
export type SplitName = (typeof SplitNames)[number]

export interface SplitRegistryEntry {
  readonly id: string
  readonly split: SplitName
}

export interface SplitRegistry {
  readonly seed: number
  readonly holdoutFraction: number
  readonly entries: ReadonlyArray<SplitRegistryEntry>
}

export interface SplitLeakageIssue {
  readonly id: string
  readonly message: string
}

export interface SplitReportRow {
  readonly split: SplitName
  readonly source: string
  readonly scenarioFamily: string
  readonly count: number
}

export const DEFAULT_SPLIT_SEED = 42
export const DEFAULT_HOLDOUT_FRACTION = 0.1

export const hashSplitBucket = (id: string, seed: number): number => {
  let hash = seed ^ id.length

  for (let index = 0; index < id.length; index += 1) {
    hash = Math.imul(hash ^ id.charCodeAt(index), 1_664_525)
    hash = (hash << 13) | (hash >>> 19)
  }

  return (hash >>> 0) / 2 ** 32
}

export const assignTrainingSplit = (
  id: string,
  seed: number,
  holdoutFraction: number,
): Exclude<SplitName, "eval"> => {
  if (holdoutFraction <= 0) {
    return "train"
  }

  if (holdoutFraction >= 1) {
    return "holdout"
  }

  return hashSplitBucket(id, seed) < holdoutFraction ? "holdout" : "train"
}

export const buildSplitRegistry = (options: {
  readonly trainingRows: ReadonlyArray<TrainingExample>
  readonly goldenEvalIds: ReadonlyArray<string>
  readonly seed?: number
  readonly holdoutFraction?: number
}): SplitRegistry => {
  const seed = options.seed ?? DEFAULT_SPLIT_SEED
  const holdoutFraction = options.holdoutFraction ?? DEFAULT_HOLDOUT_FRACTION
  const goldenIds = new Set(options.goldenEvalIds)
  const entries: SplitRegistryEntry[] = []

  for (const row of options.trainingRows) {
    if (goldenIds.has(row.id)) {
      entries.push({ id: row.id, split: "eval" })
      continue
    }

    entries.push({
      id: row.id,
      split: assignTrainingSplit(row.id, seed, holdoutFraction),
    })
  }

  for (const goldenId of options.goldenEvalIds) {
    if (!entries.some((entry) => entry.id === goldenId)) {
      entries.push({ id: goldenId, split: "eval" })
    }
  }

  entries.sort((left, right) => left.id.localeCompare(right.id))

  return { seed, holdoutFraction, entries }
}

export const splitRegistryById = (registry: SplitRegistry): ReadonlyMap<string, SplitName> =>
  new Map(registry.entries.map((entry) => [entry.id, entry.split]))

export const findSplitLeakage = (registry: SplitRegistry): ReadonlyArray<SplitLeakageIssue> => {
  const byId = new Map<string, Set<SplitName>>()

  for (const entry of registry.entries) {
    const splits = byId.get(entry.id) ?? new Set<SplitName>()
    splits.add(entry.split)
    byId.set(entry.id, splits)
  }

  const issues: SplitLeakageIssue[] = []

  for (const [id, splits] of byId) {
    if (splits.size > 1) {
      issues.push({
        id,
        message: `id appears in multiple splits: ${[...splits].join(", ")}`,
      })
    }
  }

  return issues
}

export const assertSplitRegistry = (registry: SplitRegistry): void => {
  const issues = findSplitLeakage(registry)
  if (issues.length > 0) {
    throw new SplitRegistryError(issues.map((issue) => `${issue.id}: ${issue.message}`).join("\n"))
  }
}

export const trainExportIds = (registry: SplitRegistry): ReadonlySet<string> =>
  new Set(registry.entries.filter((entry) => entry.split === "train").map((entry) => entry.id))

export const holdoutExportIds = (registry: SplitRegistry): ReadonlySet<string> =>
  new Set(registry.entries.filter((entry) => entry.split === "holdout").map((entry) => entry.id))

export const assertGoldenEvalNotInTrainExport = (
  registry: SplitRegistry,
  goldenEvalIds: ReadonlyArray<string>,
): void => {
  const byId = splitRegistryById(registry)
  const leaked = goldenEvalIds.filter((id) => {
    const split = byId.get(id)
    return split !== undefined && split !== "eval"
  })

  if (leaked.length > 0) {
    throw new SplitRegistryError(`golden eval ids must use eval split only: ${leaked.join(", ")}`)
  }

  const trainIds = trainExportIds(registry)
  const trainLeaked = goldenEvalIds.filter((id) => trainIds.has(id))

  if (trainLeaked.length > 0) {
    throw new SplitRegistryError(`golden eval ids in train export: ${trainLeaked.join(", ")}`)
  }
}

export const assertTrainingRowsCoveredByRegistry = (
  registry: SplitRegistry,
  trainingRows: ReadonlyArray<TrainingExample>,
): void => {
  const byId = splitRegistryById(registry)
  const missing = trainingRows.filter((row) => !byId.has(row.id)).map((row) => row.id)

  if (missing.length > 0) {
    throw new SplitRegistryError(`training rows missing from split registry: ${missing.join(", ")}`)
  }
}

export const buildSplitReport = (
  registry: SplitRegistry,
  trainingRows: ReadonlyArray<TrainingExample>,
): ReadonlyArray<SplitReportRow> => {
  const splits = splitRegistryById(registry)
  const counts = new Map<string, SplitReportRow>()

  for (const row of trainingRows) {
    const split = splits.get(row.id) ?? "train"
    const source = row.source
    const scenarioFamily = row.scenarioFamily ?? "(unset)"
    const key = `${split}\0${source}\0${scenarioFamily}`
    const existing = counts.get(key)

    if (existing) {
      counts.set(key, { ...existing, count: existing.count + 1 })
    } else {
      counts.set(key, { split, source, scenarioFamily, count: 1 })
    }
  }

  return [...counts.values()].sort((left, right) => {
    const bySplit = left.split.localeCompare(right.split)
    if (bySplit !== 0) {
      return bySplit
    }

    const bySource = left.source.localeCompare(right.source)
    if (bySource !== 0) {
      return bySource
    }

    return left.scenarioFamily.localeCompare(right.scenarioFamily)
  })
}

export const formatSplitReport = (rows: ReadonlyArray<SplitReportRow>): string => {
  const header = "split\tsource\tscenarioFamily\tcount"
  const lines = rows.map(
    (row) => `${row.split}\t${row.source}\t${row.scenarioFamily}\t${row.count}`,
  )

  return [header, ...lines].join("\n")
}

export class SplitRegistryError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "SplitRegistryError"
  }
}
