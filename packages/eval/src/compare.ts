import type { DeterministicCheckId, ScoredEvalOutput } from "./deterministic-checks.js"
import { type EvalResultInput, scoreEvalOutputs } from "./deterministic-checks.js"
import { buildEvalReport, type EvalReport } from "./report.js"

export interface EvalRunSnapshot {
  readonly label: string
  readonly results: ReadonlyArray<ScoredEvalOutput>
  readonly report: EvalReport
}

export interface CheckRegressionRow {
  readonly checkId: DeterministicCheckId
  readonly baselineFailures: number
  readonly candidateFailures: number
  readonly delta: number
}

export interface ScenarioRegressionRow {
  readonly scenarioFamily: string
  readonly baselinePassRate: number
  readonly candidatePassRate: number
  readonly delta: number
}

export interface EvalRegressionComparison {
  readonly baselineLabel: string
  readonly candidateLabel: string
  readonly baselinePassRate: number
  readonly candidatePassRate: number
  readonly passRateDelta: number
  readonly byCheck: ReadonlyArray<CheckRegressionRow>
  readonly byScenarioFamily: ReadonlyArray<ScenarioRegressionRow>
  readonly regressions: ReadonlyArray<string>
  readonly improvements: ReadonlyArray<string>
}

export const buildEvalRunSnapshot = (
  label: string,
  outputs: ReadonlyArray<EvalResultInput>,
): EvalRunSnapshot => {
  const results = scoreEvalOutputs(outputs)
  return { label, results, report: buildEvalReport(results) }
}

export const compareEvalRuns = (
  baseline: EvalRunSnapshot,
  candidate: EvalRunSnapshot,
): EvalRegressionComparison => {
  const baselineChecks = new Map(baseline.report.byCheck.map((row) => [row.checkId, row.failures]))
  const candidateChecks = new Map(
    candidate.report.byCheck.map((row) => [row.checkId, row.failures]),
  )
  const checkIds = new Set([...baselineChecks.keys(), ...candidateChecks.keys()])

  const byCheck = [...checkIds]
    .map((checkId) => {
      const baselineFailures = baselineChecks.get(checkId) ?? 0
      const candidateFailures = candidateChecks.get(checkId) ?? 0
      return {
        checkId,
        baselineFailures,
        candidateFailures,
        delta: candidateFailures - baselineFailures,
      }
    })
    .sort((left, right) => left.checkId.localeCompare(right.checkId))

  const baselineFamilies = new Map(
    baseline.report.byScenarioFamily.map((row) => [row.scenarioFamily, row.passRate]),
  )
  const candidateFamilies = new Map(
    candidate.report.byScenarioFamily.map((row) => [row.scenarioFamily, row.passRate]),
  )
  const families = new Set([...baselineFamilies.keys(), ...candidateFamilies.keys()])

  const byScenarioFamily = [...families]
    .map((scenarioFamily) => {
      const baselinePassRate = baselineFamilies.get(scenarioFamily) ?? 0
      const candidatePassRate = candidateFamilies.get(scenarioFamily) ?? 0
      return {
        scenarioFamily,
        baselinePassRate,
        candidatePassRate,
        delta: candidatePassRate - baselinePassRate,
      }
    })
    .sort((left, right) => left.scenarioFamily.localeCompare(right.scenarioFamily))

  const regressions: string[] = []
  const improvements: string[] = []

  if (candidate.report.passRate < baseline.report.passRate) {
    regressions.push(
      `overall pass rate dropped ${(baseline.report.passRate - candidate.report.passRate).toFixed(3)}`,
    )
  } else if (candidate.report.passRate > baseline.report.passRate) {
    improvements.push(
      `overall pass rate improved ${(candidate.report.passRate - baseline.report.passRate).toFixed(3)}`,
    )
  }

  for (const row of byCheck) {
    if (row.delta > 0) {
      regressions.push(`${row.checkId} failures +${row.delta}`)
    } else if (row.delta < 0) {
      improvements.push(`${row.checkId} failures ${row.delta}`)
    }
  }

  for (const row of byScenarioFamily) {
    if (row.delta < 0) {
      regressions.push(`${row.scenarioFamily} pass rate ${row.delta.toFixed(3)}`)
    } else if (row.delta > 0) {
      improvements.push(`${row.scenarioFamily} pass rate +${row.delta.toFixed(3)}`)
    }
  }

  return {
    baselineLabel: baseline.label,
    candidateLabel: candidate.label,
    baselinePassRate: baseline.report.passRate,
    candidatePassRate: candidate.report.passRate,
    passRateDelta: candidate.report.passRate - baseline.report.passRate,
    byCheck,
    byScenarioFamily,
    regressions,
    improvements,
  }
}

const optionalStringArray = (
  record: Record<string, unknown>,
  field: string,
): ReadonlyArray<string> | undefined => {
  const raw = record[field]
  if (!Array.isArray(raw) || raw.some((entry) => typeof entry !== "string")) {
    return undefined
  }

  return raw as ReadonlyArray<string>
}

export const parseEvalResultJson = (raw: unknown): EvalResultInput[] => {
  if (!Array.isArray(raw)) {
    throw new EvalCompareError("eval result json must be an array")
  }

  return raw.map((entry, index) => {
    if (typeof entry !== "object" || entry === null) {
      throw new EvalCompareError(`result row ${index} must be an object`)
    }

    const record = entry as Record<string, unknown>
    const required = ["id", "promptId", "scenarioFamily", "rawOutput"] as const

    for (const field of required) {
      if (typeof record[field] !== "string" || record[field].length === 0) {
        throw new EvalCompareError(`result row ${index} missing ${field}`)
      }
    }

    const groundingPatterns = optionalStringArray(record, "groundingPatterns")
    const uncertaintyPatterns = optionalStringArray(record, "uncertaintyPatterns")
    const roleplayForbiddenPatterns = optionalStringArray(record, "roleplayForbiddenPatterns")
    const bookFactForbiddenPatterns = optionalStringArray(record, "bookFactForbiddenPatterns")

    const result: EvalResultInput = {
      id: record.id as string,
      promptId: record.promptId as string,
      scenarioFamily: record.scenarioFamily as string,
      rawOutput: record.rawOutput as string,
      ...(typeof record.expectsStillness === "boolean"
        ? { expectsStillness: record.expectsStillness }
        : {}),
      ...(typeof record.maxSpokenLength === "number"
        ? { maxSpokenLength: record.maxSpokenLength }
        : {}),
      ...(groundingPatterns !== undefined ? { groundingPatterns } : {}),
      ...(uncertaintyPatterns !== undefined ? { uncertaintyPatterns } : {}),
      ...(roleplayForbiddenPatterns !== undefined ? { roleplayForbiddenPatterns } : {}),
      ...(bookFactForbiddenPatterns !== undefined ? { bookFactForbiddenPatterns } : {}),
    }

    return result
  })
}

export class EvalCompareError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "EvalCompareError"
  }
}
