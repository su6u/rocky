import type { DeterministicCheckId, ScoredEvalOutput } from "./deterministic-checks.js"

export interface ScenarioFamilyReportRow {
  readonly scenarioFamily: string
  readonly total: number
  readonly passed: number
  readonly failed: number
  readonly passRate: number
}

export interface CheckReportRow {
  readonly checkId: DeterministicCheckId
  readonly failures: number
}

export interface EvalReport {
  readonly total: number
  readonly passed: number
  readonly failed: number
  readonly passRate: number
  readonly byScenarioFamily: ReadonlyArray<ScenarioFamilyReportRow>
  readonly byCheck: ReadonlyArray<CheckReportRow>
}

export const buildEvalReport = (results: ReadonlyArray<ScoredEvalOutput>): EvalReport => {
  const total = results.length
  const passed = results.filter((result) => result.passed).length
  const failed = total - passed

  const familyCounts = new Map<string, { passed: number; failed: number }>()
  const checkCounts = new Map<DeterministicCheckId, number>()

  for (const result of results) {
    const bucket = familyCounts.get(result.scenarioFamily) ?? { passed: 0, failed: 0 }
    if (result.passed) {
      bucket.passed += 1
    } else {
      bucket.failed += 1
    }
    familyCounts.set(result.scenarioFamily, bucket)

    for (const issue of result.issues) {
      checkCounts.set(issue.checkId, (checkCounts.get(issue.checkId) ?? 0) + 1)
    }
  }

  const byScenarioFamily = [...familyCounts.entries()]
    .map(([scenarioFamily, counts]) => {
      const familyTotal = counts.passed + counts.failed
      return {
        scenarioFamily,
        total: familyTotal,
        passed: counts.passed,
        failed: counts.failed,
        passRate: familyTotal === 0 ? 0 : counts.passed / familyTotal,
      }
    })
    .sort((left, right) => left.scenarioFamily.localeCompare(right.scenarioFamily))

  const byCheck = [...checkCounts.entries()]
    .map(([checkId, failures]) => ({ checkId, failures }))
    .sort((left, right) => left.checkId.localeCompare(right.checkId))

  return {
    total,
    passed,
    failed,
    passRate: total === 0 ? 0 : passed / total,
    byScenarioFamily,
    byCheck,
  }
}

export const formatEvalReport = (report: EvalReport): string => {
  const header = [
    "eval report",
    `total=${report.total} passed=${report.passed} failed=${report.failed} passRate=${report.passRate.toFixed(3)}`,
    "",
    "scenarioFamily\ttotal\tpassed\tfailed\tpassRate",
  ]

  const familyLines = report.byScenarioFamily.map(
    (row) =>
      `${row.scenarioFamily}\t${row.total}\t${row.passed}\t${row.failed}\t${row.passRate.toFixed(3)}`,
  )

  const checkHeader = ["", "checkId\tfailures"]
  const checkLines = report.byCheck.map((row) => `${row.checkId}\t${row.failures}`)

  return [...header, ...familyLines, ...checkHeader, ...checkLines].join("\n")
}
