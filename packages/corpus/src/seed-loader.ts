/** loads only versioned v1 corpus and evaluation records */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import {
  type GoldenEvalPrompt,
  type HumorExpectation,
  HumorExpectations,
  type TrainingExample,
  type ValidationIssue,
  type ValidationResult,
  validateGoldenEvalPrompt,
  validateMetadata,
  validateTrainingExample,
} from "./schema.js"

export interface ParsedJsonlLine {
  readonly line: number
  readonly value: unknown
}

export interface LoadedTrainingCorpus {
  readonly filePath: string
  readonly rows: ReadonlyArray<TrainingExample>
}

export interface LoadedGoldenCorpus {
  readonly filePath: string
  readonly rows: ReadonlyArray<GoldenEvalPrompt>
}

export const defaultSeedCorpusDir = (): string => resolve(import.meta.dirname, "../../../data/v1")

export const defaultHandAuthoredPath = (): string =>
  resolve(defaultSeedCorpusDir(), "corpus", "sft.jsonl")

export const defaultGoldenEvalPath = (): string =>
  resolve(defaultSeedCorpusDir(), "eval", "golden.jsonl")

export const defaultPersonaHoldoutPath = (): string =>
  resolve(defaultSeedCorpusDir(), "eval", "persona-holdout.jsonl")

export const defaultPreferenceDatasetPath = (): string =>
  resolve(defaultSeedCorpusDir(), "corpus", "preferences.jsonl")

export interface PreferencePair {
  readonly id: string
  readonly prompt: string
  readonly chosen: Readonly<Record<string, unknown>>
  readonly rejected: Readonly<Record<string, unknown>>
}

export interface LoadedPreferenceCorpus {
  readonly filePath: string
  readonly rows: ReadonlyArray<PreferencePair>
}

const validatePreferencePair = (
  value: unknown,
  line: number,
): ValidationResult & {
  readonly value?: PreferencePair
} => {
  const issues: ValidationIssue[] = []
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return { filePath: "", rowCount: 0, issues: [{ line, path: "$", message: "must be object" }] }
  }
  const row = value as Record<string, unknown>
  for (const field of ["id", "prompt"] as const) {
    if (typeof row[field] !== "string" || row[field].length === 0) {
      issues.push({ line, path: field, message: `${field} must be a non-empty string` })
    }
  }
  if (
    typeof row.chosen === "object" &&
    typeof row.rejected === "object" &&
    JSON.stringify(row.chosen) === JSON.stringify(row.rejected)
  ) {
    issues.push({ line, path: "rejected", message: "chosen and rejected must differ" })
  }
  for (const field of ["chosen", "rejected"] as const) {
    const response = row[field]
    if (typeof response !== "object" || response === null || Array.isArray(response)) {
      issues.push({ line, path: field, message: `${field} must be a response object` })
      continue
    }
    const responseObject = response as Record<string, unknown>
    const keys = Object.keys(responseObject).sort().join(",")
    if (keys !== "callbackId,emotion,gesture,intensity,spoken") {
      issues.push({ line, path: field, message: `${field} has invalid response fields` })
      continue
    }
    if (typeof responseObject.spoken !== "string" || responseObject.spoken.trim().length === 0) {
      issues.push({ line, path: `${field}.spoken`, message: "spoken must be non-empty" })
    }
    if (
      !(
        responseObject.callbackId === null ||
        (typeof responseObject.callbackId === "string" &&
          /^callback-[a-z0-9-]+$/.test(responseObject.callbackId))
      )
    ) {
      issues.push({ line, path: `${field}.callbackId`, message: "callbackId is invalid" })
    }
    issues.push(...validateMetadata(responseObject, field).map((issue) => ({ ...issue, line })))
  }
  if (issues.length > 0) {
    return { filePath: "", rowCount: 0, issues }
  }
  return {
    filePath: "",
    rowCount: 1,
    issues: [],
    value: {
      id: String(row.id),
      prompt: String(row.prompt),
      chosen: row.chosen as Record<string, unknown>,
      rejected: row.rejected as Record<string, unknown>,
    },
  }
}

export interface EvalPromptRow {
  readonly id: string
  readonly user: string
  readonly scenarioFamily?: string
  readonly qualityFocus?: string
  readonly humorExpectation?: HumorExpectation
}

export interface LoadedEvalPromptCorpus {
  readonly filePath: string
  readonly rows: ReadonlyArray<EvalPromptRow>
}

/** Lightweight loader for persona holdout (id + user; qualityFocus optional). */
export const loadEvalPromptJsonl = (filePath: string): LoadedEvalPromptCorpus => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const rows: EvalPromptRow[] = []

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value !== "object" ||
      parsedLine.value === null ||
      Array.isArray(parsedLine.value) ||
      "__jsonParseError" in parsedLine.value
    ) {
      throw new CorpusValidationError(filePath, [
        { line: parsedLine.line, path: "$", message: "invalid JSON object" },
      ])
    }

    const row = parsedLine.value as Record<string, unknown>
    const issues: ValidationIssue[] = []
    if (typeof row.id !== "string" || row.id.length === 0) {
      issues.push({ line: parsedLine.line, path: "id", message: "id must be a non-empty string" })
    }
    if (typeof row.user !== "string" || row.user.length === 0) {
      issues.push({
        line: parsedLine.line,
        path: "user",
        message: "user must be a non-empty string",
      })
    }
    if (
      row.qualityFocus !== undefined &&
      (typeof row.qualityFocus !== "string" || row.qualityFocus.length === 0)
    ) {
      issues.push({
        line: parsedLine.line,
        path: "qualityFocus",
        message: "qualityFocus must be a non-empty string when present",
      })
    }
    if (
      row.humorExpectation !== undefined &&
      (typeof row.humorExpectation !== "string" ||
        !(HumorExpectations as readonly string[]).includes(row.humorExpectation))
    ) {
      issues.push({
        line: parsedLine.line,
        path: "humorExpectation",
        message: `humorExpectation must be one of: ${HumorExpectations.join(", ")}`,
      })
    }
    if (issues.length > 0) {
      throw new CorpusValidationError(filePath, issues)
    }

    rows.push({
      id: String(row.id),
      user: String(row.user),
      ...(typeof row.scenarioFamily === "string" ? { scenarioFamily: row.scenarioFamily } : {}),
      ...(typeof row.qualityFocus === "string" ? { qualityFocus: row.qualityFocus } : {}),
      ...(typeof row.humorExpectation === "string"
        ? { humorExpectation: row.humorExpectation as HumorExpectation }
        : {}),
    })
  }

  return { filePath, rows }
}

export const loadPreferenceJsonl = (filePath: string): LoadedPreferenceCorpus => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const rows: PreferencePair[] = []
  const seenIds = new Set<string>()
  const seenPrompts = new Set<string>()

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value === "object" &&
      parsedLine.value !== null &&
      "__jsonParseError" in parsedLine.value
    ) {
      throw new CorpusValidationError(filePath, [
        { line: parsedLine.line, path: "$", message: "invalid JSON" },
      ])
    }

    const result = validatePreferencePair(parsedLine.value, parsedLine.line)
    if (result.issues.length > 0) {
      throw new CorpusValidationError(filePath, result.issues)
    }
    if (result.value) {
      if (seenIds.has(result.value.id)) {
        throw new CorpusValidationError(filePath, [
          { line: parsedLine.line, path: "id", message: "duplicate preference id" },
        ])
      }
      const normalizedPrompt = result.value.prompt.trim().toLowerCase().replaceAll(/\s+/g, " ")
      if (seenPrompts.has(normalizedPrompt)) {
        throw new CorpusValidationError(filePath, [
          { line: parsedLine.line, path: "prompt", message: "duplicate preference prompt" },
        ])
      }
      seenIds.add(result.value.id)
      seenPrompts.add(normalizedPrompt)
      rows.push(result.value)
    }
  }

  return { filePath, rows }
}

export const parseJsonl = (content: string): ParsedJsonlLine[] => {
  const lines = content.split(/\r?\n/)
  const parsed: ParsedJsonlLine[] = []

  for (let index = 0; index < lines.length; index += 1) {
    const lineNumber = index + 1
    const line = lines[index]?.trim() ?? ""

    if (line.length === 0) {
      continue
    }

    try {
      parsed.push({ line: lineNumber, value: JSON.parse(line) })
    } catch {
      parsed.push({ line: lineNumber, value: { __jsonParseError: true } })
    }
  }

  return parsed
}

export const loadTrainingJsonl = (filePath: string): LoadedTrainingCorpus => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const rows: TrainingExample[] = []

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value === "object" &&
      parsedLine.value !== null &&
      "__jsonParseError" in parsedLine.value
    ) {
      throw new CorpusValidationError(filePath, [
        { line: parsedLine.line, path: "$", message: "invalid JSON" },
      ])
    }

    const result = validateTrainingExample(parsedLine.value, parsedLine.line)
    if (result.issues.length > 0) {
      throw new CorpusValidationError(filePath, result.issues)
    }

    if (result.value) {
      rows.push(result.value)
    }
  }

  return { filePath, rows }
}

export interface LoadedTrainingCorpusSet {
  readonly handAuthoredPath: string
  readonly rows: ReadonlyArray<TrainingExample>
}

export const loadTrainingCorpus = (options?: {
  readonly handAuthoredPath?: string
}): LoadedTrainingCorpusSet => {
  const handAuthoredPath = options?.handAuthoredPath ?? defaultHandAuthoredPath()
  const handAuthored = loadTrainingJsonl(handAuthoredPath)

  return {
    handAuthoredPath,
    rows: handAuthored.rows,
  }
}

export const loadGoldenJsonl = (filePath: string): LoadedGoldenCorpus => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const rows: GoldenEvalPrompt[] = []

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value === "object" &&
      parsedLine.value !== null &&
      "__jsonParseError" in parsedLine.value
    ) {
      throw new CorpusValidationError(filePath, [
        { line: parsedLine.line, path: "$", message: "invalid JSON" },
      ])
    }

    const result = validateGoldenEvalPrompt(parsedLine.value, parsedLine.line)
    if (result.issues.length > 0) {
      throw new CorpusValidationError(filePath, result.issues)
    }

    if (result.value) {
      rows.push(result.value)
    }
  }

  return { filePath, rows }
}

export class CorpusValidationError extends Error {
  readonly filePath: string
  readonly issues: ReadonlyArray<ValidationIssue>

  constructor(filePath: string, issues: ReadonlyArray<ValidationIssue>) {
    super(formatValidationIssues(filePath, issues))
    this.name = "CorpusValidationError"
    this.filePath = filePath
    this.issues = issues
  }
}

export const formatValidationIssue = (filePath: string, issue: ValidationIssue): string =>
  `${filePath}:${issue.line}: ${issue.path}: ${issue.message}`

export const formatValidationIssues = (
  filePath: string,
  issues: ReadonlyArray<ValidationIssue>,
): string => issues.map((issue) => formatValidationIssue(filePath, issue)).join("\n")

export const validateTrainingJsonlFile = (filePath: string): ValidationResult => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const issues: ValidationIssue[] = []

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value === "object" &&
      parsedLine.value !== null &&
      "__jsonParseError" in parsedLine.value
    ) {
      issues.push({ line: parsedLine.line, path: "$", message: "invalid JSON" })
      continue
    }

    const result = validateTrainingExample(parsedLine.value, parsedLine.line)
    issues.push(...result.issues)
  }

  return {
    filePath,
    rowCount: parsedLines.length,
    issues,
  }
}

export const validateGoldenJsonlFile = (filePath: string): ValidationResult => {
  const content = readFileSync(filePath, "utf8")
  const parsedLines = parseJsonl(content)
  const issues: ValidationIssue[] = []

  for (const parsedLine of parsedLines) {
    if (
      typeof parsedLine.value === "object" &&
      parsedLine.value !== null &&
      "__jsonParseError" in parsedLine.value
    ) {
      issues.push({ line: parsedLine.line, path: "$", message: "invalid JSON" })
      continue
    }

    const result = validateGoldenEvalPrompt(parsedLine.value, parsedLine.line)
    issues.push(...result.issues)
  }

  return {
    filePath,
    rowCount: parsedLines.length,
    issues,
  }
}

export interface CorpusValidationSummary {
  readonly results: ReadonlyArray<ValidationResult>
  readonly ok: boolean
}

export const validateCorpus = (options?: {
  readonly handAuthoredPath?: string
  readonly goldenPath?: string
}): CorpusValidationSummary => {
  const handAuthoredPath = options?.handAuthoredPath ?? defaultHandAuthoredPath()
  const goldenPath = options?.goldenPath ?? defaultGoldenEvalPath()

  const results = [validateTrainingJsonlFile(handAuthoredPath), validateGoldenJsonlFile(goldenPath)]

  return {
    results,
    ok: results.every((result) => result.issues.length === 0),
  }
}
