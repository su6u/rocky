/** validates source manifests without exposing restricted source material */

import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join, resolve } from "node:path"

import type { ValidationIssue, ValidationResult } from "./schema.js"

export interface SourceManifest {
  readonly schemaVersion: "rocky-source-v1"
  readonly sourceId: string
  readonly title: string
  readonly canonVersion: string
  readonly sourceLocator: string
  readonly licenseId: string
  readonly rightsStatus: "verified" | "restricted" | "forbidden"
  readonly redistributionAllowed: boolean
  readonly trainingAllowed: boolean
  readonly quoteAllowed: boolean
  readonly contentSha256: string
  readonly sourceNotes?: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0

export const defaultSourceManifestsDir = (): string =>
  resolve(import.meta.dirname, "../../../data/v1/sources/manifests")

export const validateSourceManifest = (
  raw: unknown,
  line = 0,
): { issues: ValidationIssue[]; value?: SourceManifest } => {
  if (!isRecord(raw)) {
    return { issues: [{ line, path: "$", message: "manifest must be an object" }] }
  }

  const issues: ValidationIssue[] = []

  if (raw.schemaVersion !== "rocky-source-v1") {
    issues.push({
      line,
      path: "schemaVersion",
      message: "schemaVersion must be rocky-source-v1",
    })
  }

  if (!isNonEmptyString(raw.sourceId)) {
    issues.push({ line, path: "sourceId", message: "sourceId must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.title)) {
    issues.push({ line, path: "title", message: "title must be a non-empty string" })
  }

  for (const field of ["canonVersion", "sourceLocator", "licenseId"] as const) {
    if (!isNonEmptyString(raw[field])) {
      issues.push({ line, path: field, message: `${field} must be a non-empty string` })
    }
  }

  if (
    !isNonEmptyString(raw.rightsStatus) ||
    !["verified", "restricted", "forbidden"].includes(raw.rightsStatus)
  ) {
    issues.push({
      line,
      path: "rightsStatus",
      message: "rightsStatus must be verified, restricted, or forbidden",
    })
  }

  for (const field of ["redistributionAllowed", "trainingAllowed", "quoteAllowed"] as const) {
    if (typeof raw[field] !== "boolean") {
      issues.push({ line, path: field, message: `${field} must be boolean` })
    }
  }

  if (!isNonEmptyString(raw.contentSha256) || !/^[a-f0-9]{64}$/.test(raw.contentSha256)) {
    issues.push({
      line,
      path: "contentSha256",
      message: "contentSha256 must be a lowercase SHA-256",
    })
  }

  if (
    raw.rightsStatus === "forbidden" &&
    (raw.trainingAllowed === true || raw.quoteAllowed === true)
  ) {
    issues.push({
      line,
      path: "rightsStatus",
      message: "forbidden sources cannot allow training or quotation",
    })
  }

  if (raw.sourceNotes !== undefined && !isNonEmptyString(raw.sourceNotes)) {
    issues.push({
      line,
      path: "sourceNotes",
      message: "sourceNotes must be a non-empty string",
    })
  }

  if (issues.length > 0) {
    return { issues }
  }

  if (
    raw.schemaVersion !== "rocky-source-v1" ||
    !isNonEmptyString(raw.sourceId) ||
    !isNonEmptyString(raw.title) ||
    !isNonEmptyString(raw.canonVersion) ||
    !isNonEmptyString(raw.sourceLocator) ||
    !isNonEmptyString(raw.licenseId) ||
    !isNonEmptyString(raw.rightsStatus) ||
    !["verified", "restricted", "forbidden"].includes(raw.rightsStatus) ||
    typeof raw.redistributionAllowed !== "boolean" ||
    typeof raw.trainingAllowed !== "boolean" ||
    typeof raw.quoteAllowed !== "boolean" ||
    !isNonEmptyString(raw.contentSha256)
  ) {
    return { issues: [{ line, path: "$", message: "manifest is missing required fields" }] }
  }

  const manifest: SourceManifest = {
    schemaVersion: "rocky-source-v1",
    sourceId: raw.sourceId,
    title: raw.title,
    canonVersion: raw.canonVersion,
    sourceLocator: raw.sourceLocator,
    licenseId: raw.licenseId,
    rightsStatus: raw.rightsStatus as SourceManifest["rightsStatus"],
    redistributionAllowed: raw.redistributionAllowed,
    trainingAllowed: raw.trainingAllowed,
    quoteAllowed: raw.quoteAllowed,
    contentSha256: raw.contentSha256,
    ...(isNonEmptyString(raw.sourceNotes) ? { sourceNotes: raw.sourceNotes } : {}),
  }

  return { issues: [], value: manifest }
}

export const loadSourceManifest = (manifestPath: string): SourceManifest => {
  const content = readFileSync(manifestPath, "utf8")
  let parsed: unknown

  try {
    parsed = JSON.parse(content)
  } catch {
    throw new SourceManifestError(manifestPath, [{ line: 0, path: "$", message: "invalid JSON" }])
  }

  const result = validateSourceManifest(parsed)
  if (result.issues.length > 0) {
    throw new SourceManifestError(manifestPath, result.issues)
  }

  if (!result.value) {
    throw new SourceManifestError(manifestPath, [
      { line: 0, path: "$", message: "manifest is invalid" },
    ])
  }

  return result.value
}

export const listSourceManifests = (
  manifestsDir = defaultSourceManifestsDir(),
): ReadonlyArray<string> =>
  existsSync(manifestsDir)
    ? readdirSync(manifestsDir)
        .filter((name) => name.endsWith(".json"))
        .map((name) => join(manifestsDir, name))
        .sort()
    : []

export const loadAllSourceManifests = (
  manifestsDir = defaultSourceManifestsDir(),
): ReadonlyArray<SourceManifest> => listSourceManifests(manifestsDir).map(loadSourceManifest)

export const validateSourceManifestFile = (manifestPath: string): ValidationResult => {
  const content = readFileSync(manifestPath, "utf8")
  let parsed: unknown

  try {
    parsed = JSON.parse(content)
  } catch {
    return {
      filePath: manifestPath,
      rowCount: 1,
      issues: [{ line: 0, path: "$", message: "invalid JSON" }],
    }
  }

  const result = validateSourceManifest(parsed)
  return {
    filePath: manifestPath,
    rowCount: 1,
    issues: result.issues,
  }
}

export class SourceManifestError extends Error {
  readonly manifestPath: string
  readonly issues: ReadonlyArray<ValidationIssue>

  constructor(manifestPath: string, issues: ReadonlyArray<ValidationIssue>) {
    super(
      issues
        .map((issue) => `${manifestPath}:${issue.line}: ${issue.path}: ${issue.message}`)
        .join("\n"),
    )
    this.name = "SourceManifestError"
    this.manifestPath = manifestPath
    this.issues = issues
  }
}
