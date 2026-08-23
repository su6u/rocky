/** tests rights-aware source manifests without repository source files */

import assert from "node:assert/strict"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, it } from "node:test"

import {
  listSourceManifests,
  loadAllSourceManifests,
  validateSourceManifest,
  validateSourceManifestFile,
} from "./source-manifest.js"

const validManifest = {
  schemaVersion: "rocky-source-v1",
  sourceId: "film-callback-library",
  title: "Rights-cleared film callback source",
  canonVersion: "film-v1",
  sourceLocator: "restricted://rocky/film-callback-library",
  licenseId: "rights-grant-001",
  rightsStatus: "restricted",
  redistributionAllowed: false,
  trainingAllowed: true,
  quoteAllowed: true,
  contentSha256: "a".repeat(64),
}

describe("validateSourceManifest", () => {
  it("accepts a restricted source without exposing its contents", () => {
    const result = validateSourceManifest(validManifest)

    assert.equal(result.issues.length, 0)
    assert.equal(result.value?.sourceLocator, "restricted://rocky/film-callback-library")
    assert.equal(result.value?.redistributionAllowed, false)
  })

  it("rejects a forbidden source that claims training rights", () => {
    const result = validateSourceManifest({
      ...validManifest,
      rightsStatus: "forbidden",
      trainingAllowed: true,
    })

    assert.ok(result.issues.some((issue) => issue.path === "rightsStatus"))
  })
})

describe("source manifest directory", () => {
  it("lists, loads, and validates manifest metadata only", () => {
    const manifestsDir = mkdtempSync(join(tmpdir(), "rocky-source-manifests-"))
    const manifestPath = join(manifestsDir, "film.json")

    try {
      writeFileSync(manifestPath, JSON.stringify(validManifest), "utf8")

      assert.deepEqual(listSourceManifests(manifestsDir), [manifestPath])
      assert.equal(loadAllSourceManifests(manifestsDir)[0]?.sourceId, "film-callback-library")
      assert.equal(validateSourceManifestFile(manifestPath).issues.length, 0)
    } finally {
      rmSync(manifestsDir, { recursive: true, force: true })
    }
  })
})
