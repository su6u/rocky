/**
 * Fail if contracts/ are stale vs TypeScript package exports.
 * Run: node scripts/check-contracts.mjs
 */
import { spawnSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

const repoRoot = resolve(import.meta.dirname, "..")
const contractPaths = [
  "contracts/domain.json",
  "contracts/eval-gates.json",
  "contracts/model-spec-meta.json",
  "contracts/protocol.json",
  "contracts/prompts/rocky-system.txt",
]

function snapshot(path) {
  try {
    return readFileSync(resolve(repoRoot, path))
  } catch (error) {
    if (error?.code === "ENOENT") {
      return null
    }
    throw error
  }
}

const before = new Map(contractPaths.map((path) => [path, snapshot(path)]))

const sync = spawnSync("pnpm", ["sync-contracts"], {
  cwd: repoRoot,
  encoding: "utf8",
  stdio: "inherit",
})

if (sync.status !== 0) {
  process.exit(sync.status ?? 1)
}

const changed = contractPaths.filter((path) => {
  const previous = before.get(path)
  const current = snapshot(path)
  if (previous === null || previous === undefined) {
    return current !== null
  }
  return current === null || !previous.equals(current)
})

if (changed.length === 0) {
  console.log("contracts are fresh")
  process.exit(0)
}

console.error("contracts were stale and have been regenerated:")
for (const path of changed) {
  console.error(`- ${path}`)
}
process.exit(1)
