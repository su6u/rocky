import { readFile, writeFile } from "node:fs/promises"
import { basename } from "node:path"

import { buildEvalRunSnapshot, formatEvalReport, parseEvalResultJson } from "../dist/index.js"

const usage = `usage: pnpm --filter @rocky/eval score-results -- --input <results.json> --output <report.json> [--label <label>]

input may be either a raw results array or a run payload with a results array`

const readArgs = (argv) => {
  const args = new Map()
  const tokens = argv[0] === "--" ? argv.slice(1) : argv
  for (let index = 0; index < tokens.length; index += 2) {
    const key = tokens[index]
    const value = tokens[index + 1]
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(usage)
    }
    args.set(key.slice(2), value)
  }

  const input = args.get("input")
  const output = args.get("output")
  if (!input || !output) {
    throw new Error(usage)
  }

  return {
    input,
    output,
    label: args.get("label") ?? basename(input, ".json"),
  }
}

const main = async () => {
  const args = readArgs(process.argv.slice(2))
  const raw = JSON.parse(await readFile(args.input, "utf8"))
  const rows = parseEvalResultJson(Array.isArray(raw) ? raw : raw.results)
  const snapshot = buildEvalRunSnapshot(args.label, rows)
  const payload = {
    label: snapshot.label,
    report: snapshot.report,
    scoredResults: snapshot.results,
  }

  await writeFile(args.output, `${JSON.stringify(payload, null, 2)}\n`, "utf8")
  await writeFile(
    args.output.replace(/\.json$/u, ".txt"),
    `${formatEvalReport(snapshot.report)}\n`,
    "utf8",
  )
  console.log(formatEvalReport(snapshot.report))
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
