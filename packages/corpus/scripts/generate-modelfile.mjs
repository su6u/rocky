import { mkdir, writeFile } from "node:fs/promises"
import { dirname, isAbsolute, resolve } from "node:path"

import { defaultModelSpecPath, generateModelfile, loadModelSpec } from "../dist/index.js"

const usage = `usage: pnpm --filter @rocky/corpus generate-modelfile -- [--spec <path>] [--gguf <path>] [--output <path>]`

const repoRoot = resolve(import.meta.dirname, "../../..")

const resolveRepoPath = (path) => (isAbsolute(path) ? path : resolve(repoRoot, path))

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
  return args
}

const main = async () => {
  const args = readArgs(process.argv.slice(2))
  const specPath = resolveRepoPath(args.get("spec") ?? defaultModelSpecPath())
  const spec = loadModelSpec(specPath)
  const ggufPath = args.get("gguf") ?? spec.artifacts.gguf_path
  const outputPath = resolveRepoPath(args.get("output") ?? spec.artifacts.modelfile_path)
  const modelfile = generateModelfile({ spec, ggufPath })

  await mkdir(dirname(outputPath), { recursive: true })
  await writeFile(outputPath, modelfile, "utf8")
  console.log(outputPath)
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
