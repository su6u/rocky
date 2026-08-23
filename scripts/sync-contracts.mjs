/** sync shared TypeScript contracts for Python training and tooling */
import { mkdir, writeFile } from "node:fs/promises"
import { resolve } from "node:path"
import { MODEL_SPEC_META } from "../packages/corpus/dist/index.js"
import { Emotions, Gestures } from "../packages/domain/dist/index.js"
import { EVAL_GATE_PHRASE_CONTRACT } from "../packages/eval/dist/index.js"
import { SYSTEM_PROMPT } from "../packages/prompt/dist/index.js"
import { CONTEXT_PREAMBLE } from "../packages/protocol/dist/index.js"

const repoRoot = resolve(import.meta.dirname, "..")
const contractsDir = resolve(repoRoot, "contracts")
const promptsDir = resolve(contractsDir, "prompts")

const GENERATED =
  "GENERATED — run `pnpm sync-contracts` after changing @rocky/domain, @rocky/prompt, @rocky/eval, @rocky/corpus, or @rocky/protocol"

const writeText = async (path, body) => {
  await writeFile(path, `${body.trimEnd()}\n`, "utf8")
}

const writeJson = async (path, value) => {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8")
}

const main = async () => {
  await mkdir(promptsDir, { recursive: true })

  await writeJson(resolve(contractsDir, "domain.json"), {
    _generated: GENERATED,
    emotions: Emotions,
    gestures: Gestures,
  })

  await writeJson(resolve(contractsDir, "eval-gates.json"), {
    _generated: GENERATED,
    ...EVAL_GATE_PHRASE_CONTRACT,
  })

  await writeJson(resolve(contractsDir, "model-spec-meta.json"), {
    _generated: GENERATED,
    ...MODEL_SPEC_META,
  })

  await writeJson(resolve(contractsDir, "protocol.json"), {
    _generated: GENERATED,
    contextPreamble: CONTEXT_PREAMBLE,
  })

  await writeText(resolve(promptsDir, "rocky-system.txt"), SYSTEM_PROMPT)

  console.log("contracts/domain.json")
  console.log("contracts/eval-gates.json")
  console.log("contracts/model-spec-meta.json")
  console.log("contracts/protocol.json")
  console.log("contracts/prompts/rocky-system.txt")
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
