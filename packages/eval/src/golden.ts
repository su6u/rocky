import { defaultGoldenEvalPath, type GoldenEvalPrompt, loadGoldenJsonl } from "@rocky/corpus"

export const loadGoldenPrompts = (
  filePath = defaultGoldenEvalPath(),
): ReadonlyArray<GoldenEvalPrompt> => loadGoldenJsonl(filePath).rows
