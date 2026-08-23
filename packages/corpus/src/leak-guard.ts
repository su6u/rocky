/** rejects normalized prompt duplication across every release split */

export const normalizePromptText = (text: string): string =>
  text.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim()

export interface PromptBucket {
  readonly name: string
  readonly texts: ReadonlyArray<string>
}

export interface ContentLeakIssue {
  readonly leftBucket: string
  readonly rightBucket: string
  readonly normalizedText: string
  readonly sampleText: string
}

export const findContentLeakage = (
  buckets: ReadonlyArray<PromptBucket>,
): ReadonlyArray<ContentLeakIssue> => {
  const issues: ContentLeakIssue[] = []

  for (let i = 0; i < buckets.length; i += 1) {
    for (let j = i + 1; j < buckets.length; j += 1) {
      const left = buckets[i]
      const right = buckets[j]
      if (left === undefined || right === undefined) {
        continue
      }
      const rightByNorm = new Map<string, string>()

      for (const text of right.texts) {
        const normalized = normalizePromptText(text)
        if (normalized.length > 0 && !rightByNorm.has(normalized)) {
          rightByNorm.set(normalized, text)
        }
      }

      const seen = new Set<string>()
      for (const text of left.texts) {
        const normalized = normalizePromptText(text)
        if (normalized.length === 0 || seen.has(normalized)) {
          continue
        }
        seen.add(normalized)
        const sample = rightByNorm.get(normalized)
        if (sample !== undefined) {
          issues.push({
            leftBucket: left.name,
            rightBucket: right.name,
            normalizedText: normalized,
            sampleText: sample,
          })
        }
      }
    }
  }

  return issues.sort((a, b) => {
    const byPair = `${a.leftBucket}/${a.rightBucket}`.localeCompare(
      `${b.leftBucket}/${b.rightBucket}`,
    )
    if (byPair !== 0) {
      return byPair
    }
    return a.normalizedText.localeCompare(b.normalizedText)
  })
}

export class ContentLeakError extends Error {
  readonly issues: ReadonlyArray<ContentLeakIssue>

  constructor(issues: ReadonlyArray<ContentLeakIssue>) {
    const preview = issues
      .slice(0, 8)
      .map(
        (issue) => `${issue.leftBucket}∩${issue.rightBucket}: ${JSON.stringify(issue.sampleText)}`,
      )
      .join("\n")
    const more = issues.length > 8 ? `\n…and ${issues.length - 8} more` : ""
    super(`content leak across prompt sets (${issues.length}):\n${preview}${more}`)
    this.name = "ContentLeakError"
    this.issues = issues
  }
}

export const assertNoContentLeakage = (buckets: ReadonlyArray<PromptBucket>): void => {
  const issues = findContentLeakage(buckets)
  if (issues.length > 0) {
    throw new ContentLeakError(issues)
  }
}
