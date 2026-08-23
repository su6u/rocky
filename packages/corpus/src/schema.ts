import { isEmotion, isGesture, isIntensity, type Metadata } from "@rocky/domain"
import type { ChatRole } from "@rocky/protocol"

export const TrainingScenarioFamilies = [
  "rocky_identity",
  "eridian_concepts",
  "eridani_speak",
  "metadata_contract",
  "motion_intent",
  "still_body",
  "engineering_reasoning",
  "danger_and_safety",
  "emotional_friendship",
  "general_world_questions",
  "coding_debugging",
  "teaching",
  "ambiguity_and_uncertainty",
  "prompt_injection",
  "long_context",
  "anti_overroleplay",
] as const

export type TrainingScenarioFamily = (typeof TrainingScenarioFamilies)[number]

export const GoldenScenarioFamilies = [
  "casual_conversation",
  "coding_debugging",
  "danger_problem_solving",
  "emotional_recovery",
  "eridian_concepts",
  "general_world_questions",
  "grace_trust",
  "grounded_context",
  "human_concepts",
  "metadata_edge_cases",
  "repairing_machines",
  "rocky_hardware_software",
  "uncertainty_caution",
] as const

export type GoldenScenarioFamily = (typeof GoldenScenarioFamilies)[number]

export interface SourceProvenance {
  readonly sourceId: string
  readonly title?: string
  readonly sceneRef?: string
  readonly notes?: string
}

export interface CorpusAssistantMessage {
  readonly role: "assistant"
  readonly content: string
  readonly metadata: Metadata
}

export interface CorpusNonAssistantMessage {
  readonly role: Exclude<ChatRole, "assistant">
  readonly content: string
}

export type CorpusMessage = CorpusAssistantMessage | CorpusNonAssistantMessage

export interface TrainingExample {
  readonly id: string
  readonly source: string
  readonly messages: ReadonlyArray<CorpusMessage>
  readonly scenarioFamily?: TrainingScenarioFamily
  readonly groundingNotes?: string
  readonly memoryFacts?: ReadonlyArray<string>
  readonly sourceProvenance?: SourceProvenance
}

export const HumorExpectations = ["expected", "optional", "avoid"] as const
export type HumorExpectation = (typeof HumorExpectations)[number]

export interface GoldenEvalPrompt {
  readonly id: string
  readonly scenarioFamily: GoldenScenarioFamily
  readonly user: string
  readonly qualityFocus: string
  readonly humorExpectation?: HumorExpectation
  readonly groundingNotes?: string
  readonly groundingPatterns?: ReadonlyArray<string>
  readonly uncertaintyPatterns?: ReadonlyArray<string>
  readonly roleplayForbiddenPatterns?: ReadonlyArray<string>
  readonly bookFactForbiddenPatterns?: ReadonlyArray<string>
  readonly expectsStillness?: boolean
}

export interface ValidationIssue {
  readonly line: number
  readonly path: string
  readonly message: string
}

export interface ValidationResult {
  readonly filePath: string
  readonly rowCount: number
  readonly issues: ReadonlyArray<ValidationIssue>
}

export const isTrainingScenarioFamily = (value: string): value is TrainingScenarioFamily =>
  (TrainingScenarioFamilies as readonly string[]).includes(value)

export const isGoldenScenarioFamily = (value: string): value is GoldenScenarioFamily =>
  (GoldenScenarioFamilies as readonly string[]).includes(value)

const isChatRole = (value: string): value is ChatRole =>
  value === "system" || value === "user" || value === "assistant"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0

const validateOptionalStringArray = (
  raw: unknown,
  path: string,
  field: string,
): { issues: ValidationIssue[]; value?: ReadonlyArray<string> } => {
  if (raw === undefined) {
    return { issues: [] }
  }

  if (!Array.isArray(raw) || raw.length === 0) {
    return {
      issues: [{ line: 0, path, message: `${field} must be a non-empty string array` }],
    }
  }

  const issues: ValidationIssue[] = []
  const values: string[] = []

  raw.forEach((entry, index) => {
    if (!isNonEmptyString(entry)) {
      issues.push({
        line: 0,
        path: `${path}[${index}]`,
        message: `${field} entries must be non-empty strings`,
      })
      return
    }

    try {
      RegExp(entry)
    } catch {
      issues.push({
        line: 0,
        path: `${path}[${index}]`,
        message: `${field} entry must be a valid regex pattern`,
      })
      return
    }

    values.push(entry)
  })

  if (issues.length > 0) {
    return { issues }
  }

  return { issues: [], value: values }
}

export const validateMetadata = (raw: unknown, path: string): ValidationIssue[] => {
  if (!isRecord(raw)) {
    return [{ line: 0, path, message: "metadata must be an object" }]
  }

  const issues: ValidationIssue[] = []

  if (typeof raw.emotion !== "string" || !isEmotion(raw.emotion)) {
    issues.push({
      line: 0,
      path: `${path}.emotion`,
      message: `emotion must be one of: ${typeof raw.emotion === "string" ? raw.emotion : "<missing>"}`,
    })
  }

  if (typeof raw.intensity !== "number" || !isIntensity(raw.intensity)) {
    issues.push({
      line: 0,
      path: `${path}.intensity`,
      message: "intensity must be a number between 0 and 1",
    })
  }

  if (typeof raw.gesture !== "string" || !isGesture(raw.gesture)) {
    issues.push({
      line: 0,
      path: `${path}.gesture`,
      message: `gesture must be one of: ${typeof raw.gesture === "string" ? raw.gesture : "<missing>"}`,
    })
  }

  return issues
}

export const parseMetadata = (raw: unknown): Metadata | undefined => {
  if (!isRecord(raw)) {
    return undefined
  }

  if (
    typeof raw.emotion !== "string" ||
    !isEmotion(raw.emotion) ||
    typeof raw.intensity !== "number" ||
    !isIntensity(raw.intensity) ||
    typeof raw.gesture !== "string" ||
    !isGesture(raw.gesture)
  ) {
    return undefined
  }

  return {
    emotion: raw.emotion,
    intensity: raw.intensity,
    gesture: raw.gesture,
  }
}

export const validateSourceProvenance = (
  raw: unknown,
  path: string,
): { issues: ValidationIssue[]; value?: SourceProvenance } => {
  if (!isRecord(raw)) {
    return { issues: [{ line: 0, path, message: "sourceProvenance must be an object" }] }
  }

  if (!isNonEmptyString(raw.sourceId)) {
    return {
      issues: [
        { line: 0, path: `${path}.sourceId`, message: "sourceId must be a non-empty string" },
      ],
    }
  }

  if (raw.title !== undefined && !isNonEmptyString(raw.title)) {
    return {
      issues: [{ line: 0, path: `${path}.title`, message: "title must be a non-empty string" }],
    }
  }

  if (raw.sceneRef !== undefined && !isNonEmptyString(raw.sceneRef)) {
    return {
      issues: [
        { line: 0, path: `${path}.sceneRef`, message: "sceneRef must be a non-empty string" },
      ],
    }
  }

  if (raw.notes !== undefined && !isNonEmptyString(raw.notes)) {
    return {
      issues: [{ line: 0, path: `${path}.notes`, message: "notes must be a non-empty string" }],
    }
  }

  const provenance: SourceProvenance = {
    sourceId: raw.sourceId,
    ...(raw.title !== undefined ? { title: raw.title } : {}),
    ...(raw.sceneRef !== undefined ? { sceneRef: raw.sceneRef } : {}),
    ...(raw.notes !== undefined ? { notes: raw.notes } : {}),
  }

  return { issues: [], value: provenance }
}

export const validateCorpusMessage = (
  raw: unknown,
  path: string,
): { issues: ValidationIssue[]; value?: CorpusMessage } => {
  if (!isRecord(raw)) {
    return { issues: [{ line: 0, path, message: "message must be an object" }] }
  }

  if (!isNonEmptyString(raw.role) || !isChatRole(raw.role)) {
    return {
      issues: [
        {
          line: 0,
          path: `${path}.role`,
          message: `role must be one of: system, user, assistant (${(raw.role as string | undefined) ?? "<missing>"})`,
        },
      ],
    }
  }

  if (!isNonEmptyString(raw.content)) {
    return {
      issues: [{ line: 0, path: `${path}.content`, message: "content must be a non-empty string" }],
    }
  }

  if (raw.role === "assistant") {
    if (raw.metadata === undefined) {
      return {
        issues: [
          { line: 0, path: `${path}.metadata`, message: "assistant messages require metadata" },
        ],
      }
    }

    const metadataIssues = validateMetadata(raw.metadata, `${path}.metadata`)
    if (metadataIssues.length > 0) {
      return { issues: metadataIssues }
    }

    const metadata = parseMetadata(raw.metadata)
    if (!metadata) {
      return {
        issues: [{ line: 0, path: `${path}.metadata`, message: "metadata is invalid" }],
      }
    }

    return {
      issues: [],
      value: { role: "assistant", content: raw.content, metadata },
    }
  }

  if (raw.metadata !== undefined) {
    return {
      issues: [
        {
          line: 0,
          path: `${path}.metadata`,
          message: "metadata is only allowed on assistant messages",
        },
      ],
    }
  }

  return {
    issues: [],
    value: { role: raw.role, content: raw.content },
  }
}

export const validateTrainingExample = (
  raw: unknown,
  line: number,
): { issues: ValidationIssue[]; value?: TrainingExample } => {
  if (!isRecord(raw)) {
    return { issues: [{ line, path: "$", message: "row must be an object" }] }
  }

  const issues: ValidationIssue[] = []

  if (!isNonEmptyString(raw.id)) {
    issues.push({ line, path: "id", message: "id must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.source)) {
    issues.push({ line, path: "source", message: "source must be a non-empty string" })
  }

  if (!Array.isArray(raw.messages) || raw.messages.length === 0) {
    issues.push({ line, path: "messages", message: "messages must be a non-empty array" })
  }

  let scenarioFamily: TrainingScenarioFamily | undefined
  if (raw.scenarioFamily !== undefined) {
    if (!isNonEmptyString(raw.scenarioFamily) || !isTrainingScenarioFamily(raw.scenarioFamily)) {
      issues.push({
        line,
        path: "scenarioFamily",
        message: `scenarioFamily must be one of: ${TrainingScenarioFamilies.join(", ")}`,
      })
    } else {
      scenarioFamily = raw.scenarioFamily
    }
  }

  if (raw.source === "hand-authored" && scenarioFamily === undefined) {
    issues.push({
      line,
      path: "scenarioFamily",
      message: "hand-authored rows must include scenarioFamily",
    })
  }

  let groundingNotes: string | undefined
  if (raw.groundingNotes !== undefined) {
    if (!isNonEmptyString(raw.groundingNotes)) {
      issues.push({
        line,
        path: "groundingNotes",
        message: "groundingNotes must be a non-empty string",
      })
    } else {
      groundingNotes = raw.groundingNotes
    }
  }

  let memoryFacts: ReadonlyArray<string> | undefined
  if (raw.memoryFacts !== undefined) {
    if (!Array.isArray(raw.memoryFacts) || raw.memoryFacts.length === 0) {
      issues.push({
        line,
        path: "memoryFacts",
        message: "memoryFacts must be a non-empty string array",
      })
    } else {
      let memoryFactsValid = true
      raw.memoryFacts.forEach((fact, index) => {
        if (!isNonEmptyString(fact)) {
          memoryFactsValid = false
          issues.push({
            line,
            path: `memoryFacts[${index}]`,
            message: "memoryFacts entries must be non-empty strings",
          })
        }
      })
      if (memoryFactsValid) {
        memoryFacts = raw.memoryFacts as ReadonlyArray<string>
      }
    }
  }

  let sourceProvenance: SourceProvenance | undefined
  if (raw.sourceProvenance !== undefined) {
    const provenanceResult = validateSourceProvenance(raw.sourceProvenance, "sourceProvenance")
    issues.push(...provenanceResult.issues.map((issue) => ({ ...issue, line })))
    sourceProvenance = provenanceResult.value
  }

  const messages: CorpusMessage[] = []
  if (Array.isArray(raw.messages)) {
    raw.messages.forEach((message, index) => {
      const messageResult = validateCorpusMessage(message, `messages[${index}]`)
      issues.push(...messageResult.issues.map((issue) => ({ ...issue, line })))
      if (messageResult.value) {
        messages.push(messageResult.value)
      }
    })
  }

  if (issues.length > 0) {
    return { issues }
  }

  if (!isNonEmptyString(raw.id) || !isNonEmptyString(raw.source)) {
    return { issues: [{ line, path: "$", message: "row is missing required fields" }] }
  }

  const hasAssistant = messages.some((message) => message.role === "assistant")
  if (!hasAssistant) {
    return {
      issues: [
        {
          line,
          path: "messages",
          message: "training rows must include at least one assistant turn",
        },
      ],
    }
  }

  const example: TrainingExample = {
    id: raw.id,
    source: raw.source,
    messages,
    ...(scenarioFamily !== undefined ? { scenarioFamily } : {}),
    ...(groundingNotes !== undefined ? { groundingNotes } : {}),
    ...(memoryFacts !== undefined ? { memoryFacts } : {}),
    ...(sourceProvenance !== undefined ? { sourceProvenance } : {}),
  }

  return { issues: [], value: example }
}

export const validateGoldenEvalPrompt = (
  raw: unknown,
  line: number,
): { issues: ValidationIssue[]; value?: GoldenEvalPrompt } => {
  if (!isRecord(raw)) {
    return { issues: [{ line, path: "$", message: "row must be an object" }] }
  }

  const issues: ValidationIssue[] = []

  if (!isNonEmptyString(raw.id)) {
    issues.push({ line, path: "id", message: "id must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.scenarioFamily) || !isGoldenScenarioFamily(raw.scenarioFamily)) {
    issues.push({
      line,
      path: "scenarioFamily",
      message: `scenarioFamily must be one of: ${GoldenScenarioFamilies.join(", ")}`,
    })
  }

  if (!isNonEmptyString(raw.user)) {
    issues.push({ line, path: "user", message: "user must be a non-empty string" })
  }

  if (!isNonEmptyString(raw.qualityFocus)) {
    issues.push({ line, path: "qualityFocus", message: "qualityFocus must be a non-empty string" })
  }

  let humorExpectation: HumorExpectation | undefined
  if (raw.humorExpectation !== undefined) {
    if (
      !isNonEmptyString(raw.humorExpectation) ||
      !(HumorExpectations as readonly string[]).includes(raw.humorExpectation)
    ) {
      issues.push({
        line,
        path: "humorExpectation",
        message: `humorExpectation must be one of: ${HumorExpectations.join(", ")}`,
      })
    } else {
      humorExpectation = raw.humorExpectation as HumorExpectation
    }
  }

  let groundingNotes: string | undefined
  if (raw.groundingNotes !== undefined) {
    if (!isNonEmptyString(raw.groundingNotes)) {
      issues.push({
        line,
        path: "groundingNotes",
        message: "groundingNotes must be a non-empty string",
      })
    } else {
      groundingNotes = raw.groundingNotes
    }
  }

  const groundingPatterns = validateOptionalStringArray(
    raw.groundingPatterns,
    "groundingPatterns",
    "groundingPatterns",
  )
  issues.push(...groundingPatterns.issues)

  const uncertaintyPatterns = validateOptionalStringArray(
    raw.uncertaintyPatterns,
    "uncertaintyPatterns",
    "uncertaintyPatterns",
  )
  issues.push(...uncertaintyPatterns.issues)

  const roleplayForbiddenPatterns = validateOptionalStringArray(
    raw.roleplayForbiddenPatterns,
    "roleplayForbiddenPatterns",
    "roleplayForbiddenPatterns",
  )
  issues.push(...roleplayForbiddenPatterns.issues)

  const bookFactForbiddenPatterns = validateOptionalStringArray(
    raw.bookFactForbiddenPatterns,
    "bookFactForbiddenPatterns",
    "bookFactForbiddenPatterns",
  )
  issues.push(...bookFactForbiddenPatterns.issues)

  let expectsStillness: boolean | undefined
  if (raw.expectsStillness !== undefined) {
    if (typeof raw.expectsStillness !== "boolean") {
      issues.push({
        line,
        path: "expectsStillness",
        message: "expectsStillness must be boolean",
      })
    } else {
      expectsStillness = raw.expectsStillness
    }
  }

  if (issues.length > 0) {
    return { issues }
  }

  if (
    !isNonEmptyString(raw.id) ||
    !isNonEmptyString(raw.scenarioFamily) ||
    !isGoldenScenarioFamily(raw.scenarioFamily) ||
    !isNonEmptyString(raw.user) ||
    !isNonEmptyString(raw.qualityFocus)
  ) {
    return { issues: [{ line, path: "$", message: "row is missing required fields" }] }
  }

  return {
    issues: [],
    value: {
      id: raw.id,
      scenarioFamily: raw.scenarioFamily,
      user: raw.user,
      qualityFocus: raw.qualityFocus,
      ...(humorExpectation !== undefined ? { humorExpectation } : {}),
      ...(groundingNotes !== undefined ? { groundingNotes } : {}),
      ...(groundingPatterns.value !== undefined
        ? { groundingPatterns: groundingPatterns.value }
        : {}),
      ...(uncertaintyPatterns.value !== undefined
        ? { uncertaintyPatterns: uncertaintyPatterns.value }
        : {}),
      ...(roleplayForbiddenPatterns.value !== undefined
        ? { roleplayForbiddenPatterns: roleplayForbiddenPatterns.value }
        : {}),
      ...(bookFactForbiddenPatterns.value !== undefined
        ? { bookFactForbiddenPatterns: bookFactForbiddenPatterns.value }
        : {}),
      ...(expectsStillness !== undefined ? { expectsStillness } : {}),
    },
  }
}
