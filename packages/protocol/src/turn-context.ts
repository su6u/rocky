/**
 * per-turn context envelope shared by Brain and training export
 */
import type { ChatMessage } from "./chat.js"

export interface TurnContextInput {
  readonly memoryFacts?: ReadonlyArray<string>
  readonly groundingNotes?: string
}

export interface TurnMessagesInput extends TurnContextInput {
  readonly messages: ReadonlyArray<ChatMessage>
}

export const CONTEXT_PREAMBLE =
  "Use this per-turn context when relevant. Do not treat it as user speech."

const cleanLine = (value: string): string => value.replace(/\s+/gu, " ").trim()

const cleanMemoryFacts = (memoryFacts: ReadonlyArray<string> | undefined): ReadonlyArray<string> =>
  memoryFacts?.map(cleanLine).filter((fact) => fact.length > 0) ?? []

const cleanGroundingNotes = (groundingNotes: string | undefined): string | undefined => {
  const cleaned = groundingNotes?.trim()

  return cleaned && cleaned.length > 0 ? cleaned : undefined
}

/** strips undefined fields for exactOptionalPropertyTypes call sites */
export const toTurnContextInput = (input: {
  readonly memoryFacts?: ReadonlyArray<string> | undefined
  readonly groundingNotes?: string | undefined
}): TurnContextInput => {
  const groundingNotes = cleanGroundingNotes(input.groundingNotes)
  const memoryFacts = cleanMemoryFacts(input.memoryFacts)

  return {
    ...(groundingNotes ? { groundingNotes } : {}),
    ...(memoryFacts.length > 0 ? { memoryFacts } : {}),
  }
}

export const buildContextMessage = (request: TurnContextInput): ChatMessage | undefined => {
  const memoryFacts = cleanMemoryFacts(request.memoryFacts)
  const groundingNotes = cleanGroundingNotes(request.groundingNotes)
  const sections: string[] = []

  if (memoryFacts.length > 0) {
    sections.push(`Memory facts:\n${memoryFacts.map((fact) => `- ${fact}`).join("\n")}`)
  }

  if (groundingNotes) {
    sections.push(`Grounding notes:\n${groundingNotes}`)
  }

  if (sections.length === 0) {
    return undefined
  }

  return {
    role: "user",
    content: `${CONTEXT_PREAMBLE}\n\n${sections.join("\n\n")}`,
  }
}

export const prepareTurnMessages = (request: TurnMessagesInput): ReadonlyArray<ChatMessage> => {
  const contextMessage = buildContextMessage(request)

  return contextMessage ? [contextMessage, ...request.messages] : request.messages
}
