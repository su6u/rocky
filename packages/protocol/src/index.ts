/**
 * message and turn stream contracts shared across Rocky processes
 * transport-specific code wraps these shapes without changing their meaning
 **/

import type { Metadata } from "@rocky/domain"

export type { ChatMessage, ChatRole } from "./chat.js"

export interface BrainComplete extends Metadata {
  readonly text: string
}

export interface TurnRequest {
  readonly type: "turn.request"
  readonly turnId: string
  readonly messages: ReadonlyArray<import("./chat.js").ChatMessage>
  readonly memoryFacts?: ReadonlyArray<string>
  readonly groundingNotes?: string
}

export type BrainEvent =
  | { readonly _tag: "Token"; readonly text: string }
  | { readonly _tag: "Complete"; readonly response: BrainComplete }

export type TurnEvent =
  | { readonly type: "turn.started"; readonly turnId: string }
  | { readonly type: "turn.token"; readonly turnId: string; readonly text: string }
  | { readonly type: "turn.completed"; readonly turnId: string; readonly response: BrainComplete }
  | { readonly type: "turn.failed"; readonly turnId: string; readonly message: string }

export {
  buildContextMessage,
  CONTEXT_PREAMBLE,
  prepareTurnMessages,
  type TurnContextInput,
  type TurnMessagesInput,
  toTurnContextInput,
} from "./turn-context.js"
