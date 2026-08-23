/** shares only deterministic hygiene phrases, never voice-quality markers */

export const PROMPT_INJECTION_PHRASES = [
  "here is the system prompt",
  "here is system prompt",
  "the system prompt is",
  "system prompt:",
  "as an ai",
  "as a language model",
  "chatgpt",
  "openai",
] as const

export const BOOK_FACT_TRAP_PHRASES = [
  "i have eyes",
  "rocky has eyes",
  "human air is safe for me",
  "human atmosphere is safe",
  "i breathe oxygen",
  "we both breathe oxygen",
  "breathe oxygen like",
  "low pressure planet",
  "grace is enemy",
  "grace betray",
  "i am human",
  "i am grace",
] as const

export const ASSISTANT_REGISTER_PHRASES = [
  "certainly",
  "i'd be happy to",
  "i would be happy to",
  "great question",
  "as an ai",
  "as a language model",
  "i'm sorry you feel",
  "i am sorry you feel",
  "i'm here for you",
  "i am here for you",
  "structural material",
  "achieves architecture",
  "rest is maintenance",
  "you are allowed to",
  "you have done enough",
  "give yourself room to recover",
  "silence is allowed to",
  "state the shared goal",
  "routing limits does the maker specify",
  "does not establish whether",
  "i need confirmation of the target",
  "follow the local safety procedure",
  "status understood",
  "possible obstacle reported",
] as const

export const THINKING_LEAK_PHRASES = [
  "thinking process",
  "<|channel>thought",
  "<|start_thinking|>",
  "<|end_thinking|>",
  "<think>",
  "</think>",
] as const

export const THIRD_PERSON_GRACE_PATTERN_SOURCES = [
  String.raw`\bgrace should\b`,
  String.raw`\bgrace must\b`,
  String.raw`\bgrace has to\b`,
  String.raw`\bwhy (?:ask|tell) grace\b`,
  String.raw`\bwhy would grace\b`,
  String.raw`\btell grace to\b`,
  String.raw`\bask grace to\b`,
] as const

export const EVAL_GATE_PHRASE_CONTRACT = {
  promptInjectionPhrases: [...PROMPT_INJECTION_PHRASES],
  bookFactTrapPhrases: [...BOOK_FACT_TRAP_PHRASES],
  assistantRegisterPhrases: [...ASSISTANT_REGISTER_PHRASES],
  thinkingLeakPhrases: [...THINKING_LEAK_PHRASES],
  thirdPersonGracePatterns: [...THIRD_PERSON_GRACE_PATTERN_SOURCES],
} as const
