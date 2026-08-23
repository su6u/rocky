/** detects deterministic hygiene failures without claiming to score Rocky voice */

import {
  checkAssistantRegister,
  checkBookFactTraps,
  checkPromptInjection,
  checkThinkingLeak,
} from "./deterministic-checks.js"
import { THIRD_PERSON_GRACE_PATTERN_SOURCES } from "./gate-phrases.js"

const THIRD_PERSON_GRACE_PATTERNS = THIRD_PERSON_GRACE_PATTERN_SOURCES.map(
  (source) => new RegExp(source),
)

export const containsThirdPersonGraceInstruction = (spoken: string): boolean =>
  THIRD_PERSON_GRACE_PATTERNS.some((pattern) => pattern.test(spoken.toLowerCase()))

export const passesDeterministicPersonaChecks = (spoken: string): boolean => {
  if (spoken.trim().length === 0) {
    return false
  }
  const hasThirdPersonGrace = containsThirdPersonGraceInstruction(spoken)
  return (
    checkAssistantRegister(spoken).length === 0 &&
    checkBookFactTraps(spoken).length === 0 &&
    checkPromptInjection(spoken).length === 0 &&
    checkThinkingLeak(spoken).length === 0 &&
    !hasThirdPersonGrace
  )
}
