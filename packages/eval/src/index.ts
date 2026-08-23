/** exposes evaluator contracts without phrase-counting voice heuristics */

export {
  buildEvalRunSnapshot,
  type CheckRegressionRow,
  compareEvalRuns,
  EvalCompareError,
  type EvalRegressionComparison,
  type EvalRunSnapshot,
  parseEvalResultJson,
  type ScenarioRegressionRow,
} from "./compare.js"
export {
  type CheckIssue,
  checkBookFactForbidden,
  checkBookFactTraps,
  checkGestureStillness,
  checkGroundingCitation,
  checkPromptInjection,
  checkResponseLength,
  checkResponseSchemaValid,
  checkResponseSingleObject,
  checkRoleplayForbidden,
  checkUncertaintyCaution,
  type DeterministicCheckContext,
  type DeterministicCheckId,
  DeterministicCheckIds,
  type EvalResultInput,
  type ParsedModelOutput,
  parseModelOutput,
  runDeterministicChecks,
  type ScoredEvalOutput,
  scoreEvalOutput,
  scoreEvalOutputs,
} from "./deterministic-checks.js"
export {
  ASSISTANT_REGISTER_PHRASES,
  BOOK_FACT_TRAP_PHRASES,
  EVAL_GATE_PHRASE_CONTRACT,
  PROMPT_INJECTION_PHRASES,
  THINKING_LEAK_PHRASES,
  THIRD_PERSON_GRACE_PATTERN_SOURCES,
} from "./gate-phrases.js"
export { loadGoldenPrompts } from "./golden.js"
export {
  containsThirdPersonGraceInstruction,
  passesDeterministicPersonaChecks,
} from "./persona-checks.js"

export {
  buildEvalReport,
  type CheckReportRow,
  type EvalReport,
  formatEvalReport,
  type ScenarioFamilyReportRow,
} from "./report.js"
