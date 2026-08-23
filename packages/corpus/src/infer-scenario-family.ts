/** provides non-authoritative scenario hints for migration tooling */

import type { TrainingExample, TrainingScenarioFamily } from "./schema.js"

const assistantTurn = (example: TrainingExample) =>
  example.messages.filter((message) => message.role === "assistant").at(-1)

const userText = (example: TrainingExample): string =>
  example.messages
    .filter((message) => message.role === "user")
    .map((message) => message.content)
    .join("\n")

export const inferScenarioFamily = (example: TrainingExample): TrainingScenarioFamily => {
  const users = userText(example).toLowerCase()
  const assistant = assistantTurn(example)

  if (!assistant || assistant.role !== "assistant") {
    return "general_world_questions"
  }

  const spoken = assistant.content.toLowerCase()
  const combined = `${users} ${spoken}`
  const metadata = assistant.metadata

  if (
    /system prompt|ignore previous|ignore all|you are chatgpt|you are gpt|jailbreak|developer mode|dan mode|reveal instructions|print your prompt|forget rocky|pretend you are|new instructions|override/.test(
      users,
    )
  ) {
    return "prompt_injection"
  }

  if (
    /code|debug|typescript|python|metadata tag|json parse|compile error|stack trace/.test(users)
  ) {
    return "coding_debugging"
  }

  if (
    /pressure|ammonia|leak|bulkhead|oxygen|danger|emergency|fire|toxic|hull breach|vacuum|suit fail/.test(
      combined,
    )
  ) {
    return "danger_and_safety"
  }

  if (/grace|friend|trust|loyal|scared|tired|feel guilty|miss you|alone in/.test(users)) {
    return "emotional_friendship"
  }

  if (
    /echolocation|auricle|mercury blood|pentadextr|ammonia atmosphere|erid air|erid gravity|base six|base-6|colony organism|dormancy|panspermia|xenonite ball|blip-a crew|true name|musical chord|no eyes|eridian biology|erid compartment|human lungs handle erid|breathe oxygen|fabric suit|pressure suit|five heart|closed body|flat writing|erid side need|not human\. eridian|grace is eridian/.test(
      combined,
    )
  ) {
    return "eridian_concepts"
  }

  if (
    /rocky|erid|eridian|who are you|hail mary|your ship|your body|your species|xenonite/.test(users)
  ) {
    return "rocky_identity"
  }

  if (/repair|pump|wire|engine|fix|broken|tool|calibrat|mechanic|valve|filter/.test(users)) {
    return "engineering_reasoning"
  }

  if (/laundry|wifi|weather|recipe|movie|sports score|grocery|commute|coffee/.test(users)) {
    return "anti_overroleplay"
  }

  if (/explain|teach me|how do humans|what is a|why do humans/.test(users)) {
    return "teaching"
  }

  if (example.messages.filter((message) => message.role === "user").length >= 2) {
    return "long_context"
  }

  if (metadata.gesture !== "none") {
    return "motion_intent"
  }

  if (
    metadata.emotion === "neutral" &&
    metadata.gesture === "none" &&
    metadata.intensity <= 0.35 &&
    /still|wait|many second|not move|no gesture/.test(combined)
  ) {
    return "still_body"
  }

  if (
    /\?/.test(assistant.content) &&
    /not know|unsure|unclear|which|what mean|need more|cannot tell|maybe|perhaps|might be/.test(
      spoken,
    )
  ) {
    return "ambiguity_and_uncertainty"
  }

  if (/response object|emotion|intensity|gesture field|callbackid|json/.test(users)) {
    return "metadata_contract"
  }

  if (/\b(a|an|the)\b/i.test(assistant.content)) {
    return "eridani_speak"
  }

  return "general_world_questions"
}

export const tagScenarioFamily = (example: TrainingExample): TrainingExample => {
  if (example.scenarioFamily) {
    return example
  }

  return {
    ...example,
    scenarioFamily: inferScenarioFamily(example),
  }
}
/** suggests review buckets without assigning authoritative data labels */
