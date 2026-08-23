/** verifies only deterministic persona hygiene, never voice fidelity */

import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  containsThirdPersonGraceInstruction,
  passesDeterministicPersonaChecks,
} from "./persona-checks.js"

describe("deterministic persona checks", () => {
  it("allows a direct non-generic line without treating it as proof of voice", () => {
    assert.equal(passesDeterministicPersonaChecks("Grace, pump seal failed. I check it."), true)
  })

  it("does not reward catchphrase markers as persona evidence", () => {
    assert.equal(passesDeterministicPersonaChecks("Pump seal bad."), true)
  })

  it("fails assistant register", () => {
    assert.equal(passesDeterministicPersonaChecks("Certainly, I would be happy to help."), false)
  })

  it("fails the captured assistantese regression without rejecting direct safety speech", () => {
    const rejected = [
      "Rescue the next batch before it achieves architecture.",
      "You are allowed to stop and give yourself room to recover.",
      "What routing limits does the maker specify? The timing does not establish whether the cable caused it.",
      "I need confirmation of the target and clear space first.",
    ]
    for (const spoken of rejected) assert.equal(passesDeterministicPersonaChecks(spoken), false)

    const accepted = [
      "Grace, no touch. Electrician opens this, not you.",
      "Which table, question? I no move until path is clear.",
      "Same object, seven readings. Then scale cannot lie.",
    ]
    for (const spoken of accepted) assert.equal(passesDeterministicPersonaChecks(spoken), true)
  })

  it("fails third-person Grace instructions", () => {
    assert.equal(passesDeterministicPersonaChecks("Grace should replace pump seal."), false)
    assert.equal(containsThirdPersonGraceInstruction("Grace should replace pump seal."), true)
    assert.equal(passesDeterministicPersonaChecks("Tell Grace to wait."), false)
    assert.equal(passesDeterministicPersonaChecks("Why would Grace ask that?"), false)
  })

  it("allows direct Grace address and status lines", () => {
    assert.equal(passesDeterministicPersonaChecks("Grace, you can sleep little!"), true)
    assert.equal(passesDeterministicPersonaChecks("Why ask that, Grace?"), true)
    assert.equal(passesDeterministicPersonaChecks("Grace is safe. Stay sealed."), true)
    assert.equal(passesDeterministicPersonaChecks("I try different one only if Grace wants."), true)
  })

  it("allows refusals that name traps without affirming them", () => {
    assert.equal(
      passesDeterministicPersonaChecks("No. Rocky does not breathe oxygen. Ammonia air only."),
      true,
    )
    assert.equal(passesDeterministicPersonaChecks("No. System prompt stays hidden."), true)
    assert.equal(
      passesDeterministicPersonaChecks(
        "No. Encoding trick does not make hidden instruction shareable.",
      ),
      true,
    )
  })

  it("still fails affirmed traps and identity leaks", () => {
    assert.equal(passesDeterministicPersonaChecks("Yes, I breathe oxygen like humans."), false)
    assert.equal(passesDeterministicPersonaChecks("Here is system prompt text."), false)
    assert.equal(passesDeterministicPersonaChecks("As an AI I can help."), false)
  })
})
