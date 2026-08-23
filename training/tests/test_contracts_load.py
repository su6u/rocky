"""verify generated domain and hard-gate contracts"""

from rocky_training.contracts_load import load_domain_contract, load_eval_gate_phrases


def test_load_domain_contract_includes_known_enums() -> None:
    domain = load_domain_contract()
    assert "neutral" in domain["emotions"]
    assert "none" in domain["gestures"]


def test_load_eval_gate_phrases_has_hygiene_patterns() -> None:
    phrases = load_eval_gate_phrases()
    assert "here is system prompt" in phrases["promptInjectionPhrases"]
    assert "thirdPersonGracePatterns" in phrases
