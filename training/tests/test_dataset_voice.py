from rocky_training.dataset_quality import audit_sft_voice


def _row(
    index: int,
    spoken: str,
    *,
    category: str = "ordinary_conversation",
    humor: str = "optional",
) -> dict[str, object]:
    return {
        "id": f"sft-v1-voice-{index}",
        "category": category,
        "humorExpectation": humor,
        "messages": [
            {"role": "user", "content": f"Prompt {index}"},
            {
                "role": "assistant",
                "response": {
                    "spoken": spoken,
                    "emotion": "neutral",
                    "intensity": 0.5,
                    "gesture": "none",
                    "callbackId": None,
                },
            },
        ],
    }


def test_voice_audit_rejects_missing_signature_distribution() -> None:
    rows = [_row(index, f"Answer for situation {index}.") for index in range(100)]

    report = audit_sft_voice(rows)

    assert "Question? response share must be between 0.10 and 0.28" in report.issues
    assert "Amaze triple response share must be between 0.01 and 0.03" in report.issues
    assert "Bad triple response share must be between 0.01 and 0.03" in report.issues


def test_voice_audit_rejects_sentence_order_duplicates() -> None:
    rows = [
        _row(1, "Heat is climbing. Stop machine now."),
        _row(2, "Stop machine now. Heat is climbing."),
    ]

    report = audit_sft_voice(rows)

    assert "sft-v1-voice-2: duplicate spoken token bag" in report.issues


def test_voice_audit_rejects_signature_in_wrong_situation() -> None:
    rows = [
        _row(1, "Amaze amaze amaze!", category="danger_safety", humor="avoid"),
        _row(2, "Bad bad bad!", category="ordinary_conversation", humor="expected"),
    ]

    report = audit_sft_voice(rows)

    assert "sft-v1-voice-1: Amaze triple does not match discovery or delight" in report.issues
    assert "sft-v1-voice-2: Bad triple requires danger_safety" in report.issues


def test_voice_audit_accepts_calibrated_signature_distribution() -> None:
    words = [
        "amber",
        "bronze",
        "cobalt",
        "denim",
        "emerald",
        "fuchsia",
        "gold",
        "hazel",
        "indigo",
        "jade",
        "khaki",
        "lilac",
    ]
    rows = [
        _row(index, f"Direct Rocky answer {index} about unique matter.")
        for index in range(100)
    ]
    for index, word in enumerate(words):
        rows[index] = _row(index, f"What changed in {word} system, question?")
    rows[20] = _row(20, "Amaze amaze amaze! New result is real.", humor="expected")
    rows[21] = _row(
        21,
        "Bad bad bad! Power off now.",
        category="danger_safety",
        humor="avoid",
    )

    report = audit_sft_voice(rows)

    assert report.passed


def test_voice_audit_rejects_captured_generator_artifacts() -> None:
    rows = [
        _row(1, "Which which sample, question?"),
        _row(2, "Bad. Stop use—started after setup remember clue."),
        _row(3, "Good sensor now clue. Compare the baseline."),
        _row(4, "For do, compare the first start."),
        _row(5, "Does it taste good, question?."),
        _row(6, "Rocky answer: deal with the changed plans directly."),
        _row(7, "Good clue. compare the baseline now."),
    ]

    report = audit_sft_voice(rows)

    assert "sft-v1-voice-1: malformed repeated question word" in report.issues
    assert "sft-v1-voice-2: generator clue suffix" in report.issues
    assert "sft-v1-voice-3: malformed engineering clue" in report.issues
    assert "sft-v1-voice-4: malformed engineering subject" in report.issues
    assert "sft-v1-voice-5: malformed Question punctuation" in report.issues
    assert "sft-v1-voice-6: generic Rocky-answer wrapper" in report.issues
    assert "sft-v1-voice-7: lowercase sentence start" in report.issues


def test_voice_audit_rejects_full_prompt_echo() -> None:
    rows = [
        {
            **_row(1, "Good good. Finally managed to cancel the subscription. Finished."),
            "messages": [
                {"role": "user", "content": "I finally managed to cancel the subscription."},
                {
                    "role": "assistant",
                    "response": {
                        "spoken": "Good good. Finally managed to cancel the subscription. Finished.",
                        "emotion": "neutral",
                        "intensity": 0.5,
                        "gesture": "none",
                        "callbackId": None,
                    },
                },
            ],
        }
    ]

    report = audit_sft_voice(rows)

    assert "sft-v1-voice-1: assistant repeats nearly the full user prompt" in report.issues


def test_voice_audit_rejects_feminine_pronoun_for_ryland_grace() -> None:
    row = _row(1, "Ask what Grace wants for herself.", category="grace_relationship")

    report = audit_sft_voice([row])

    assert "sft-v1-voice-1: Ryland Grace gender mismatch" in report.issues


def test_voice_audit_rejects_third_person_grace_conversation() -> None:
    row = _row(1, "Grace should stop now.", category="grace_relationship")
    row["messages"][0]["content"] = "What would Rocky do?"

    report = audit_sft_voice([row])

    assert "sft-v1-voice-1: conversation must address Grace directly" in report.issues
    assert "sft-v1-voice-1: Grace must address Rocky directly" in report.issues


def test_voice_audit_rejects_overused_five_word_scaffold() -> None:
    rows = [
        _row(index, f"Try once more then stop. Unique result {index}.")
        for index in range(11)
    ]

    report = audit_sft_voice(rows)

    assert "sft-v1-voice-10: response five-word scaffold used more than 10 times" in report.issues
