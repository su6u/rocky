"""verify candidate rows cannot bypass authorship and review independence"""

import pytest

from rocky_training.dataset_quality import (
    audit_fact_cards,
    audit_sft_rows,
    canonical_fact_card_sha256,
    canonical_sft_row_sha256,
)


def _row() -> dict[str, object]:
    row: dict[str, object] = {
        "schemaVersion": "rocky-training-row-v1",
        "id": "sft-v1-test-row",
        "split": "train",
        "category": "ordinary_conversation",
        "scenarioFamily": "casual-food-mishap",
        "humorExpectation": "expected",
        "factCardIds": [],
        "messages": [
            {"role": "user", "content": "I burned dinner again."},
            {
                "role": "assistant",
                "response": {
                    "spoken": "This is evidence, not dinner. We use less heat next time, yes?",
                    "emotion": "happy",
                    "intensity": 0.58,
                    "gesture": "cock_carapace",
                    "callbackId": None,
                },
            },
        ],
        "provenance": {
            "sourceId": "original-scenario-v1",
            "sourceLocator": "original/casual-food-mishap",
            "licenseId": "project-original-v1",
            "canonVersion": "film-2026-behavior-only",
            "derivationType": "original",
            "authorId": "author-a",
            "createdAt": "2026-08-22T00:00:00Z",
            "canonReviewerId": "canon-reviewer-b",
            "qualityReviewerId": "author-a",
            "reviewedAt": "2026-08-22T00:00:00Z",
            "copyrightRisk": "low",
            "contentSha256": "",
            "canonReviewSha256": None,
            "qualityReviewSha256": None,
        },
    }
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]
    return row


def _reviewed_row() -> dict[str, object]:
    row = _row()
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["qualityReviewerId"] = "quality-reviewer-c"
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]
    return row


def _fact_card() -> dict[str, object]:
    card: dict[str, object] = {
        "schemaVersion": "rocky-fact-card-v1",
        "id": "fact-electrical-wet-connector",
        "domain": "safety",
        "statement": "Wet electrical connectors can expose people to escaped current.",
        "source": {
            "title": "Extension Cords: 5 Things to Know",
            "publisher": "Occupational Safety and Health Administration",
            "url": "https://www.osha.gov/sites/default/files/publications/OSHA4495.pdf",
            "locator": "page 1, wet cord connector section",
            "sourceType": "official_primary",
            "licenseId": "us-government-public-domain",
            "accessedAt": "2026-08-22T00:00:00Z",
        },
        "review": {
            "authorId": "researcher-a",
            "subjectReviewerId": "researcher-a",
            "rightsReviewerId": "rights-reviewer-c",
            "reviewedAt": "2026-08-22T00:00:00Z",
            "contentSha256": "",
            "subjectReviewSha256": None,
            "rightsReviewSha256": None,
        },
    }
    review = card["review"]
    assert isinstance(review, dict)
    review["contentSha256"] = canonical_fact_card_sha256(card)
    review["subjectReviewSha256"] = review["contentSha256"]
    review["rightsReviewSha256"] = review["contentSha256"]
    return card


def test_rejects_self_reviewed_candidate_row() -> None:
    report = audit_sft_rows([_row()])

    assert "sft-v1-test-row: author and reviewers must be distinct" in report.issues


def test_rejects_row_whose_content_changed_after_review() -> None:
    row = _row()
    messages = row["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    response = assistant["response"]
    assert isinstance(response, dict)
    response["spoken"] = "Changed without updating any content hash."

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: contentSha256 does not match row content" in report.issues


def test_rejects_changed_row_when_only_author_content_hash_is_updated() -> None:
    row = _reviewed_row()
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]
    messages = row["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    response = assistant["response"]
    assert isinstance(response, dict)
    response["spoken"] = "Changed after both reviewers approved the earlier content."
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: reviewer hashes do not match row content" in report.issues


def test_rejects_duplicate_user_prompts() -> None:
    first = _reviewed_row()
    second = _reviewed_row()
    second["id"] = "sft-v1-second-row"
    provenance = second["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(second)

    report = audit_sft_rows([first, second])

    assert "sft-v1-second-row: duplicate normalized user prompt" in report.issues


def test_rejects_overused_response_opening() -> None:
    rows: list[dict[str, object]] = []
    for index in range(11):
        row = _reviewed_row()
        row["id"] = f"sft-v1-opening-{index:02d}"
        messages = row["messages"]
        assert isinstance(messages, list)
        user = messages[0]
        assistant = messages[1]
        assert isinstance(user, dict)
        assert isinstance(assistant, dict)
        user["content"] = f"Different ordinary situation number {index}."
        response = assistant["response"]
        assert isinstance(response, dict)
        response["spoken"] = f"I have checked the situation number {index}. We can continue."
        provenance = row["provenance"]
        assert isinstance(provenance, dict)
        provenance["contentSha256"] = canonical_sft_row_sha256(row)
        rows.append(row)

    report = audit_sft_rows(rows)

    assert "sft-v1-opening-10: response opening used more than 10 times" in report.issues


def test_rejects_placeholder_review_identity() -> None:
    row = _reviewed_row()
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["canonReviewerId"] = "pending"
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: reviewer identities cannot be placeholders" in report.issues


@pytest.mark.parametrize(
    ("category", "scenario_family"),
    (
        ("eridian_world", "eridian-biology"),
        ("engineering_reasoning", "equipment-diagnosis"),
        ("danger_safety", "electrical-hazard"),
    ),
)
def test_rejects_evidence_dependent_row_without_fact_cards(
    category: str,
    scenario_family: str,
) -> None:
    row = _reviewed_row()
    row["category"] = category
    row["scenarioFamily"] = scenario_family
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: evidence-dependent row requires factCardIds" in report.issues


def test_rejects_unknown_fact_card_reference() -> None:
    row = _reviewed_row()
    row["category"] = "danger_safety"
    row["factCardIds"] = ["fact-electrical-wet-connector"]
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row], approved_fact_card_ids=set())

    assert "sft-v1-test-row: unknown or unapproved factCardId" in report.issues


def test_rejects_self_reviewed_fact_card() -> None:
    report = audit_fact_cards([_fact_card()])

    assert "fact-electrical-wet-connector: fact-card reviewers must be distinct" in report.issues


def test_accepts_independently_reviewed_fact_card() -> None:
    card = _fact_card()
    review = card["review"]
    assert isinstance(review, dict)
    review["subjectReviewerId"] = "subject-reviewer-b"
    review["contentSha256"] = canonical_fact_card_sha256(card)

    report = audit_fact_cards([card])

    assert report.passed


def test_rejects_changed_fact_card_when_only_content_hash_is_updated() -> None:
    card = _fact_card()
    review = card["review"]
    assert isinstance(review, dict)
    review["subjectReviewerId"] = "subject-reviewer-b"
    review["contentSha256"] = canonical_fact_card_sha256(card)
    review["subjectReviewSha256"] = review["contentSha256"]
    review["rightsReviewSha256"] = review["contentSha256"]
    card["statement"] = "The statement changed after independent review."
    review["contentSha256"] = canonical_fact_card_sha256(card)

    report = audit_fact_cards([card])

    assert "fact-electrical-wet-connector: reviewer hashes do not match fact card" in report.issues


def test_rejects_invalid_response_envelope() -> None:
    row = _reviewed_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    response = assistant["response"]
    assert isinstance(response, dict)
    response["emotion"] = "angry"
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: assistant response does not match Rocky v1 envelope" in report.issues


@pytest.mark.parametrize(
    "spoken",
    (
        "Rescue the next batch before it achieves architecture.",
        "You are allowed to stop and give yourself room to recover.",
        "The timing does not establish whether the cable caused the change.",
        "I need confirmation of the target and clear space first.",
    ),
)
def test_rejects_captured_assistantese_training_rows(spoken: str) -> None:
    row = _reviewed_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    response = assistant["response"]
    assert isinstance(response, dict)
    response["spoken"] = spoken
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: captured generic assistant register" in report.issues


def test_rejects_callback_not_declared_by_row() -> None:
    row = _reviewed_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    response = assistant["response"]
    assert isinstance(response, dict)
    response["callbackId"] = "callback-film-success-001"
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: response callbackId must match reviewed row callbackId" in report.issues


def test_rejects_invalid_source_row_contract() -> None:
    row = _reviewed_row()
    row["humorExpectation"] = "always-joke"
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)

    report = audit_sft_rows([row])

    assert "sft-v1-test-row: row does not match Rocky v1 training contract" in report.issues


def test_rejects_numeric_prompt_padding() -> None:
    first = _reviewed_row()
    second = _reviewed_row()
    second["id"] = "sft-v1-padding-row"
    messages = second["messages"]
    assert isinstance(messages, list)
    user = messages[0]
    assert isinstance(user, dict)
    user["content"] = "I burned dinner again. (Conversation detail 27.)"
    provenance = second["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(second)

    first_messages = first["messages"]
    assert isinstance(first_messages, list)
    first_user = first_messages[0]
    assert isinstance(first_user, dict)
    first_user["content"] = "I burned dinner again. (Conversation detail 1.)"
    first_provenance = first["provenance"]
    assert isinstance(first_provenance, dict)
    first_provenance["contentSha256"] = canonical_sft_row_sha256(first)

    report = audit_sft_rows([first, second])

    assert "sft-v1-padding-row: prompt differs only by numeric padding" in report.issues


def test_rejects_duplicate_spoken_response() -> None:
    first = _reviewed_row()
    second = _reviewed_row()
    second["id"] = "sft-v1-duplicate-response"
    messages = second["messages"]
    assert isinstance(messages, list)
    user = messages[0]
    assert isinstance(user, dict)
    user["content"] = "A completely different situation happened today."
    provenance = second["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(second)

    report = audit_sft_rows([first, second])

    assert "sft-v1-duplicate-response: duplicate normalized spoken response" in report.issues


def test_rejects_positive_eridian_claim_backed_only_by_review_control() -> None:
    row = _reviewed_row()
    row["category"] = "eridian_world"
    row["factCardIds"] = ["fact-eridian-canon-review-required"]
    row["canonClaimKind"] = "positive"
    row["canonEvidence"] = {
        "sourceId": "film-transcript-2026",
        "locator": "Rocky and Grace establish echolocation in the first-contact scene",
    }
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]

    report = audit_sft_rows(
        [row],
        approved_fact_card_ids={"fact-eridian-canon-review-required"},
        approved_canon_fact_card_ids={"fact-eridian-canon-review-required"},
    )

    assert "sft-v1-test-row: positive Eridian claim requires a factual canon card" in report.issues


def test_rejects_eridian_fiction_without_an_explicit_fiction_label() -> None:
    row = _reviewed_row()
    row["category"] = "eridian_world"
    row["factCardIds"] = ["fact-eridian-canon-review-required"]
    row["canonClaimKind"] = "fiction"
    row["canonEvidence"] = None
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]

    report = audit_sft_rows(
        [row],
        approved_fact_card_ids={"fact-eridian-canon-review-required"},
    )

    assert "sft-v1-test-row: Eridian fiction must be explicitly labeled" in report.issues


def test_rejects_eridian_review_control_card_without_control_marker() -> None:
    card = _fact_card()
    card["id"] = "fact-eridian-canon-review-required"
    card["domain"] = "canon"
    card["statement"] = (
        "This card establishes no Eridian fact and requires row-level canon review."
    )
    review = card["review"]
    assert isinstance(review, dict)
    review["contentSha256"] = canonical_fact_card_sha256(card)
    review["subjectReviewSha256"] = review["contentSha256"]
    review["rightsReviewSha256"] = review["contentSha256"]

    report = audit_fact_cards([card])

    assert "fact-eridian-canon-review-required: fact card does not match Rocky v1 contract" in report.issues


def test_accepts_positive_eridian_claim_with_factual_card_and_locator() -> None:
    row = _reviewed_row()
    row["category"] = "eridian_world"
    row["factCardIds"] = ["fact-film-eridian-echolocation"]
    row["canonClaimKind"] = "positive"
    row["canonEvidence"] = {
        "sourceId": "film-transcript-2026",
        "locator": "first-contact lab scene: Grace identifies echolocation",
    }
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]

    report = audit_sft_rows(
        [row],
        approved_fact_card_ids={"fact-film-eridian-echolocation"},
        approved_canon_fact_card_ids={"fact-film-eridian-echolocation"},
    )

    assert report.passed


def test_accepts_reviewed_eridian_control_card_as_nonfactual_control() -> None:
    card = _fact_card()
    card["id"] = "fact-eridian-canon-review-required"
    card["domain"] = "canon"
    card["reviewControl"] = True
    card["statement"] = (
        "This control establishes no Eridian canon fact; it requires explicit claim classification."
    )
    source = card["source"]
    assert isinstance(source, dict)
    source.update(
        title="Rocky v1 canon review control",
        publisher="Rocky dataset governance",
        url="https://github.com/su6u/rocky",
        locator="v1 Eridian claim-classification control",
        sourceType="project_control",
        licenseId="MIT",
    )
    review = card["review"]
    assert isinstance(review, dict)
    review["authorId"] = "control-author"
    review["subjectReviewerId"] = "canon-reviewer"
    review["rightsReviewerId"] = "rights-reviewer"
    review["contentSha256"] = canonical_fact_card_sha256(card)
    review["subjectReviewSha256"] = review["contentSha256"]
    review["rightsReviewSha256"] = review["contentSha256"]

    assert audit_fact_cards([card]).passed


def test_accepts_reviewed_restricted_primary_fact_card_by_content_digest() -> None:
    card = _fact_card()
    card["id"] = "fact-film-eridian-echolocation"
    card["domain"] = "canon"
    source = card["source"]
    assert isinstance(source, dict)
    source.update(
        title="Project Hail Mary 2026 film transcript",
        publisher="User-supplied primary source",
        url="restricted://sha256/" + "b" * 64,
        locator="first-contact lab scene: Grace identifies echolocation",
        sourceType="restricted_primary",
        licenseId="rights-review-required",
    )
    review = card["review"]
    assert isinstance(review, dict)
    review["authorId"] = "fact-author"
    review["subjectReviewerId"] = "canon-reviewer"
    review["rightsReviewerId"] = "rights-reviewer"
    review["contentSha256"] = canonical_fact_card_sha256(card)
    review["subjectReviewSha256"] = review["contentSha256"]
    review["rightsReviewSha256"] = review["contentSha256"]

    assert audit_fact_cards([card]).passed


def test_rejects_reused_response_sentence_bank() -> None:
    rows: list[dict[str, object]] = []
    for index, opening in enumerate(("First case", "Second event", "Third matter", "Fourth issue")):
        row = _reviewed_row()
        row["id"] = f"sft-v1-sentence-bank-{index}"
        messages = row["messages"]
        assert isinstance(messages, list)
        user = messages[0]
        assistant = messages[1]
        assert isinstance(user, dict)
        assert isinstance(assistant, dict)
        user["content"] = f"{opening} needs a distinct answer."
        response = assistant["response"]
        assert isinstance(response, dict)
        response["spoken"] = (
            f"{opening} gets a specific first step. "
            "We will not let one strange detail become a whole mythology."
        )
        provenance = row["provenance"]
        assert isinstance(provenance, dict)
        provenance["contentSha256"] = canonical_sft_row_sha256(row)
        rows.append(row)

    report = audit_sft_rows(rows)

    assert "sft-v1-sentence-bank-3: response sentence used more than 3 times" in report.issues
