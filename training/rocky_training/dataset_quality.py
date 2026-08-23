"""audit candidate SFT rows before they can enter a release inventory"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Iterable

from rocky_training.eval_gates import passes_deterministic_persona_checks, response_is_valid

PLACEHOLDER_REVIEWER_IDS = {"", "pending", "todo", "tbd", "unknown", "unreviewed", "none", "n/a"}
SFT_MINIMUM_ROWS = 3000
CATEGORY_SHARE_RANGES = {
    "ordinary_conversation": (0.55, 0.65),
    "engineering_reasoning": (0.10, 0.14),
    "grace_relationship": (0.08, 0.12),
    "eridian_world": (0.05, 0.09),
    "ambiguity_repair": (0.04, 0.07),
    "danger_safety": (0.03, 0.06),
    "output_motion": (0.015, 0.035),
}
HUMOR_EXPECTATIONS = {"expected", "optional", "avoid"}
DERIVATION_TYPES = {"original", "paraphrase", "licensed_quote", "synthetic_revision"}
COPYRIGHT_RISKS = {"low", "medium", "high"}
EVIDENCE_DEPENDENT_CATEGORIES = {
    "engineering_reasoning",
    "eridian_world",
    "danger_safety",
}
ERIDIAN_CANON_REVIEW_CONTROL_ID = "fact-eridian-canon-review-required"
ERIDIAN_CANON_CLAIM_KINDS = {"boundary", "fiction", "positive"}
TOP_LEVEL_FIELDS = {
    "schemaVersion",
    "id",
    "split",
    "category",
    "scenarioFamily",
    "humorExpectation",
    "callbackId",
    "factCardIds",
    "messages",
    "provenance",
    "canonClaimKind",
    "canonEvidence",
}
PROVENANCE_FIELDS = {
    "sourceId",
    "sourceLocator",
    "licenseId",
    "canonVersion",
    "derivationType",
    "authorId",
    "createdAt",
    "canonReviewerId",
    "qualityReviewerId",
    "reviewedAt",
    "copyrightRisk",
    "contentSha256",
    "canonReviewSha256",
    "qualityReviewSha256",
}
FACT_CARD_REQUIRED_FIELDS = {"schemaVersion", "id", "domain", "statement", "source", "review"}
FACT_CARD_FIELDS = FACT_CARD_REQUIRED_FIELDS | {"reviewControl"}
FACT_CARD_SOURCE_FIELDS = {
    "title",
    "publisher",
    "url",
    "locator",
    "sourceType",
    "licenseId",
    "accessedAt",
}
FACT_CARD_REVIEW_FIELDS = {
    "authorId",
    "subjectReviewerId",
    "rightsReviewerId",
    "reviewedAt",
    "contentSha256",
    "subjectReviewSha256",
    "rightsReviewSha256",
}
FACT_CARD_DOMAINS = {"canon", "engineering", "safety"}
FACT_CARD_SOURCE_TYPES = {
    "licensed_primary",
    "official_primary",
    "manufacturer_manual",
    "project_control",
    "restricted_primary",
}


@dataclass(frozen=True)
class DatasetQualityReport:
    row_count: int
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def canonical_sft_row_sha256(row: dict[str, Any]) -> str:
    canonical_row = copy.deepcopy(row)
    provenance = canonical_row.get("provenance")
    if isinstance(provenance, dict):
        for field in (
            "canonReviewerId",
            "qualityReviewerId",
            "reviewedAt",
            "contentSha256",
            "canonReviewSha256",
            "qualityReviewSha256",
        ):
            provenance.pop(field, None)
    payload = json.dumps(
        canonical_row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_fact_card_sha256(card: dict[str, Any]) -> str:
    canonical_card = copy.deepcopy(card)
    review = canonical_card.get("review")
    if isinstance(review, dict):
        for field in (
            "subjectReviewerId",
            "rightsReviewerId",
            "reviewedAt",
            "contentSha256",
            "subjectReviewSha256",
            "rightsReviewSha256",
        ):
            review.pop(field, None)
    payload = json.dumps(
        canonical_card,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_user_trajectory(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    user_texts = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    normalized = unicodedata.normalize("NFKC", "\n".join(user_texts)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _prompt_template_fingerprint(normalized_prompt: str) -> str:
    return re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", normalized_prompt)


def _response_openings(row: dict[str, Any]) -> tuple[str, ...]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ()
    openings: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("spoken"), str):
            continue
        tokens = re.findall(r"[\w']+", response["spoken"].casefold())
        if len(tokens) >= 4:
            openings.append(" ".join(tokens[:4]))
    return tuple(openings)


def _normalized_spoken_responses(row: dict[str, Any]) -> tuple[str, ...]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ()
    responses: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("spoken"), str):
            continue
        normalized = unicodedata.normalize("NFKC", response["spoken"]).casefold()
        responses.append(re.sub(r"\s+", " ", normalized).strip())
    return tuple(responses)


def _response_sentences(row: dict[str, Any]) -> tuple[str, ...]:
    sentences: list[str] = []
    for response in _normalized_spoken_responses(row):
        for sentence in re.split(r"[.!?]+", response):
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if len(normalized.split()) >= 6:
                sentences.append(normalized)
    return tuple(sentences)


def _responses_are_valid(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    assistant_count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_count += 1
        response = message.get("response")
        if not isinstance(response, dict) or not response_is_valid(
            json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        ):
            return False
    return assistant_count > 0


def _responses_pass_hygiene(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    spoken_values = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("spoken"), str):
            return False
        spoken_values.append(response["spoken"])
    return bool(spoken_values) and all(
        passes_deterministic_persona_checks(spoken) for spoken in spoken_values
    )


def _response_callback_ids(row: dict[str, Any]) -> tuple[str, ...]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ()
    callback_ids: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        response = message.get("response")
        if isinstance(response, dict) and isinstance(response.get("callbackId"), str):
            callback_ids.append(response["callbackId"])
    return tuple(callback_ids)


def _eridian_fiction_is_explicitly_labeled(row: dict[str, Any]) -> bool:
    labels = re.compile(
        r"\b(?:fiction|fan(?:[- ]fiction| story| scene| scenario| setting)?|"
        r"invent(?:ed|ion)?|non-canon|not canon|not Erid fact|alternate (?:fiction|setting|story))\b",
        re.IGNORECASE,
    )
    return any(labels.search(spoken) for spoken in _normalized_spoken_responses(row))


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_optional_sha256(value: object) -> bool:
    return value is None or (
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None
    )


def _fact_card_matches_contract(card: dict[str, Any]) -> bool:
    if not FACT_CARD_REQUIRED_FIELDS <= set(card) or not set(card) <= FACT_CARD_FIELDS:
        return False
    if card.get("schemaVersion") != "rocky-fact-card-v1":
        return False
    if not isinstance(card.get("id"), str) or re.fullmatch(
        r"fact-[a-z0-9-]+", card["id"]
    ) is None:
        return False
    if card.get("domain") not in FACT_CARD_DOMAINS or not _is_nonempty_string(
        card.get("statement")
    ):
        return False
    is_review_control = card.get("reviewControl", False)
    if not isinstance(is_review_control, bool):
        return False
    if card.get("id") == ERIDIAN_CANON_REVIEW_CONTROL_ID:
        if is_review_control is not True or card.get("domain") != "canon":
            return False
    elif is_review_control:
        return False
    source = card.get("source")
    if not isinstance(source, dict) or set(source) != FACT_CARD_SOURCE_FIELDS:
        return False
    if any(
        not _is_nonempty_string(source.get(field))
        for field in ("title", "publisher", "url", "locator", "licenseId")
    ):
        return False
    url = source.get("url")
    if not isinstance(url, str):
        return False
    if source.get("sourceType") not in FACT_CARD_SOURCE_TYPES or not _is_datetime(
        source.get("accessedAt")
    ):
        return False
    if (source.get("sourceType") == "project_control") != is_review_control:
        return False
    if source.get("sourceType") == "restricted_primary":
        if re.fullmatch(r"restricted://sha256/[a-f0-9]{64}", url) is None:
            return False
    elif not url.startswith("https://"):
        return False
    review = card.get("review")
    if not isinstance(review, dict) or set(review) != FACT_CARD_REVIEW_FIELDS:
        return False
    if any(
        not _is_nonempty_string(review.get(field))
        for field in ("authorId", "subjectReviewerId", "rightsReviewerId")
    ):
        return False
    if not _is_datetime(review.get("reviewedAt")):
        return False
    content_sha256 = review.get("contentSha256")
    return (
        isinstance(content_sha256, str)
        and re.fullmatch(r"[a-f0-9]{64}", content_sha256) is not None
        and _is_optional_sha256(review.get("subjectReviewSha256"))
        and _is_optional_sha256(review.get("rightsReviewSha256"))
    )


def audit_fact_cards(cards: Iterable[dict[str, Any]]) -> DatasetQualityReport:
    materialized = list(cards)
    issues: list[str] = []
    seen_ids: set[str] = set()
    for index, card in enumerate(materialized):
        card_id = card.get("id") if isinstance(card.get("id"), str) else f"fact-card-{index + 1}"
        if not _fact_card_matches_contract(card):
            issues.append(f"{card_id}: fact card does not match Rocky v1 contract")
        if card_id in seen_ids:
            issues.append(f"{card_id}: duplicate fact-card id")
        seen_ids.add(card_id)
        review = card.get("review")
        if not isinstance(review, dict):
            continue
        reviewer_ids = (
            review.get("authorId"),
            review.get("subjectReviewerId"),
            review.get("rightsReviewerId"),
        )
        if len(set(reviewer_ids)) != 3:
            issues.append(f"{card_id}: fact-card reviewers must be distinct")
        if any(
            not isinstance(reviewer_id, str)
            or reviewer_id.strip().casefold() in PLACEHOLDER_REVIEWER_IDS
            for reviewer_id in reviewer_ids
        ):
            issues.append(f"{card_id}: fact-card reviewer identities cannot be placeholders")
        if review.get("contentSha256") != canonical_fact_card_sha256(card):
            issues.append(f"{card_id}: fact-card contentSha256 does not match content")
        content_sha256 = review.get("contentSha256")
        approved_review_hashes = (
            (review.get("subjectReviewerId"), review.get("subjectReviewSha256")),
            (review.get("rightsReviewerId"), review.get("rightsReviewSha256")),
        )
        if any(
            isinstance(reviewer_id, str)
            and reviewer_id.strip().casefold() not in PLACEHOLDER_REVIEWER_IDS
            and review_sha256 != content_sha256
            for reviewer_id, review_sha256 in approved_review_hashes
        ):
            issues.append(f"{card_id}: reviewer hashes do not match fact card")
    return DatasetQualityReport(row_count=len(materialized), issues=tuple(issues))


def _row_matches_contract(row: dict[str, Any]) -> bool:
    if not set(row) <= TOP_LEVEL_FIELDS:
        return False
    if row.get("schemaVersion") != "rocky-training-row-v1":
        return False
    if not isinstance(row.get("id"), str) or re.fullmatch(r"sft-v1-[a-z0-9-]+", row["id"]) is None:
        return False
    if row.get("split") not in {"train", "holdout"}:
        return False
    if row.get("category") not in CATEGORY_SHARE_RANGES:
        return False
    if row.get("category") == "eridian_world":
        claim_kind = row.get("canonClaimKind")
        evidence = row.get("canonEvidence")
        if claim_kind not in ERIDIAN_CANON_CLAIM_KINDS:
            return False
        if claim_kind == "positive":
            if (
                not isinstance(evidence, dict)
                or set(evidence) != {"sourceId", "locator"}
                or not _is_nonempty_string(evidence.get("sourceId"))
                or not _is_nonempty_string(evidence.get("locator"))
            ):
                return False
        elif evidence is not None:
            return False
    elif "canonClaimKind" in row or "canonEvidence" in row:
        return False
    if not _is_nonempty_string(row.get("scenarioFamily")):
        return False
    if row.get("humorExpectation") not in HUMOR_EXPECTATIONS:
        return False
    callback_id = row.get("callbackId")
    if callback_id is not None and (
        not isinstance(callback_id, str)
        or re.fullmatch(r"callback-[a-z0-9-]+", callback_id) is None
    ):
        return False
    fact_card_ids = row.get("factCardIds", [])
    if (
        not isinstance(fact_card_ids, list)
        or len(fact_card_ids) != len(set(fact_card_ids))
        or any(
            not isinstance(fact_card_id, str)
            or re.fullmatch(r"fact-[a-z0-9-]+", fact_card_id) is None
            for fact_card_id in fact_card_ids
        )
    ):
        return False

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        if role in {"system", "user"}:
            if set(message) != {"role", "content"} or not _is_nonempty_string(message.get("content")):
                return False
            has_user = has_user or role == "user"
        elif role == "assistant":
            if set(message) != {"role", "response"} or not isinstance(message.get("response"), dict):
                return False
            has_assistant = True
        else:
            return False
    if not has_user or not has_assistant:
        return False

    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
        return False
    string_fields = {
        "sourceId",
        "sourceLocator",
        "licenseId",
        "canonVersion",
        "authorId",
        "canonReviewerId",
        "qualityReviewerId",
    }
    if any(not _is_nonempty_string(provenance.get(field)) for field in string_fields):
        return False
    if provenance.get("derivationType") not in DERIVATION_TYPES:
        return False
    if provenance.get("copyrightRisk") not in COPYRIGHT_RISKS:
        return False
    if not _is_datetime(provenance.get("createdAt")) or not _is_datetime(
        provenance.get("reviewedAt")
    ):
        return False
    content_sha256 = provenance.get("contentSha256")
    return (
        isinstance(content_sha256, str)
        and re.fullmatch(r"[a-f0-9]{64}", content_sha256) is not None
        and _is_optional_sha256(provenance.get("canonReviewSha256"))
        and _is_optional_sha256(provenance.get("qualityReviewSha256"))
    )


def audit_sft_rows(
    rows: Iterable[dict[str, Any]],
    *,
    approved_fact_card_ids: Collection[str] | None = None,
    approved_canon_fact_card_ids: Collection[str] | None = None,
) -> DatasetQualityReport:
    materialized = list(rows)
    issues: list[str] = []
    seen_user_trajectories: set[str] = set()
    seen_prompt_templates: set[str] = set()
    seen_spoken_responses: set[str] = set()
    response_sentence_counts: dict[str, int] = {}
    opening_counts: dict[str, int] = {}

    for index, row in enumerate(materialized):
        row_id = row.get("id") if isinstance(row.get("id"), str) else f"row-{index + 1}"
        if not _row_matches_contract(row):
            issues.append(f"{row_id}: row does not match Rocky v1 training contract")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            continue
        reviewer_ids = (
            provenance.get("authorId"),
            provenance.get("canonReviewerId"),
            provenance.get("qualityReviewerId"),
        )
        if len(set(reviewer_ids)) != 3:
            issues.append(f"{row_id}: author and reviewers must be distinct")
        if any(
            not isinstance(reviewer_id, str)
            or reviewer_id.strip().casefold() in PLACEHOLDER_REVIEWER_IDS
            for reviewer_id in reviewer_ids
        ):
            issues.append(f"{row_id}: reviewer identities cannot be placeholders")
        if provenance.get("contentSha256") != canonical_sft_row_sha256(row):
            issues.append(f"{row_id}: contentSha256 does not match row content")
        content_sha256 = provenance.get("contentSha256")
        approved_review_hashes = (
            (provenance.get("canonReviewerId"), provenance.get("canonReviewSha256")),
            (provenance.get("qualityReviewerId"), provenance.get("qualityReviewSha256")),
        )
        if any(
            isinstance(reviewer_id, str)
            and reviewer_id.strip().casefold() not in PLACEHOLDER_REVIEWER_IDS
            and review_sha256 != content_sha256
            for reviewer_id, review_sha256 in approved_review_hashes
        ):
            issues.append(f"{row_id}: reviewer hashes do not match row content")
        fact_card_ids = row.get("factCardIds")
        if row.get("category") in EVIDENCE_DEPENDENT_CATEGORIES and (
            not isinstance(fact_card_ids, list) or not fact_card_ids
        ):
            issues.append(f"{row_id}: evidence-dependent row requires factCardIds")
        if approved_fact_card_ids is not None and isinstance(fact_card_ids, list):
            if any(fact_card_id not in approved_fact_card_ids for fact_card_id in fact_card_ids):
                issues.append(f"{row_id}: unknown or unapproved factCardId")
        if row.get("category") == "eridian_world":
            claim_kind = row.get("canonClaimKind")
            if claim_kind in {"boundary", "fiction"} and (
                not isinstance(fact_card_ids, list)
                or ERIDIAN_CANON_REVIEW_CONTROL_ID not in fact_card_ids
            ):
                issues.append(f"{row_id}: Eridian boundary or fiction row requires review control")
            if claim_kind == "positive":
                factual_canon_ids = set(approved_canon_fact_card_ids or ()) - {
                    ERIDIAN_CANON_REVIEW_CONTROL_ID
                }
                if not isinstance(fact_card_ids, list) or not any(
                    fact_card_id in factual_canon_ids for fact_card_id in fact_card_ids
                ):
                    issues.append(
                        f"{row_id}: positive Eridian claim requires a factual canon card"
                    )
            if claim_kind == "fiction" and not _eridian_fiction_is_explicitly_labeled(row):
                issues.append(f"{row_id}: Eridian fiction must be explicitly labeled")
        if not _responses_are_valid(row):
            issues.append(f"{row_id}: assistant response does not match Rocky v1 envelope")
        if not _responses_pass_hygiene(row):
            issues.append(f"{row_id}: captured generic assistant register")
        if any(callback_id != row.get("callbackId") for callback_id in _response_callback_ids(row)):
            issues.append(f"{row_id}: response callbackId must match reviewed row callbackId")
        user_trajectory = _normalized_user_trajectory(row)
        if user_trajectory and user_trajectory in seen_user_trajectories:
            issues.append(f"{row_id}: duplicate normalized user prompt")
        prompt_template = _prompt_template_fingerprint(user_trajectory)
        if (
            prompt_template
            and prompt_template in seen_prompt_templates
            and user_trajectory not in seen_user_trajectories
        ):
            issues.append(f"{row_id}: prompt differs only by numeric padding")
        seen_user_trajectories.add(user_trajectory)
        seen_prompt_templates.add(prompt_template)
        for spoken_response in _normalized_spoken_responses(row):
            if spoken_response and spoken_response in seen_spoken_responses:
                issues.append(f"{row_id}: duplicate normalized spoken response")
            seen_spoken_responses.add(spoken_response)
        for sentence in _response_sentences(row):
            response_sentence_counts[sentence] = response_sentence_counts.get(sentence, 0) + 1
            if response_sentence_counts[sentence] == 4:
                issues.append(f"{row_id}: response sentence used more than 3 times")
        for opening in _response_openings(row):
            opening_counts[opening] = opening_counts.get(opening, 0) + 1
            if opening_counts[opening] == 11:
                issues.append(f"{row_id}: response opening used more than 10 times")

    return DatasetQualityReport(row_count=len(materialized), issues=tuple(issues))


def audit_sft_voice(rows: Iterable[dict[str, Any]]) -> DatasetQualityReport:
    materialized = list(rows)
    issues: list[str] = []
    seen_token_bags: set[tuple[str, ...]] = set()
    five_gram_counts: dict[tuple[str, ...], int] = {}
    question_count = 0
    amaze_count = 0
    bad_count = 0

    for index, row in enumerate(materialized):
        row_id = row.get("id") if isinstance(row.get("id"), str) else f"row-{index + 1}"
        user_text = " ".join(
            message.get("content", "")
            for message in row.get("messages", [])
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ).casefold()
        user_tokens = set(re.findall(r"[a-z0-9']+", user_text))
        raw_spoken = [
            message.get("response", {}).get("spoken", "")
            for message in row.get("messages", [])
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("response"), dict)
            and isinstance(message.get("response", {}).get("spoken"), str)
        ]
        raw_user = [
            message.get("content", "")
            for message in row.get("messages", [])
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        if any(re.search(r"\bGrace(?:'s)?\b", text, re.IGNORECASE) for text in raw_user + raw_spoken):
            issues.append(f"{row_id}: conversation must address Grace directly")
        if any(re.search(r"\bwhat would Rocky do\b", text, re.IGNORECASE) for text in raw_user):
            issues.append(f"{row_id}: Grace must address Rocky directly")
        if any(re.search(r"[.!?]\s+[a-z]", text) for text in raw_spoken):
            issues.append(f"{row_id}: lowercase sentence start")
        grace_gender_pattern = re.compile(
            r"(?:\bgrace\b[^.!?]{0,80}\b(?:she|her|hers|herself)\b|"
            r"\b(?:she|her|hers|herself)\b[^.!?]{0,80}\bgrace\b)",
            re.IGNORECASE,
        )
        if any(grace_gender_pattern.search(text) for text in raw_spoken):
            issues.append(f"{row_id}: Ryland Grace gender mismatch")
        for spoken_response in _normalized_spoken_responses(row):
            tokens = tuple(re.findall(r"[a-z0-9']+", spoken_response))
            five_grams = {tokens[position : position + 5] for position in range(len(tokens) - 4)}
            has_new_overused_scaffold = False
            for five_gram in five_grams:
                five_gram_counts[five_gram] = five_gram_counts.get(five_gram, 0) + 1
                if five_gram_counts[five_gram] == 11:
                    has_new_overused_scaffold = True
            if has_new_overused_scaffold:
                issues.append(f"{row_id}: response five-word scaffold used more than 10 times")
            token_bag = tuple(sorted(tokens))
            if len(tokens) >= 6 and token_bag in seen_token_bags:
                issues.append(f"{row_id}: duplicate spoken token bag")
            seen_token_bags.add(token_bag)

            has_question = "question?" in spoken_response
            has_amaze = re.search(r"\bamaze\W+amaze\W+amaze\b", spoken_response) is not None
            has_bad = re.search(r"\bbad\W+bad\W+bad\b", spoken_response) is not None
            question_count += has_question
            amaze_count += has_amaze
            bad_count += has_bad

            category = row.get("category")
            humor = row.get("humorExpectation")
            if has_amaze and (
                category in {"danger_safety", "ambiguity_repair", "output_motion"}
                or (humor == "avoid" and category not in {"engineering_reasoning", "eridian_world"})
            ):
                issues.append(f"{row_id}: Amaze triple does not match discovery or delight")
            if has_bad and category != "danger_safety":
                issues.append(f"{row_id}: Bad triple requires danger_safety")
            artifact_patterns = (
                (r"\bwhich\s+which\b", "malformed repeated question word"),
                (r"\bremember clue\b", "generator clue suffix"),
                (r"\bgood\s+[^.!?]{1,40}\s+now clue\b", "malformed engineering clue"),
                (r"\bfor do\b", "malformed engineering subject"),
                (r"\bquestion\?\.", "malformed Question punctuation"),
                (r"\brocky answer:\s*deal with\b", "generic Rocky-answer wrapper"),
            )
            for pattern, label in artifact_patterns:
                if re.search(pattern, spoken_response):
                    issues.append(f"{row_id}: {label}")
            if row.get("category") != "ambiguity_repair" and len(user_tokens) >= 7:
                assistant_tokens = set(tokens)
                prompt_recall = len(user_tokens & assistant_tokens) / len(user_tokens)
                if prompt_recall >= 0.85:
                    issues.append(f"{row_id}: assistant repeats nearly the full user prompt")

    total = len(materialized)
    if total:
        signature_ranges = (
            (question_count / total, 0.10, 0.28, "Question?"),
            (amaze_count / total, 0.01, 0.03, "Amaze triple"),
            (bad_count / total, 0.01, 0.03, "Bad triple"),
        )
        for share, minimum, maximum, label in signature_ranges:
            if not minimum <= share <= maximum:
                issues.append(
                    f"{label} response share must be between {minimum:.2f} and {maximum:.2f}"
                )

    return DatasetQualityReport(row_count=total, issues=tuple(issues))


def audit_sft_release(
    rows: Iterable[dict[str, Any]],
    *,
    approved_fact_card_ids: Collection[str] | None = None,
    approved_canon_fact_card_ids: Collection[str] | None = None,
) -> DatasetQualityReport:
    materialized = list(rows)
    row_report = audit_sft_rows(
        materialized,
        approved_fact_card_ids=approved_fact_card_ids,
        approved_canon_fact_card_ids=approved_canon_fact_card_ids,
    )
    issues = list(row_report.issues)
    total = len(materialized)

    if total < SFT_MINIMUM_ROWS:
        issues.append(f"release requires at least {SFT_MINIMUM_ROWS} SFT rows")
    if total == 0:
        return DatasetQualityReport(row_count=0, issues=tuple(issues))

    for category, (minimum, maximum) in CATEGORY_SHARE_RANGES.items():
        share = sum(row.get("category") == category for row in materialized) / total
        if not minimum <= share <= maximum:
            issues.append(
                f"{category} share {share:.4f} outside required range {minimum:.3f}-{maximum:.3f}"
            )

    ordinary_rows = [row for row in materialized if row.get("category") == "ordinary_conversation"]
    ordinary_humor_share = (
        0.0
        if not ordinary_rows
        else sum(row.get("humorExpectation") == "expected" for row in ordinary_rows)
        / len(ordinary_rows)
    )
    if not 0.60 <= ordinary_humor_share <= 0.70:
        issues.append(
            "humor-expected share within ordinary conversation must be between 0.60 and 0.70"
        )

    callback_share = sum(isinstance(row.get("callbackId"), str) for row in materialized) / total
    if not 0.02 <= callback_share <= 0.05:
        issues.append("movie callback share must be between 0.02 and 0.05")

    return DatasetQualityReport(row_count=total, issues=tuple(issues))
