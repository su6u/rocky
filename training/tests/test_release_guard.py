"""verify that data quantity never bypasses release quality gates"""

import hashlib
import json
from pathlib import Path

import pytest

from rocky_training.release_guard import (
    DataReleaseError,
    dataset_file_set_sha256,
    require_approved_data_release,
)
from rocky_training.dataset_quality import (
    canonical_fact_card_sha256,
    canonical_sft_row_sha256,
)


def _fixture_label(index: int) -> str:
    letters: list[str] = []
    value = index
    while True:
        letters.append(chr(ord("a") + value % 26))
        value = value // 26 - 1
        if value < 0:
            return "".join(reversed(letters))


def _synthetic_sft_source_row(index: int, category: str) -> dict[str, object]:
    label = _fixture_label(index)
    callback_id = f"callback-synthetic-{index:04d}" if index < 90 else None
    row: dict[str, object] = {
        "schemaVersion": "rocky-training-row-v1",
        "id": f"sft-v1-fixture-{index:04d}",
        "split": "train",
        "category": category,
        "scenarioFamily": f"fixture-{category}-{index:04d}",
        "humorExpectation": "expected" if category == "ordinary_conversation" and index < 1170 else "optional",
        "factCardIds": ["fact-fixture-evidence"]
        if category in {"engineering_reasoning", "eridian_world", "danger_safety"}
        else [],
        "callbackId": callback_id,
        "messages": [
            {"role": "user", "content": f"Synthetic fixture prompt {label}."},
            {
                "role": "assistant",
                "response": {
                    "spoken": f"Synthetic fixture response {label} stays distinct.",
                    "emotion": "neutral",
                    "intensity": 0.5,
                    "gesture": "none",
                    "callbackId": callback_id,
                },
            },
        ],
        "provenance": {
            "sourceId": "test-fixture-only",
            "sourceLocator": f"fixture/{index:04d}",
            "licenseId": "test-only",
            "canonVersion": "test-only",
            "derivationType": "original",
            "authorId": "fixture-author",
            "createdAt": "2026-08-22T00:00:00Z",
            "canonReviewerId": "fixture-canon-reviewer",
            "qualityReviewerId": "fixture-quality-reviewer",
            "reviewedAt": "2026-08-22T00:00:00Z",
            "copyrightRisk": "low",
            "contentSha256": "",
            "canonReviewSha256": None,
            "qualityReviewSha256": None,
        },
    }
    if category == "eridian_world":
        row["canonClaimKind"] = "positive"
        row["canonEvidence"] = {
            "sourceId": "test-fixture-only",
            "locator": f"synthetic canon evidence {index:04d}",
        }
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["contentSha256"] = canonical_sft_row_sha256(row)
    provenance["canonReviewSha256"] = provenance["contentSha256"]
    provenance["qualityReviewSha256"] = provenance["contentSha256"]
    return row


def _synthetic_sft_source_rows(total: int) -> list[dict[str, object]]:
    allocations = (
        ("ordinary_conversation", total - 1200),
        ("engineering_reasoning", 360),
        ("grace_relationship", 300),
        ("eridian_world", 210),
        ("ambiguity_repair", 150),
        ("danger_safety", 120),
        ("output_motion", 60),
    )
    rows: list[dict[str, object]] = []
    for category, count in allocations:
        for _ in range(count):
            rows.append(_synthetic_sft_source_row(len(rows), category))
    return rows


def _synthetic_fact_card() -> dict[str, object]:
    card: dict[str, object] = {
        "schemaVersion": "rocky-fact-card-v1",
        "id": "fact-fixture-evidence",
        "domain": "canon",
        "statement": "Synthetic evidence exists only to exercise release plumbing.",
        "source": {
            "title": "Synthetic test source",
            "publisher": "Rocky test suite",
            "url": "https://example.invalid/rocky-test-source",
            "locator": "test fixture",
            "sourceType": "official_primary",
            "licenseId": "test-only",
            "accessedAt": "2026-08-22T00:00:00Z",
        },
        "review": {
            "authorId": "fixture-fact-author",
            "subjectReviewerId": "fixture-subject-reviewer",
            "rightsReviewerId": "fixture-rights-reviewer",
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


def _manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schemaVersion": "rocky-dataset-release-v1",
        "releaseId": "rocky-data-v1-test",
        "status": "approved",
        "corpusSha256": None,
        "datasetFiles": [],
        "rowCounts": {"sft": 3000, "preference": 1200, "golden": 300, "interactive": 100},
        "categoryShares": {
            "ordinaryConversation": 0.6,
            "ordinaryHumorExpected": 0.65,
            "movieCallbacks": 0.03,
        },
        "checks": {
            "schema": True,
            "provenance": True,
            "rights": True,
            "independentReview": True,
            "exactLeakage": True,
            "semanticLeakage": True,
            "distribution": True,
            "safety": True,
        },
        "approvals": [
            {"reviewerId": "data-reviewer", "role": "data"},
            {"reviewerId": "canon-reviewer", "role": "canon"},
            {"reviewerId": "rights-reviewer", "role": "rights"},
            {"reviewerId": "safety-reviewer", "role": "safety"},
        ],
    }
    manifest.update(overrides)
    return manifest


def _write_release(
    tmp_path: Path,
    manifest: dict[str, object],
    *,
    include_fact_cards: bool = True,
    include_sft_source: bool = True,
) -> tuple[Path, Path]:
    release_root = tmp_path / "data" / "v1"
    row_counts = manifest["rowCounts"]
    assert isinstance(row_counts, dict)
    files = {
        "sft": ("exports/sft.jsonl", row_counts["sft"]),
        "preference": ("exports/preferences.jsonl", row_counts["preference"]),
        "golden": ("eval/golden.jsonl", row_counts["golden"]),
        "interactive": ("eval/interactive.jsonl", row_counts["interactive"]),
    }
    inventory: list[dict[str, object]] = []
    if include_fact_cards:
        fact_cards_path = release_root / "sources" / "fact-cards.jsonl"
        fact_cards_path.parent.mkdir(parents=True, exist_ok=True)
        fact_cards_path.write_text(
            json.dumps(_synthetic_fact_card(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        inventory.append(
            {
                "kind": "factCards",
                "path": "sources/fact-cards.jsonl",
                "rowCount": 1,
                "sha256": hashlib.sha256(fact_cards_path.read_bytes()).hexdigest(),
            }
        )
    if include_sft_source:
        source_path = release_root / "corpus" / "sft-source.jsonl"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_rows = _synthetic_sft_source_rows(row_counts["sft"])
        assert len(source_rows) == row_counts["sft"]
        source_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in source_rows),
            encoding="utf-8",
        )
        inventory.append(
            {
                "kind": "sftSource",
                "path": "corpus/sft-source.jsonl",
                "rowCount": row_counts["sft"],
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        )
    for kind, (relative_path, row_count) in files.items():
        file_path = release_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{}\n" * row_count, encoding="utf-8")
        inventory.append(
            {
                "kind": kind,
                "path": relative_path,
                "rowCount": row_count,
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
        )
    manifest["datasetFiles"] = inventory
    manifest["corpusSha256"] = dataset_file_set_sha256(inventory)
    dataset_path = release_root / files["sft"][0]
    manifest_path = release_root / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_path, manifest_path


def test_accepts_only_complete_approved_release(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(tmp_path, _manifest())

    release = require_approved_data_release(dataset_path, manifest_path=manifest_path)

    assert release.release_id == "rocky-data-v1-test"
    assert release.sft_rows == 3000


def test_rejects_approved_release_with_2999_sft_rows(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(
        tmp_path,
        _manifest(
            rowCounts={"sft": 2999, "preference": 1200, "golden": 300, "interactive": 100}
        ),
    )

    with pytest.raises(DataReleaseError, match="at least 3000 reviewed rows"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_release_without_audited_sft_source_inventory(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(
        tmp_path,
        _manifest(),
        include_sft_source=False,
    )

    with pytest.raises(DataReleaseError, match="sftSource"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_release_without_audited_fact_card_inventory(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(
        tmp_path,
        _manifest(),
        include_fact_cards=False,
    )

    with pytest.raises(DataReleaseError, match="factCards"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_invalid_sft_source_rows(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(tmp_path, _manifest())
    sft_source = manifest_path.parent / "corpus" / "sft-source.jsonl"
    sft_source.write_text("{}\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entry = next(entry for entry in manifest["datasetFiles"] if entry["kind"] == "sftSource")
    source_entry["rowCount"] = 1
    source_entry["sha256"] = hashlib.sha256(sft_source.read_bytes()).hexdigest()
    manifest["corpusSha256"] = dataset_file_set_sha256(manifest["datasetFiles"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataReleaseError, match="SFT source quality audit"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_invalid_fact_cards(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(tmp_path, _manifest())
    fact_cards = manifest_path.parent / "sources" / "fact-cards.jsonl"
    fact_cards.write_text("{}\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fact_entry = next(entry for entry in manifest["datasetFiles"] if entry["kind"] == "factCards")
    fact_entry["sha256"] = hashlib.sha256(fact_cards.read_bytes()).hexdigest()
    manifest["corpusSha256"] = dataset_file_set_sha256(manifest["datasetFiles"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataReleaseError, match="fact-card quality audit"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_blocked_release_even_when_dataset_exists(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(
        tmp_path,
        _manifest(status="blocked", blockReason="rights review incomplete"),
    )

    with pytest.raises(DataReleaseError, match="rights review incomplete"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_quantity_without_all_quality_checks(tmp_path: Path) -> None:
    manifest = _manifest()
    checks = dict(manifest["checks"])
    checks["semanticLeakage"] = False
    manifest["checks"] = checks
    dataset_path, manifest_path = _write_release(tmp_path, manifest)

    with pytest.raises(DataReleaseError, match="semanticLeakage"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_dataset_outside_release_root(tmp_path: Path) -> None:
    _, manifest_path = _write_release(tmp_path, _manifest())
    outside = tmp_path / "old-research.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DataReleaseError, match="approved release root"):
        require_approved_data_release(outside, manifest_path=manifest_path)


def test_rejects_file_content_that_no_longer_matches_approval(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(tmp_path, _manifest())
    dataset_path.write_text("{\"tampered\":true}\n", encoding="utf-8")

    with pytest.raises(DataReleaseError, match="SHA-256 mismatch"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_claimed_row_count_that_differs_from_file(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_release(tmp_path, _manifest())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasetFiles"][0]["rowCount"] = 2401
    manifest["corpusSha256"] = dataset_file_set_sha256(manifest["datasetFiles"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataReleaseError, match="row count mismatch"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)


def test_rejects_one_reviewer_approving_multiple_independent_roles(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["approvals"] = [
        {"reviewerId": "same-person", "role": role}
        for role in ("data", "canon", "rights", "safety")
    ]
    dataset_path, manifest_path = _write_release(tmp_path, manifest)

    with pytest.raises(DataReleaseError, match="distinct reviewers"):
        require_approved_data_release(dataset_path, manifest_path=manifest_path)
