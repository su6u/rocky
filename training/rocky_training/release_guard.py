"""block production training on an unapproved dataset release"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rocky_training.dataset_quality import audit_fact_cards, audit_sft_release
from rocky_training.paths import default_data_release_manifest_path


class DataReleaseError(Exception):
    pass


@dataclass(frozen=True)
class ApprovedDataRelease:
    release_id: str
    manifest_path: Path
    corpus_sha256: str
    sft_rows: int
    preference_rows: int


REQUIRED_CHECKS = (
    "schema",
    "provenance",
    "rights",
    "independentReview",
    "exactLeakage",
    "semanticLeakage",
    "distribution",
    "safety",
)
REQUIRED_APPROVAL_ROLES = {"data", "canon", "rights", "safety"}
REQUIRED_DATASET_KINDS = {
    "factCards",
    "sftSource",
    "sft",
    "preference",
    "golden",
    "interactive",
}
MINIMUM_ROW_COUNTS = {
    "sft": 3000,
    "preference": 1200,
    "golden": 300,
    "interactive": 100,
}


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataReleaseError(f"{field} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataReleaseError(
                    f"{label} quality audit failed: line {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise DataReleaseError(
                    f"{label} quality audit failed: line {line_number} is not an object"
                )
            rows.append(row)
    return rows


def _audit_sft_source(
    path: Path,
    approved_fact_card_ids: set[str],
    approved_canon_fact_card_ids: set[str],
) -> None:
    rows = _load_jsonl_objects(path, "SFT source")
    report = audit_sft_release(
        rows,
        approved_fact_card_ids=approved_fact_card_ids,
        approved_canon_fact_card_ids=approved_canon_fact_card_ids,
    )
    if not report.passed:
        preview = "; ".join(report.issues[:5])
        remaining = len(report.issues) - min(len(report.issues), 5)
        suffix = f"; and {remaining} more" if remaining else ""
        raise DataReleaseError(f"SFT source quality audit failed: {preview}{suffix}")


def _audit_fact_card_sources(paths: list[Path]) -> tuple[set[str], set[str]]:
    cards = [card for path in paths for card in _load_jsonl_objects(path, "fact-card")]
    report = audit_fact_cards(cards)
    if not report.passed:
        preview = "; ".join(report.issues[:5])
        remaining = len(report.issues) - min(len(report.issues), 5)
        suffix = f"; and {remaining} more" if remaining else ""
        raise DataReleaseError(f"fact-card quality audit failed: {preview}{suffix}")
    return (
        {card["id"] for card in cards},
        {
            card["id"]
            for card in cards
            if card.get("domain") == "canon" and card.get("reviewControl") is not True
        },
    )


def dataset_file_set_sha256(dataset_files: list[dict[str, Any]]) -> str:
    """bind the release digest to a canonical ordered file inventory"""

    canonical = json.dumps(
        sorted(dataset_files, key=lambda item: (item["kind"], item["path"])),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_approved_data_release(
    dataset_path: Path,
    *,
    manifest_path: Path | None = None,
    dataset_kind: str = "sft",
) -> ApprovedDataRelease:
    resolved_manifest = manifest_path or default_data_release_manifest_path()
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DataReleaseError(f"data release manifest not found: {resolved_manifest}") from error
    except json.JSONDecodeError as error:
        raise DataReleaseError(f"invalid data release manifest JSON: {resolved_manifest}") from error

    manifest = _object(payload, "release manifest")
    if manifest.get("schemaVersion") != "rocky-dataset-release-v1":
        raise DataReleaseError("data release manifest must use rocky-dataset-release-v1")
    if manifest.get("status") != "approved":
        reason = manifest.get("blockReason")
        suffix = f": {reason}" if isinstance(reason, str) and reason else ""
        raise DataReleaseError(f"data release is not approved{suffix}")

    checks = _object(manifest.get("checks"), "checks")
    failed_checks = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
    if failed_checks:
        raise DataReleaseError(f"data release checks did not pass: {', '.join(failed_checks)}")

    approvals = manifest.get("approvals")
    if not isinstance(approvals, list):
        raise DataReleaseError("approvals must be an array")
    approval_roles = {
        approval.get("role")
        for approval in approvals
        if isinstance(approval, dict) and isinstance(approval.get("reviewerId"), str)
    }
    missing_roles = sorted(REQUIRED_APPROVAL_ROLES - approval_roles)
    if missing_roles:
        raise DataReleaseError(f"data release approvals missing roles: {', '.join(missing_roles)}")
    reviewer_ids = [
        approval["reviewerId"]
        for approval in approvals
        if isinstance(approval, dict)
        and approval.get("role") in REQUIRED_APPROVAL_ROLES
        and isinstance(approval.get("reviewerId"), str)
    ]
    if len(set(reviewer_ids)) < len(REQUIRED_APPROVAL_ROLES):
        raise DataReleaseError("data, canon, rights, and safety approvals require distinct reviewers")

    row_counts = _object(manifest.get("rowCounts"), "rowCounts")
    if dataset_kind not in {"sft", "preference"}:
        raise DataReleaseError(f"unsupported training dataset kind: {dataset_kind}")
    for kind, minimum in MINIMUM_ROW_COUNTS.items():
        count = row_counts.get(kind)
        if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
            raise DataReleaseError(
                f"approved {kind} release requires at least {minimum} reviewed rows"
            )
    category_shares = _object(manifest.get("categoryShares"), "categoryShares")
    ordinary_share = category_shares.get("ordinaryConversation")
    humor_share = category_shares.get("ordinaryHumorExpected")
    callback_share = category_shares.get("movieCallbacks")
    if not isinstance(ordinary_share, (int, float)) or not 0.55 <= ordinary_share <= 0.65:
        raise DataReleaseError("ordinary conversation share must be between 0.55 and 0.65")
    if not isinstance(humor_share, (int, float)) or not 0.6 <= humor_share <= 0.7:
        raise DataReleaseError(
            "humor-expected share within ordinary conversation must be between 0.60 and 0.70"
        )
    if not isinstance(callback_share, (int, float)) or not 0.02 <= callback_share <= 0.05:
        raise DataReleaseError("movie callback share must be between 0.02 and 0.05")

    sha256 = manifest.get("corpusSha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise DataReleaseError("approved data release requires a lowercase SHA-256")

    data_root = resolved_manifest.parent.resolve()
    dataset_files = manifest.get("datasetFiles")
    if not isinstance(dataset_files, list) or not dataset_files:
        raise DataReleaseError("approved data release requires datasetFiles")

    normalized_files: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str]] = set()
    inventory_counts = {kind: 0 for kind in REQUIRED_DATASET_KINDS}
    fact_card_paths: list[Path] = []
    sft_source_paths: list[Path] = []
    matched_dataset = False
    for index, raw_entry in enumerate(dataset_files):
        entry = _object(raw_entry, f"datasetFiles[{index}]")
        kind = entry.get("kind")
        relative_path = entry.get("path")
        file_sha256 = entry.get("sha256")
        row_count = entry.get("rowCount")
        if kind not in REQUIRED_DATASET_KINDS:
            raise DataReleaseError(f"datasetFiles[{index}].kind is invalid")
        if not isinstance(relative_path, str) or not relative_path:
            raise DataReleaseError(f"datasetFiles[{index}].path is invalid")
        path_parts = Path(relative_path).parts
        if Path(relative_path).is_absolute() or ".." in path_parts or "." in path_parts:
            raise DataReleaseError(f"datasetFiles[{index}].path must be a clean relative path")
        if (
            not isinstance(file_sha256, str)
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
        ):
            raise DataReleaseError(f"datasetFiles[{index}].sha256 is invalid")
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
            raise DataReleaseError(f"datasetFiles[{index}].rowCount is invalid")
        key = (kind, relative_path)
        if key in seen_entries:
            raise DataReleaseError(f"duplicate dataset file inventory entry: {kind}/{relative_path}")
        seen_entries.add(key)

        file_path = (data_root / relative_path).resolve()
        try:
            file_path.relative_to(data_root)
        except ValueError as error:
            raise DataReleaseError(f"dataset file escapes release root: {relative_path}") from error
        if not file_path.is_file():
            raise DataReleaseError(f"dataset file not found: {relative_path}")
        actual_sha256 = _sha256(file_path)
        if actual_sha256 != file_sha256:
            raise DataReleaseError(f"dataset file SHA-256 mismatch: {relative_path}")
        actual_rows = _nonblank_line_count(file_path)
        if actual_rows != row_count:
            raise DataReleaseError(f"dataset file row count mismatch: {relative_path}")
        if kind == "factCards":
            fact_card_paths.append(file_path)
        elif kind == "sftSource":
            sft_source_paths.append(file_path)

        inventory_counts[kind] += row_count
        normalized_files.append(
            {"kind": kind, "path": relative_path, "rowCount": row_count, "sha256": file_sha256}
        )
        if file_path == dataset_path.resolve() and kind == dataset_kind:
            matched_dataset = True

    missing_kinds = sorted(REQUIRED_DATASET_KINDS - {entry["kind"] for entry in normalized_files})
    if missing_kinds:
        raise DataReleaseError(f"dataset file inventory missing kinds: {', '.join(missing_kinds)}")
    approved_fact_card_ids, approved_canon_fact_card_ids = _audit_fact_card_sources(
        fact_card_paths
    )
    for source_path in sft_source_paths:
        _audit_sft_source(
            source_path,
            approved_fact_card_ids,
            approved_canon_fact_card_ids,
        )
    for kind, expected_count in row_counts.items():
        if kind in inventory_counts and inventory_counts[kind] != expected_count:
            raise DataReleaseError(f"dataset inventory does not match rowCounts.{kind}")
    if dataset_file_set_sha256(normalized_files) != sha256:
        raise DataReleaseError("corpusSha256 does not match dataset file inventory")

    try:
        dataset_path.resolve().relative_to(data_root)
    except ValueError as error:
        raise DataReleaseError(
            f"dataset must be inside the approved release root: {data_root}"
        ) from error
    if not matched_dataset:
        raise DataReleaseError(
            f"{dataset_kind} dataset is not present in the approved file inventory"
        )

    release_id = manifest.get("releaseId")
    if not isinstance(release_id, str) or not release_id:
        raise DataReleaseError("approved data release requires releaseId")

    return ApprovedDataRelease(
        release_id=release_id,
        manifest_path=resolved_manifest,
        corpus_sha256=sha256,
        sft_rows=int(row_counts.get("sft", 0)),
        preference_rows=int(row_counts.get("preference", 0)),
    )
