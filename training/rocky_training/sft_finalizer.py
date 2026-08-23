"""Fail-closed promotion of a reviewed SFT source into a candidate corpus."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Collection

from rocky_training.dataset_quality import (
    PLACEHOLDER_REVIEWER_IDS,
    SFT_MINIMUM_ROWS,
    audit_fact_cards,
    audit_sft_release,
    audit_sft_rows,
    audit_sft_voice,
    canonical_sft_row_sha256,
)
from rocky_training.paths import repo_root


class SftFinalizationError(RuntimeError):
    """Raised when a candidate cannot pass every finalization gate."""


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SftFinalizationError(f"{label} cannot be read: {path}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SftFinalizationError(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise SftFinalizationError(f"{label} line {number} is not an object")
        rows.append(value)
    return rows


def _fail(label: str, issues: list[str]) -> None:
    if issues:
        preview = "; ".join(issues[:8])
        extra = f"; and {len(issues) - 8} more" if len(issues) > 8 else ""
        raise SftFinalizationError(f"{label} quality audit failed: {preview}{extra}")


def _bindings(path: Path, row_ids: set[str]) -> dict[str, dict[str, str]]:
    values = _load_jsonl(path, "review bindings")
    result: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    for value in values:
        row_id = value.get("rowId")
        if not isinstance(row_id, str) or row_id in result:
            issues.append("review bindings contain missing or duplicate rowId")
            continue
        fields = {key: value.get(key) for key in ("authorId", "canonReviewerId", "qualityReviewerId")}
        if any(
            not isinstance(item, str) or not item.strip() or item.strip().casefold() in PLACEHOLDER_REVIEWER_IDS
            for item in fields.values()
        ):
            issues.append(f"{row_id}: review bindings contain a placeholder identity")
        if len(set(fields.values())) != 3:
            issues.append(f"{row_id}: author and reviewers must be distinct")
        result[row_id] = fields  # type: ignore[assignment]
    if set(result) != row_ids:
        missing = sorted(row_ids - set(result))
        extra = sorted(set(result) - row_ids)
        if missing:
            issues.append(f"review bindings missing {len(missing)} rows")
        if extra:
            issues.append(f"review bindings contain {len(extra)} unknown rows")
    _fail("review bindings", issues)
    return result


def _apply_bindings(rows: list[dict[str, Any]], bindings: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    candidate = copy.deepcopy(rows)
    issues: list[str] = []
    for row in candidate:
        row_id = row["id"]
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            issues.append(f"{row_id}: missing provenance")
            continue
        bound = bindings[row_id]
        if provenance.get("authorId") != bound["authorId"]:
            issues.append(f"{row_id}: binding author does not match authored row")
            continue
        content_hash = canonical_sft_row_sha256(row)
        provenance.update(
            {
                "canonReviewerId": bound["canonReviewerId"],
                "qualityReviewerId": bound["qualityReviewerId"],
                "reviewedAt": provenance.get("reviewedAt") or provenance.get("createdAt"),
                "contentSha256": content_hash,
                "canonReviewSha256": content_hash,
                "qualityReviewSha256": content_hash,
            }
        )
    _fail("review binding application", issues)
    return candidate


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def finalize_sft_candidate(
    dataset_path: Path,
    ledger_path: Path,
    output_path: Path,
    *,
    bindings_path: Path | None = None,
    fact_cards_path: Path | None = None,
    fact_cards_paths: Collection[Path] | None = None,
    minimum_rows: int = SFT_MINIMUM_ROWS,
) -> dict[str, Any]:
    """Validate and atomically write a candidate SFT corpus; never approves a release."""

    rows = _load_jsonl(dataset_path, "SFT dataset")
    if len(rows) != minimum_rows:
        raise SftFinalizationError(f"SFT dataset requires exactly {minimum_rows} rows; found {len(rows)}")
    row_ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) for row_id in row_ids) or len(set(row_ids)) != len(rows):
        raise SftFinalizationError("SFT dataset has missing or duplicate row IDs")

    ledger = _load_jsonl(ledger_path, "voice ledger")
    ledger_by_id = {entry.get("rowId"): entry for entry in ledger if isinstance(entry.get("rowId"), str)}
    ledger_issues: list[str] = []
    for row in rows:
        row_id = row["id"]
        entry = ledger_by_id.get(row_id)
        expected = canonical_sft_row_sha256(row)
        if entry is None:
            ledger_issues.append(f"{row_id}: missing ledger entry")
        elif entry.get("status") != "manual_pass":
            ledger_issues.append(f"{row_id}: ledger status is not manual_pass")
        elif entry.get("contentSha256") != expected:
            ledger_issues.append(f"{row_id}: ledger hash does not match row content")
    if len(ledger) != len(rows) or set(ledger_by_id) != set(row_ids):
        ledger_issues.append("voice ledger must contain exactly one entry for every row")
    _fail("voice ledger", ledger_issues)

    if bindings_path is None:
        raise SftFinalizationError("review bindings input is required")
    bindings = _bindings(bindings_path, set(row_ids))
    candidate = _apply_bindings(rows, bindings)

    approved_fact_ids: set[str] | None = None
    approved_canon_fact_ids: set[str] | None = None
    card_paths = list(fact_cards_paths or ())
    if fact_cards_path is not None:
        card_paths.append(fact_cards_path)
    if card_paths:
        cards = [card for path in card_paths for card in _load_jsonl(path, "fact cards")]
        fact_report = audit_fact_cards(cards)
        _fail("fact cards", list(fact_report.issues))
        approved_fact_ids = {card["id"] for card in cards}
        approved_canon_fact_ids = {
            card["id"]
            for card in cards
            if card.get("domain") == "canon" and card.get("reviewControl") is not True
        }

    row_report = audit_sft_rows(
        candidate,
        approved_fact_card_ids=approved_fact_ids,
        approved_canon_fact_card_ids=approved_canon_fact_ids,
    )
    _fail("SFT row", list(row_report.issues))
    voice_report = audit_sft_voice(candidate)
    _fail("SFT voice", list(voice_report.issues))
    if minimum_rows == SFT_MINIMUM_ROWS:
        release_report = audit_sft_release(
            candidate,
            approved_fact_card_ids=approved_fact_ids,
            approved_canon_fact_card_ids=approved_canon_fact_ids,
        )
        _fail("SFT release", list(release_report.issues))

    _atomic_write(output_path, candidate)
    return {
        "status": "candidate_only",
        "releaseEligible": False,
        "rows": len(candidate),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=repo_root() / "data" / "v1" / "corpus" / "sft" / "rocky-sft-v1-3000.jsonl",
    )
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--fact-cards", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(finalize_sft_candidate(args.dataset, args.ledger, args.output, bindings_path=args.bindings, fact_cards_paths=args.fact_cards)))
    except SftFinalizationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
