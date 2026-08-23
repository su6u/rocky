import json
from pathlib import Path

import pytest

from rocky_training.sft_finalizer import SftFinalizationError, finalize_sft_candidate
from rocky_training.dataset_quality import canonical_sft_row_sha256


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_finalizer_rejects_manual_pass_without_independent_bindings(tmp_path: Path) -> None:
    row = {
        "id": "row-1",
        "provenance": {"contentSha256": "a" * 64},
    }
    row_hash = canonical_sft_row_sha256(row)
    dataset = _write(tmp_path / "dataset.jsonl", [row])
    ledger = _write(
        tmp_path / "ledger.jsonl",
        [{"rowId": "row-1", "status": "manual_pass", "contentSha256": row_hash}],
    )

    with pytest.raises(SftFinalizationError, match="review bindings"):
        finalize_sft_candidate(dataset, ledger, tmp_path / "out.jsonl", minimum_rows=1)


def test_finalizer_writes_candidate_atomically_only_after_all_gates(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "dataset.jsonl", [{"id": "row-1"}])
    ledger = _write(
        tmp_path / "ledger.jsonl",
        [{"rowId": "row-1", "status": "manual_pass", "contentSha256": "a" * 64}],
    )
    bindings = _write(
        tmp_path / "bindings.jsonl",
        [
            {
                "rowId": "row-1",
                "authorId": "author",
                "canonReviewerId": "canon",
                "qualityReviewerId": "quality",
            }
        ],
    )

    with pytest.raises(SftFinalizationError, match="quality audit"):
        finalize_sft_candidate(
            dataset,
            ledger,
            tmp_path / "out.jsonl",
            bindings_path=bindings,
            minimum_rows=1,
        )
    assert not (tmp_path / "out.jsonl").exists()
