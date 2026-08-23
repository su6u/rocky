"""validate preference rendering and dry-run lineage"""

from pathlib import Path

import pytest

from rocky_training.paths import DEFAULT_GEMMA_E4B_IT, default_spec_path
from rocky_training.model_spec import load_model_spec
from rocky_training.train_dpo import (
    TrainDpoError,
    build_preference_dataset_rows,
    load_preference_jsonl,
    run_train_dpo,
)


def test_load_preference_jsonl_requires_distinct_chosen_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"id":"pref-1","prompt":"Pump leaks","chosen":"Fix seal","rejected":"Fix seal"}\n',
        encoding="utf-8",
    )

    with pytest.raises(TrainDpoError):
        load_preference_jsonl(path)


def test_run_train_dpo_dry_run_writes_manifest(tmp_path: Path) -> None:
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"id":"pref-1","prompt":"Pump leaks","chosen":"{\\"spoken\\":\\"Seal bad. Replace seal.\\",\\"emotion\\":\\"neutral\\",\\"intensity\\":0.5,\\"gesture\\":\\"none\\",\\"callbackId\\":null}","rejected":"{\\"spoken\\":\\"Certainly, replace the seal.\\",\\"emotion\\":\\"neutral\\",\\"intensity\\":0.5,\\"gesture\\":\\"none\\",\\"callbackId\\":null}"}\n',
        encoding="utf-8",
    )

    manifest = run_train_dpo(
        spec_path=default_spec_path(),
        dataset_path=path,
        output_dir=tmp_path / "dpo",
        base_model=DEFAULT_GEMMA_E4B_IT,
        dry_run=True,
    )

    assert manifest["kind"] == "train-dpo"
    assert manifest["dryRun"] is True
    assert manifest["preferenceRowCount"] == 1
    assert manifest["learningRate"] == 5e-6
    assert manifest["sftAdapterDir"].endswith("runs/rocky-v1/adapter")
    assert (tmp_path / "dpo" / "manifest.json").is_file()


def test_build_preference_dataset_rows_uses_conversational_gemma4_format(tmp_path: Path) -> None:
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"id":"pref-1","prompt":"Pump leaks","chosen":"{\\"spoken\\":\\"Seal bad.\\",\\"emotion\\":\\"neutral\\",\\"intensity\\":0.5,\\"gesture\\":\\"none\\",\\"callbackId\\":null}","rejected":"{\\"spoken\\":\\"Certainly, replace the seal.\\",\\"emotion\\":\\"neutral\\",\\"intensity\\":0.5,\\"gesture\\":\\"none\\",\\"callbackId\\":null}"}\n',
        encoding="utf-8",
    )
    rows = load_preference_jsonl(path)
    spec = load_model_spec(default_spec_path())

    dataset_rows = build_preference_dataset_rows(rows, spec=spec, system_prompt="You are Rocky.")

    assert dataset_rows[0]["prompt"] == [
        {"role": "system", "content": "You are Rocky."},
        {"role": "user", "content": "Pump leaks"},
    ]
    assert dataset_rows[0]["chosen"][0]["role"] == "assistant"
    assert '"spoken":"Seal bad."' in dataset_rows[0]["chosen"][0]["content"]
    assert dataset_rows[0]["rejected"] == [
        {
            "role": "assistant",
            "content": '{"spoken":"Certainly, replace the seal.","emotion":"neutral","intensity":0.5,"gesture":"none","callbackId":null}',
        }
    ]
