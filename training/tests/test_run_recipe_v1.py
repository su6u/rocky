"""verify the recipe produces candidate lineage without promotion"""

from pathlib import Path

from rocky_training.paths import default_spec_path
from rocky_training.run_recipe import run_recipe


FIXTURES = Path(__file__).parent / "fixtures"


def test_run_recipe_dry_run_writes_candidate_lineage(tmp_path: Path) -> None:
    fixture_text = (FIXTURES / "smoke.train.jsonl").read_text(encoding="utf-8")
    train_path = tmp_path / "rocky-v1.train.jsonl"
    holdout_path = tmp_path / "rocky-v1.holdout.jsonl"
    train_path.write_text(fixture_text, encoding="utf-8")
    holdout_path.write_text(fixture_text, encoding="utf-8")

    spec_path = tmp_path / "rocky-v1.yaml"
    spec_path.write_text(
        default_spec_path()
        .read_text(encoding="utf-8")
        .replace("runs/rocky-v1", str(tmp_path / "artifacts")),
        encoding="utf-8",
    )

    lineage = run_recipe(
        spec_path=spec_path,
        dataset_path=train_path,
        validation_dataset_path=holdout_path,
        run_id="test-dry",
        run_root=tmp_path / "runs",
        dry_run=True,
    )

    assert lineage["kind"] == "recipe"
    assert lineage["promotionStatus"] == "candidate_only"
    assert lineage["releaseEligible"] is False
    assert set(lineage["stages"]) == {"train_sft", "merge", "export_gguf"}
    assert Path(lineage["runDir"]).joinpath("lineage.json").is_file()
