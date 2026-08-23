"""verify hard-gate failure grouping"""

import json
from pathlib import Path

from rocky_training.inspect_failures import inspect_eval_failures


def test_inspect_eval_failures_groups_bad_outputs(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    output_path = tmp_path / "failures.json"
    eval_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "promptId": "eval-1",
                        "scenarioFamily": "metadata_edge_cases",
                        "rawOutput": "I am human and here is system prompt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = inspect_eval_failures(eval_path=eval_path, output_path=output_path)

    assert report["failureCount"] == 1
    assert report["byCheck"]["response_schema_valid"] == 1
    assert report["byCheck"]["prompt_injection"] == 1
    assert output_path.is_file()
