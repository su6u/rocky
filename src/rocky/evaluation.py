"""Record complete, reviewable trials; deterministic checks are not persona scores."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rocky.data import DATA, digest, read_jsonl, verify_export
from rocky.inference import GenerationSettings, InferenceClient, format_failure, load_tokenizer
from rocky.prompting import system_message


def evaluate(endpoint: str, model: str, output: Path, split: str = "validation", mode: str = "rollout",
             *, suite: str = "dialogues", client=None, settings: GenerationSettings | None = None) -> dict:
    verify_export()
    if split not in {"validation", "test"} or mode not in {"rollout", "continuation"}:
        raise ValueError("choose held-out split and rollout/continuation mode")
    if suite not in {"dialogues", "probes", "trajectories"}:
        raise ValueError("unknown evaluation suite")
    if suite != "dialogues" and (split != "validation" or mode != "rollout"):
        raise ValueError("diagnostic suites use validation/rollout; they are not the final test set")
    settings = settings or GenerationSettings()
    client = client or InferenceClient(endpoint, model, load_tokenizer(), settings)
    if isinstance(client, InferenceClient):
        settings = client.settings
    from rocky.config import load_config

    config = load_config()
    path = DATA / (f"exports/{split}.jsonl" if suite == "dialogues" else f"eval/{suite}.jsonl")
    rows = read_jsonl(path)
    manifest = json.loads((DATA / "exports/manifest.json").read_text())
    instruction = (DATA / "system.txt").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        def record(value):
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()

        record({"type": "run", "endpoint": endpoint, "model": model, "split": split, "mode": mode,
                "suite": suite, "data_sha256": digest(path), "system_sha256": digest(DATA / "system.txt"),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "expected_base": config["model_id"], "expected_revision": config["revision"],
                "generation": settings.record(), "model_identity_verified": False, "human_review_required": True})
        count, format_failures, errors, runtime_checks = 0, 0, 0, 0
        for index, row in enumerate(rows):
            identifier = manifest["row_ids"][split][index] if suite == "dialogues" else row["id"]
            if suite == "probes" and row["kind"] == "runtime-boundary":
                from rocky.session import SceneSession
                try:
                    SceneSession(row["scene"])
                except ValueError:
                    rejected = True
                else:
                    rejected = False
                runtime_checks += 1
                errors += not rejected
                record({"type": "runtime_check", "id": identifier, "unsupported_scene_rejected": rejected,
                        "note": "No model call; this tests the scene gate, not character quality."})
                continue
            if suite == "dialogues":
                messages = row["messages"]
                steps = None
            else:
                messages = [system_message(instruction, row["context"])]
                steps = row["steps"] if suite == "trajectories" else [row]
                for step in steps:
                    messages.extend([{"role": "user", "content": step["prompt"]},
                                     {"role": "assistant", "content": ""}])
            history, position = [], 0
            for message in messages:
                if message["role"] != "assistant":
                    history.append(message)
                    continue
                started = time.monotonic()
                try:
                    response = client.complete(history)
                except (ValueError, OSError, TimeoutError) as error:
                    errors += 1
                    record({"type": "error", "id": identifier, "turn": position, "messages": history,
                            "error": type(error).__name__, "detail": str(error),
                            "latency_seconds": time.monotonic() - started})
                    break
                malformed = format_failure(response)
                result = {"type": "reply", "id": identifier, "turn": position, "messages": list(history),
                          "response": response, "format_failure": malformed,
                          "latency_seconds": time.monotonic() - started}
                if steps is None:
                    result.update(expected=message["content"], source_exact_match=response == message["content"])
                else:
                    result.update(must=steps[position]["must"], must_not=steps[position]["must_not"])
                record(result)
                count += 1
                position += 1
                format_failures += malformed
                if malformed:
                    break
                history.append({"role": "assistant", "content": response} if mode == "rollout" else message)
        summary = {"type": "summary", "cases": len(rows), "replies": count, "format_failures": format_failures,
                   "errors": errors, "runtime_checks": runtime_checks, "persona_score": None,
                   "status": "completed_with_failures" if errors or format_failures else "recorded_for_review",
                   "note": "Review full transcripts. Fixed Grace turns are not adaptive conversation; exact match is not authenticity."}
        record(summary)
    return summary
