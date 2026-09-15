import json
import shutil
from copy import deepcopy

import pytest

from rocky.data import (
    DATA,
    TRANSCRIPT,
    authored_rows,
    expansion_rows,
    export,
    read_jsonl,
    source_turns,
    validate,
    verify_export,
    verify_transcription,
)


def test_all_translated_source_turns_accounted_for():
    report = validate()
    assert report["represented_source_turns"] + report["untranslated_source_turns"] == report["source_rocky_turns"]
    assert report["splits"]["train"]["film_turns"] > report["splits"]["train"]["original_turns"] * 3


def test_wrong_speaker_and_invented_movie_line_rejected():
    row = deepcopy(read_jsonl(DATA / "corpus/film.jsonl")[0])
    row["messages"][0]["content"] = "Certainly, Grace, happy to help."
    with pytest.raises(ValueError, match="differs from transcript"):
        verify_transcription(row, source_turns(TRANSCRIPT))
    row = {"id": "wrong-speaker", "messages": [{"role": "user", "content": "Rocky happy not alone."}]}
    with pytest.raises(ValueError, match="differs from transcript"):
        verify_transcription(row, source_turns(TRANSCRIPT))


def test_export_preserves_dialogue_and_detects_tampering(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    manifest = export(data)
    verify_export(data)
    assert manifest["model_evaluated"] is False
    exported = read_jsonl(data / "exports/train.jsonl")
    originals = [r for r in authored_rows(data) if r["split"] == "train"]
    assert len(originals) == len(exported)
    for authored, formatted in zip(originals, exported):
        assert set(formatted) == {"messages"}
        assert formatted["messages"][1:] == authored["messages"]
    path = data / "exports/train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="export changed"):
        verify_export(data)


def test_expansion_formats_pairs_without_changing_text():
    raw = read_jsonl(DATA / "corpus/expansion.jsonl")
    formatted = expansion_rows(DATA)
    assert len(raw) == len(formatted)
    for source, row in zip(raw, formatted):
        assert row["context"] == source["context"]
        assert [m["content"] for m in row["messages"]] == [text for pair in source["turns"] for text in pair]
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"] * len(source["turns"])


def test_expansion_rejects_unpaired_turn_and_heldout_scene(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = data / "corpus/expansion.jsonl"
    rows = read_jsonl(path)
    broken = deepcopy(rows)
    broken[0]["turns"][0].pop()
    path.write_text("".join(json.dumps(r) + "\n" for r in broken))
    with pytest.raises(ValueError, match="Grace/Rocky pair"):
        validate(data)
    rows[0]["scene"] = "farewell"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="scene crosses"):
        validate(data)


def test_curricula_share_holdouts_and_do_not_pad_training(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    manifest = export(data)
    ids = manifest["row_ids"]
    assert set(ids["core.train"]) < set(ids["train"])
    assert not set(ids["train"]) & (set(ids["test"]) | set(ids["validation"]))
    assert len(ids["train"]) == len(set(ids["train"]))
    assert manifest["totals"]["train"]["assistant_turns"] == sum(sum(m["role"] == "assistant" for m in r["messages"]) for r in read_jsonl(data / "exports/train.jsonl"))
    altered = deepcopy(manifest)
    altered["totals"]["train"]["assistant_turns"] += 1
    (data / "exports/manifest.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="inventory differs"):
        verify_export(data)
    del manifest["inputs"]["corpus/expansion.jsonl"]
    (data / "exports/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits required"):
        verify_export(data)


def test_scene_split_leak_rejected(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = data / "corpus/film.jsonl"
    rows = read_jsonl(path)
    next(row for row in rows if row["id"] == "film-voice")["split"] = "test"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="scene crosses"):
        validate(data)


def test_release_verifies_without_local_documentation(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    for path in data.rglob("*.md"):
        path.unlink()
    assert verify_export(data)["fact_cards"] > 0
