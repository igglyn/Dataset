from __future__ import annotations

import json

import pytest

from distill_factory.corpus.extract import extract_source_to_cache
from distill_factory.corpus.schema import SourceDatasetCacheConfig


class FakeStreamingDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        yield from self._rows


def _make_source() -> SourceDatasetCacheConfig:
    return SourceDatasetCacheConfig(
        source_name="toy_source",
        source_type="huggingface",
        hf_dataset="owner/toy",
        hf_config="default",
        text_field="text",
        split_mapping={"train": "train", "eval": "validation", "validation": "test"},
        group_size=2,
        max_docs_per_split=None,
        min_bytes=None,
        max_bytes=None,
    )


def test_extract_layout_and_sidecar_metadata(tmp_path, monkeypatch):
    rows = {
        "train": [{"text": "a1"}, {"text": "a2"}, {"text": "a3"}, {"text": "a4"}],
        "validation": [{"text": "e1"}, {"text": "e2"}],
        "test": [{"text": "v1"}, {"text": "v2"}],
    }

    def fake_loader(source, split_name):
        return FakeStreamingDataset(rows[split_name])

    monkeypatch.setattr("distill_factory.corpus.extract._load_hf_streaming_split", fake_loader)

    out_dir = extract_source_to_cache(_make_source(), cache_root=tmp_path / "data" / "sources")

    assert (out_dir / "train").is_dir()
    assert (out_dir / "eval").is_dir()
    assert (out_dir / "validation").is_dir()
    assert (out_dir / "manifest.json").is_file()

    assert (out_dir / "train" / "doc_00000001.txt").read_text(encoding="utf-8") == "a1\n\na2\n"
    meta = json.loads((out_dir / "train" / "doc_00000001.meta.json").read_text(encoding="utf-8"))
    assert meta["source_name"] == "toy_source"
    assert meta["split"] == "train"
    assert meta["upstream_split"] == "train"
    assert meta["row_ordinal_start"] == 0
    assert meta["row_ordinal_end"] == 1
    assert meta["text_field"] == "text"
    assert meta["group_size"] == 2
    assert "extraction_timestamp" in meta

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_name"] == "toy_source"
    assert {entry["canonical_split"] for entry in manifest["splits"]} == {"train", "eval", "validation"}


def test_extract_resume_skips_existing_docs(tmp_path, monkeypatch):
    source = _make_source()
    rows = {
        "train": [{"text": "a1"}, {"text": "a2"}, {"text": "a3"}, {"text": "a4"}],
        "validation": [{"text": "e1"}, {"text": "e2"}],
        "test": [{"text": "v1"}, {"text": "v2"}],
    }

    def fake_loader(source, split_name):
        return FakeStreamingDataset(rows[split_name])

    monkeypatch.setattr("distill_factory.corpus.extract._load_hf_streaming_split", fake_loader)
    out_root = tmp_path / "data" / "sources"

    out_dir = extract_source_to_cache(source, cache_root=out_root)
    # second run should not duplicate docs
    out_dir = extract_source_to_cache(source, cache_root=out_root)

    train_docs = sorted(p.name for p in (out_dir / "train").glob("doc_*.txt"))
    assert train_docs == ["doc_00000001.txt", "doc_00000002.txt"]


def test_extract_fails_clearly_when_requested_split_unavailable(tmp_path, monkeypatch):
    source = _make_source()

    def fake_loader(source, split_name):
        if split_name == "test":
            raise ValueError("upstream split missing")
        return FakeStreamingDataset([{"text": "x1"}, {"text": "x2"}])

    monkeypatch.setattr("distill_factory.corpus.extract._load_hf_streaming_split", fake_loader)

    with pytest.raises(ValueError, match="upstream split missing"):
        extract_source_to_cache(source, cache_root=tmp_path / "data" / "sources")


def test_extract_applies_source_byte_filters_before_grouping(tmp_path, monkeypatch):
    source = _make_source()
    source.min_bytes = 2
    source.max_bytes = 3

    rows = {
        "train": [
            {"text": "a"},       # 1 byte -> filtered below
            {"text": "bb"},      # keep
            {"text": "ccc"},     # keep
            {"text": "dddd"},    # 4 bytes -> filtered above
            {"text": "ee"},      # keep
            {"text": "fff"},     # keep
        ],
        "validation": [{"text": "gg"}, {"text": "hhh"}],
        "test": [{"text": "ii"}, {"text": "jjj"}],
    }

    def fake_loader(source, split_name):
        return FakeStreamingDataset(rows[split_name])

    monkeypatch.setattr("distill_factory.corpus.extract._load_hf_streaming_split", fake_loader)

    out_dir = extract_source_to_cache(source, cache_root=tmp_path / "data" / "sources")

    # train accepted rows are [bb, ccc, ee, fff] -> two grouped docs
    train_docs = sorted((out_dir / "train").glob("doc_*.txt"))
    assert len(train_docs) == 2
    assert train_docs[0].read_text(encoding="utf-8") == "bb\n\nccc\n"
    assert train_docs[1].read_text(encoding="utf-8") == "ee\n\nfff\n"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    train_split = next(s for s in manifest["splits"] if s["canonical_split"] == "train")
    assert train_split["filtered_below_min_bytes"] == 1
    assert train_split["filtered_above_max_bytes"] == 1
    assert train_split["min_bytes"] == 2
    assert train_split["max_bytes"] == 3


def test_extract_enforces_max_docs_cap_without_overshoot(tmp_path, monkeypatch):
    source = _make_source()
    source.max_docs_per_split = 1

    rows = {
        "train": [
            {"text": "a1"},
            {"text": "a2"},
            {"text": "a3"},
            {"text": "a4"},
            {"text": "a5"},
            {"text": "a6"},
        ],
        "validation": [{"text": "e1"}, {"text": "e2"}],
        "test": [{"text": "v1"}, {"text": "v2"}],
    }

    def fake_loader(source, split_name):
        return FakeStreamingDataset(rows[split_name])

    monkeypatch.setattr("distill_factory.corpus.extract._load_hf_streaming_split", fake_loader)

    out_dir = extract_source_to_cache(source, cache_root=tmp_path / "data" / "sources")

    train_docs = sorted((out_dir / "train").glob("doc_*.txt"))
    assert len(train_docs) == 1
    assert train_docs[0].read_text(encoding="utf-8") == "a1\n\na2\n"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    train_split = next(s for s in manifest["splits"] if s["canonical_split"] == "train")
    assert train_split["docs_total"] == 1
    assert train_split["docs_written"] == 1


def test_cache_fingerprint_changes_when_min_bytes_changes(tmp_path):
    base = _make_source()
    modified = _make_source()
    modified.min_bytes = 5

    out_root = tmp_path / "data" / "sources"
    out_dir = out_root / base.source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"config_fingerprint": "stale", "source_config": {}}
    from distill_factory.corpus.manifest import config_fingerprint

    manifest["config_fingerprint"] = config_fingerprint(base)
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from distill_factory.corpus.extract import inspect_source_cache_state

    state = inspect_source_cache_state(source=modified, cache_root=out_root)
    assert state["state"] == "mismatch"
    assert state["existing_config_fingerprint"] != state["requested_config_fingerprint"]


def test_cache_fingerprint_changes_when_max_bytes_changes(tmp_path):
    base = _make_source()
    modified = _make_source()
    modified.max_bytes = 64

    out_root = tmp_path / "data" / "sources"
    out_dir = out_root / base.source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"config_fingerprint": "stale", "source_config": {}}
    from distill_factory.corpus.manifest import config_fingerprint

    manifest["config_fingerprint"] = config_fingerprint(base)
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from distill_factory.corpus.extract import inspect_source_cache_state

    state = inspect_source_cache_state(source=modified, cache_root=out_root)
    assert state["state"] == "mismatch"
    assert state["existing_config_fingerprint"] != state["requested_config_fingerprint"]
