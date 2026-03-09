from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_factory.corpus.mix import build_corpus_mixture_from_cache
from distill_factory.corpus.schema import (
    load_corpus_mixture_config,
    CorpusMixtureConfig,
    MixtureBuildConfig,
    MixtureDatasetConfig,
    MixtureGroupConfig,
    SourceDatasetCacheConfig,
    SourceExtractionConfig,
)


def _write_doc(split_dir: Path, idx: int, text: str) -> None:
    stem = f"doc_{idx:08d}"
    (split_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (split_dir / f"{stem}.meta.json").write_text("{}\n", encoding="utf-8")


def _make_source(name: str) -> SourceDatasetCacheConfig:
    return SourceDatasetCacheConfig(
        source_name=name,
        source_type="huggingface",
        hf_dataset=f"owner/{name}",
        hf_config="default",
        text_field="text",
        split_mapping={"train": "train", "eval": "eval", "validation": "validation"},
        group_size=1,
        max_docs_per_split=None,
        min_bytes=None,
        max_bytes=None,
    )


def _build_config() -> CorpusMixtureConfig:
    return CorpusMixtureConfig(
        source_extraction=SourceExtractionConfig(
            cache_dir="data/sources",
            datasets=[_make_source("code_a"), _make_source("gk_a")],
        ),
        mixture_build=MixtureBuildConfig(
            target_documents=6,
            random_seed=123,
            groups=[
                MixtureGroupConfig(
                    group_name="code",
                    percentage=50.0,
                    datasets=[MixtureDatasetConfig(source_name="code_a")],
                ),
                MixtureGroupConfig(
                    group_name="general_knowledge",
                    percentage=50.0,
                    datasets=[MixtureDatasetConfig(source_name="gk_a")],
                ),
            ],
            min_bytes=None,
            max_bytes=None,
            depletion_policy="rebalance",
        ),
    )


def _materialize_cached_sources(cache_root: Path) -> None:
    for source_name, prefix in [("code_a", "code"), ("gk_a", "gk")]:
        for split in ["train", "eval", "validation"]:
            split_dir = cache_root / source_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(1, 10):
                _write_doc(split_dir, idx, f"{prefix}-{split}-{idx}\n")


def _read_meta_source_paths(split_dir: Path) -> list[str]:
    out: list[str] = []
    for meta_path in sorted(split_dir.glob("doc_*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append(meta["source_doc_path"])
    return out


def test_build_mixed_corpus_layout_and_manifest(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"
    _materialize_cached_sources(cache_root)
    cfg = _build_config()

    out_dir = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_v1",
        cache_root=cache_root,
        output_root=output_root,
    )

    assert (out_dir / "train").is_dir()
    assert (out_dir / "eval").is_dir()
    assert (out_dir / "validation").is_dir()
    assert (out_dir / "manifest.json").is_file()

    for split in ["train", "eval", "validation"]:
        assert len(list((out_dir / split).glob("doc_*.txt"))) == 6
        meta = json.loads((out_dir / split / "doc_00000001.meta.json").read_text(encoding="utf-8"))
        assert "source_name" in meta
        assert "source_split" in meta
        assert "group_name" in meta
        assert "source_doc_path" in meta
        assert "source_doc_id" in meta

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mixture_name"] == "mix_v1"
    assert manifest["target_documents_per_split"] == 6


def test_sampling_is_deterministic_from_seed(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"
    _materialize_cached_sources(cache_root)
    cfg = _build_config()

    out1 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_seed_a",
        cache_root=cache_root,
        output_root=output_root,
    )
    out2 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_seed_b",
        cache_root=cache_root,
        output_root=output_root,
    )

    assert _read_meta_source_paths(out1 / "train") == _read_meta_source_paths(out2 / "train")
    assert _read_meta_source_paths(out1 / "eval") == _read_meta_source_paths(out2 / "eval")
    assert _read_meta_source_paths(out1 / "validation") == _read_meta_source_paths(out2 / "validation")


def test_mixture_percentage_validation(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"
    _materialize_cached_sources(cache_root)

    cfg = _build_config()
    cfg.mixture_build.groups[0].percentage = 70.0
    cfg.mixture_build.groups[1].percentage = 20.0

    with pytest.raises(ValueError, match="sum to 100"):
        build_corpus_mixture_from_cache(
            cfg,
            mixture_name="bad_mix",
            cache_root=cache_root,
            output_root=output_root,
        )


def test_config_loader_accepts_dataset_names_and_rejects_unknown_dataset(tmp_path):
    cfg_path = tmp_path / "mix.toml"
    cfg_path.write_text(
        """
[source_extraction]
cache_dir = "data/sources"

[[source_extraction.datasets]]
source_name = "code_a"
source_type = "huggingface"
hf_dataset = "owner/code_a"
text_field = "text"
train_split = "train"
group_size = 1

[mixture_build]
target_documents = 10
random_seed = 7

[[mixture_build.groups]]
group_name = "code"
percentage = 100
dataset_names = ["missing_source"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="references unknown source_name"):
        load_corpus_mixture_config(cfg_path)


def test_config_loader_rejects_dataset_names_percentage_total_mismatch(tmp_path):
    cfg_path = tmp_path / "mix_bad_pct.toml"
    cfg_path.write_text(
        """
[source_extraction]
cache_dir = "data/sources"

[[source_extraction.datasets]]
source_name = "code_a"
source_type = "huggingface"
hf_dataset = "owner/code_a"
text_field = "text"
train_split = "train"
group_size = 1

[mixture_build]
target_documents = 10
random_seed = 7

[[mixture_build.groups]]
group_name = "code"
percentage = 60
dataset_names = ["code_a"]

[[mixture_build.groups]]
group_name = "general_knowledge"
percentage = 30
dataset_names = ["code_a"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 100"):
        load_corpus_mixture_config(cfg_path)


def test_build_fails_when_cached_split_missing(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"
    _materialize_cached_sources(cache_root)
    # Remove one required split to ensure builder fails fast on split availability.
    missing_split = cache_root / "code_a" / "validation"
    for child in missing_split.glob("*"):
        child.unlink()
    missing_split.rmdir()

    cfg = _build_config()
    with pytest.raises(ValueError, match="Missing cached split directory"):
        build_corpus_mixture_from_cache(
            cfg,
            mixture_name="mix_missing_split",
            cache_root=cache_root,
            output_root=output_root,
        )


def test_depletion_rebalances_deterministically_across_groups(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"

    # code group is small (2 docs/split), general group is larger (10 docs/split)
    for split in ["train", "eval", "validation"]:
        code_split = cache_root / "code_a" / split
        gk_split = cache_root / "gk_a" / split
        code_split.mkdir(parents=True, exist_ok=True)
        gk_split.mkdir(parents=True, exist_ok=True)
        for idx in range(1, 3):
            _write_doc(code_split, idx, f"code-{split}-{idx}\n")
        for idx in range(1, 11):
            _write_doc(gk_split, idx, f"gk-{split}-{idx}\n")

    cfg = _build_config()
    # target per split = 6, requested 50/50 => 3 code + 3 gk; code can only provide 2
    out_dir = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_rebalanced",
        cache_root=cache_root,
        output_root=output_root,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    # rebalance should still hit per-split target because other group has spare capacity
    assert manifest["composition_deviation"] is False
    assert manifest["realized_split_counts"] == {"train": 6, "eval": 6, "validation": 6}
    # across 3 splits: code 2*3=6, gk 4*3=12 after rebalance
    assert manifest["realized_document_counts_by_group"]["code"] == 6
    assert manifest["realized_document_counts_by_group"]["general_knowledge"] == 12

    # deterministic repeated build with same config + seed + cache
    out_dir_2 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_rebalanced_repeat",
        cache_root=cache_root,
        output_root=output_root,
    )
    assert _read_meta_source_paths(out_dir / "train") == _read_meta_source_paths(out_dir_2 / "train")


def test_depletion_records_realized_shortfall_when_total_cache_insufficient(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"

    # Total docs per split is 4 but target is 6 -> unavoidable shortfall.
    for source_name, prefix in [("code_a", "code"), ("gk_a", "gk")]:
        for split in ["train", "eval", "validation"]:
            split_dir = cache_root / source_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(1, 3):
                _write_doc(split_dir, idx, f"{prefix}-{split}-{idx}\n")

    cfg = _build_config()
    out_dir = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_shortfall",
        cache_root=cache_root,
        output_root=output_root,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["composition_deviation"] is True
    assert manifest["requested_total_documents"] == 18
    assert manifest["realized_total_documents"] == 12
    assert manifest["depletion_policy"] == "rebalance"
    assert manifest["warnings"]
    for split_summary in manifest["splits"]:
        assert split_summary["deviation_from_target"] is True
        assert split_summary["output_documents"] == 4


def test_mixture_applies_cached_doc_byte_filters_deterministically(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"

    # Mix of tiny/ok/giant docs. Keep only 2..4 bytes.
    for split in ["train", "eval", "validation"]:
        code_split = cache_root / "code_a" / split
        gk_split = cache_root / "gk_a" / split
        code_split.mkdir(parents=True, exist_ok=True)
        gk_split.mkdir(parents=True, exist_ok=True)
        _write_doc(code_split, 1, "a")      # 1 byte filtered below
        _write_doc(code_split, 2, "bb")     # keep
        _write_doc(code_split, 3, "ccc")    # keep
        _write_doc(code_split, 4, "ddddd")  # 5 bytes filtered above

        _write_doc(gk_split, 1, "e")        # below
        _write_doc(gk_split, 2, "ff")       # keep
        _write_doc(gk_split, 3, "ggg")      # keep
        _write_doc(gk_split, 4, "hhhhh")    # above

    cfg = _build_config()
    cfg.mixture_build.target_documents = 4
    cfg.mixture_build.min_bytes = 2
    cfg.mixture_build.max_bytes = 4

    out1 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_len_filtered_a",
        cache_root=cache_root,
        output_root=output_root,
    )
    out2 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_len_filtered_b",
        cache_root=cache_root,
        output_root=output_root,
    )

    assert _read_meta_source_paths(out1 / "train") == _read_meta_source_paths(out2 / "train")

    manifest = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mixture_min_bytes"] == 2
    assert manifest["mixture_max_bytes"] == 4
    assert manifest["realized_split_counts"] == {"train": 4, "eval": 4, "validation": 4}
    assert any("mixture byte filtering excluded docs" in w for w in manifest["warnings"])


def test_strict_policy_fails_when_group_cannot_meet_requested_share(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"

    # code has only 2 docs/split, requested share needs 3 docs/split.
    for split in ["train", "eval", "validation"]:
        code_split = cache_root / "code_a" / split
        gk_split = cache_root / "gk_a" / split
        code_split.mkdir(parents=True, exist_ok=True)
        gk_split.mkdir(parents=True, exist_ok=True)
        for idx in range(1, 3):
            _write_doc(code_split, idx, f"code-{split}-{idx}\n")
        for idx in range(1, 10):
            _write_doc(gk_split, idx, f"gk-{split}-{idx}\n")

    cfg = _build_config()
    cfg.mixture_build.depletion_policy = "strict"

    with pytest.raises(ValueError, match="strict depletion_policy"):
        build_corpus_mixture_from_cache(
            cfg,
            mixture_name="mix_strict_fail",
            cache_root=cache_root,
            output_root=output_root,
        )


def test_record_only_policy_keeps_shortfall_without_rebalance(tmp_path):
    cache_root = tmp_path / "data" / "sources"
    output_root = tmp_path / "data" / "corpora"

    # code has only 2 docs/split; gk has many. record_only should not fill missing code quota from gk.
    for split in ["train", "eval", "validation"]:
        code_split = cache_root / "code_a" / split
        gk_split = cache_root / "gk_a" / split
        code_split.mkdir(parents=True, exist_ok=True)
        gk_split.mkdir(parents=True, exist_ok=True)
        for idx in range(1, 3):
            _write_doc(code_split, idx, f"code-{split}-{idx}\n")
        for idx in range(1, 10):
            _write_doc(gk_split, idx, f"gk-{split}-{idx}\n")

    cfg = _build_config()
    cfg.mixture_build.depletion_policy = "record_only"

    out1 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_record_only_a",
        cache_root=cache_root,
        output_root=output_root,
    )
    out2 = build_corpus_mixture_from_cache(
        cfg,
        mixture_name="mix_record_only_b",
        cache_root=cache_root,
        output_root=output_root,
    )

    manifest = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["depletion_policy"] == "record_only"
    # 2 code + 3 gk per split => 5 docs/split (no rebalance)
    assert manifest["realized_split_counts"] == {"train": 5, "eval": 5, "validation": 5}
    assert manifest["composition_deviation"] is True
    assert _read_meta_source_paths(out1 / "train") == _read_meta_source_paths(out2 / "train")
