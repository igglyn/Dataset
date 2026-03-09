"""Schema and loaders for reusable corpus source caches and mixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python 3.10 fallback
    tomllib = None

if tomllib is None:
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:
        tomllib = None


@dataclass(slots=True)
class SourceDatasetCacheConfig:
    source_name: str
    source_type: str
    hf_dataset: str
    hf_config: str | None
    text_field: str
    split_mapping: dict[str, str]
    group_size: int
    max_docs_per_split: int | None
    min_bytes: int | None
    max_bytes: int | None


@dataclass(slots=True)
class SourceExtractionConfig:
    cache_dir: str
    datasets: list[SourceDatasetCacheConfig]


@dataclass(slots=True)
class MixtureDatasetConfig:
    source_name: str


@dataclass(slots=True)
class MixtureGroupConfig:
    group_name: str
    percentage: float
    datasets: list[MixtureDatasetConfig]
    dataset_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MixtureBuildConfig:
    target_documents: int
    random_seed: int
    groups: list[MixtureGroupConfig]
    min_bytes: int | None
    max_bytes: int | None
    depletion_policy: str


@dataclass(slots=True)
class CorpusMixtureConfig:
    source_extraction: SourceExtractionConfig
    mixture_build: MixtureBuildConfig


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid table: [{key}]")
    return value


def _parse_split_mapping(dataset: dict[str, Any], source_name: str) -> dict[str, str]:
    split_mapping = dataset.get("split_mapping")
    if split_mapping is not None:
        if not isinstance(split_mapping, dict):
            raise ValueError(f"Dataset '{source_name}' split_mapping must be a table")
        parsed = {str(k): str(v) for k, v in split_mapping.items()}
    else:
        parsed = {}

    for canonical, key in [
        ("train", "train_split"),
        ("eval", "eval_split"),
        ("validation", "validation_split"),
    ]:
        explicit = dataset.get(key)
        if explicit is not None:
            parsed[canonical] = str(explicit)

    if not parsed:
        raise ValueError(
            f"Dataset '{source_name}' must set split_mapping or at least one of train_split/eval_split/validation_split"
        )
    return parsed


def _parse_source_dataset(dataset: dict[str, Any]) -> SourceDatasetCacheConfig:
    source_name = str(dataset.get("source_name", "")).strip()
    if not source_name:
        raise ValueError("Every source_extraction.datasets entry requires source_name")

    source_type = str(dataset.get("source_type", "")).strip()
    if source_type != "huggingface":
        raise ValueError(f"Dataset '{source_name}' source_type must be 'huggingface'")

    hf_dataset = str(dataset.get("hf_dataset", "")).strip()
    if not hf_dataset:
        raise ValueError(f"Dataset '{source_name}' requires hf_dataset")

    text_field = str(dataset.get("text_field", "")).strip()
    if not text_field:
        raise ValueError(f"Dataset '{source_name}' requires text_field")

    group_size = int(dataset.get("group_size", 0))
    if group_size <= 0:
        raise ValueError(f"Dataset '{source_name}' group_size must be > 0")

    max_docs_raw = dataset.get("max_docs_per_split")
    max_docs_per_split = None if max_docs_raw is None else int(max_docs_raw)
    if max_docs_per_split is not None and max_docs_per_split <= 0:
        raise ValueError(f"Dataset '{source_name}' max_docs_per_split must be > 0 when set")

    min_bytes_raw = dataset.get("min_bytes")
    min_bytes = None if min_bytes_raw is None else int(min_bytes_raw)
    if min_bytes is not None and min_bytes < 0:
        raise ValueError(f"Dataset '{source_name}' min_bytes must be >= 0 when set")

    max_bytes_raw = dataset.get("max_bytes")
    max_bytes = None if max_bytes_raw is None else int(max_bytes_raw)
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError(f"Dataset '{source_name}' max_bytes must be > 0 when set")

    if min_bytes is not None and max_bytes is not None and min_bytes > max_bytes:
        raise ValueError(f"Dataset '{source_name}' min_bytes must be <= max_bytes when both are set")

    split_mapping = _parse_split_mapping(dataset, source_name=source_name)

    hf_config_raw = dataset.get("hf_config")
    hf_config = None if hf_config_raw is None else str(hf_config_raw)

    return SourceDatasetCacheConfig(
        source_name=source_name,
        source_type=source_type,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        text_field=text_field,
        split_mapping=split_mapping,
        group_size=group_size,
        max_docs_per_split=max_docs_per_split,
        min_bytes=min_bytes,
        max_bytes=max_bytes,
    )


def _parse_group(group: dict[str, Any]) -> MixtureGroupConfig:
    group_name = str(group.get("group_name", "")).strip()
    if not group_name:
        raise ValueError("Each mixture_build.groups entry requires group_name")

    percentage = float(group.get("percentage", 0.0))
    if percentage <= 0.0:
        raise ValueError(f"Group '{group_name}' percentage must be > 0")

    raw_dataset_names = group.get("dataset_names")
    dataset_names: list[str] = []

    if raw_dataset_names is not None:
        if not isinstance(raw_dataset_names, list) or not raw_dataset_names:
            raise ValueError(f"Group '{group_name}' dataset_names must be a non-empty list")
        for raw_name in raw_dataset_names:
            source_name = str(raw_name).strip()
            if not source_name:
                raise ValueError(f"Group '{group_name}' dataset_names must not contain empty names")
            dataset_names.append(source_name)
    else:
        # Backward-compatibility path for older table-form syntax.
        raw_datasets = group.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError(
                f"Group '{group_name}' must provide dataset_names (preferred) or legacy datasets list"
            )
        for dataset in raw_datasets:
            if not isinstance(dataset, dict):
                raise ValueError(f"Group '{group_name}' datasets entries must be tables")
            source_name = str(dataset.get("source_name", "")).strip()
            if not source_name:
                raise ValueError(f"Group '{group_name}' has dataset missing source_name")
            dataset_names.append(source_name)

    datasets: list[MixtureDatasetConfig] = [
        MixtureDatasetConfig(source_name=name) for name in dataset_names
    ]

    return MixtureGroupConfig(
        group_name=group_name,
        percentage=percentage,
        datasets=datasets,
        dataset_names=dataset_names,
    )


def _validate_percentages(groups: list[MixtureGroupConfig]) -> None:
    total = sum(group.percentage for group in groups)
    if abs(total - 100.0) > 1e-9:
        raise ValueError(f"mixture_build.groups percentages must sum to 100, got {total}")


def load_corpus_mixture_config(path: str | Path) -> CorpusMixtureConfig:
    """Load corpus cache + mixture TOML into explicit dataclasses."""
    if tomllib is None:
        raise RuntimeError("tomllib (Python 3.11+) or tomli is required for corpus mixture TOML parsing")

    config_path = Path(path)
    with config_path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    source_extraction = _require_table(data, "source_extraction")
    cache_dir = str(source_extraction.get("cache_dir", "")).strip()
    if not cache_dir:
        raise ValueError("[source_extraction].cache_dir is required")

    raw_sources = source_extraction.get("datasets")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("[source_extraction].datasets must be a non-empty list")

    datasets = [_parse_source_dataset(item) for item in raw_sources]

    mixture_build = _require_table(data, "mixture_build")
    if "target_documents" not in mixture_build:
        raise ValueError("[mixture_build].target_documents is required")
    target_documents = int(mixture_build["target_documents"])
    if target_documents <= 0:
        raise ValueError("[mixture_build].target_documents must be > 0")

    random_seed = int(mixture_build.get("random_seed", 0))

    mixture_min_raw = mixture_build.get("min_bytes")
    mixture_min_bytes = None if mixture_min_raw is None else int(mixture_min_raw)
    if mixture_min_bytes is not None and mixture_min_bytes < 0:
        raise ValueError("[mixture_build].min_bytes must be >= 0 when set")

    mixture_max_raw = mixture_build.get("max_bytes")
    mixture_max_bytes = None if mixture_max_raw is None else int(mixture_max_raw)
    if mixture_max_bytes is not None and mixture_max_bytes <= 0:
        raise ValueError("[mixture_build].max_bytes must be > 0 when set")

    if mixture_min_bytes is not None and mixture_max_bytes is not None and mixture_min_bytes > mixture_max_bytes:
        raise ValueError("[mixture_build].min_bytes must be <= [mixture_build].max_bytes when both are set")

    depletion_policy = str(mixture_build.get("depletion_policy", "rebalance")).strip()
    if depletion_policy not in {"rebalance", "strict", "record_only"}:
        raise ValueError("[mixture_build].depletion_policy must be one of: rebalance, strict, record_only")

    raw_groups = mixture_build.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("[mixture_build].groups must be a non-empty list")

    groups = [_parse_group(group) for group in raw_groups]
    _validate_percentages(groups)

    source_names = {dataset.source_name for dataset in datasets}
    for group in groups:
        for dataset in group.datasets:
            if dataset.source_name not in source_names:
                raise ValueError(
                    f"Group '{group.group_name}' references unknown source_name '{dataset.source_name}'"
                )

    return CorpusMixtureConfig(
        source_extraction=SourceExtractionConfig(cache_dir=cache_dir, datasets=datasets),
        mixture_build=MixtureBuildConfig(
            target_documents=target_documents,
            random_seed=random_seed,
            groups=groups,
            min_bytes=mixture_min_bytes,
            max_bytes=mixture_max_bytes,
            depletion_policy=depletion_policy,
        ),
    )
