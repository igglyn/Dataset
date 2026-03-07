"""Dataset split helpers."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def split_records_by_doc_id(
    records: list[dict[str, Any]],
    eval_fraction: float = 0.1,
    seed: int = 0,
    method: str = "hash",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records deterministically by document id."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if method not in {"hash", "shuffle"}:
        raise ValueError("method must be 'hash' or 'shuffle'")

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        doc_id = str(record["doc_id"])
        by_doc.setdefault(doc_id, []).append(record)

    doc_ids = sorted(by_doc)
    eval_ids: set[str]

    if method == "hash":
        eval_ids = set()
        threshold = int(eval_fraction * 10_000)
        for doc_id in doc_ids:
            digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % 10_000
            if bucket < threshold:
                eval_ids.add(doc_id)
    else:
        rng = random.Random(seed)
        shuffled = list(doc_ids)
        rng.shuffle(shuffled)
        eval_count = int(round(len(shuffled) * eval_fraction))
        eval_ids = set(shuffled[:eval_count])

    train: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        target = eval_set if doc_id in eval_ids else train
        target.extend(by_doc[doc_id])

    return train, eval_set


def _doc_length_map(records: list[dict[str, Any]]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for record in records:
        doc_id = str(record.get("doc_id", ""))
        byte_end = int(record.get("byte_end", 0))
        prev = lengths.get(doc_id, 0)
        if byte_end > prev:
            lengths[doc_id] = byte_end
    return lengths


def split_records_with_longdoc_eval(
    records: list[dict[str, Any]],
    eval_fraction: float,
    eval_longdoc_fraction: float,
    eval_longdoc_min_bytes: int,
    eval_split_strategy: str,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train/eval/eval_longdoc using deterministic doc-level logic."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if not 0.0 <= eval_longdoc_fraction < 1.0:
        raise ValueError("eval_longdoc_fraction must be in [0, 1)")
    if eval_split_strategy not in {"random_docs", "prefer_long_docs"}:
        raise ValueError("eval_split_strategy must be 'random_docs' or 'prefer_long_docs'")

    train, eval_set = split_records_by_doc_id(
        records,
        eval_fraction=eval_fraction,
        seed=seed,
        method="shuffle",
    )

    if eval_longdoc_fraction <= 0.0:
        for rec in train:
            rec["split"] = "train"
        for rec in eval_set:
            rec["split"] = "eval"
        return train, eval_set, []

    by_doc_eval: dict[str, list[dict[str, Any]]] = {}
    for rec in eval_set:
        doc_id = str(rec["doc_id"])
        by_doc_eval.setdefault(doc_id, []).append(rec)

    eval_doc_ids = sorted(by_doc_eval)
    target_doc_count = int(round(len(eval_doc_ids) * eval_longdoc_fraction))
    if target_doc_count <= 0:
        for rec in train:
            rec["split"] = "train"
        for rec in eval_set:
            rec["split"] = "eval"
        return train, eval_set, []

    doc_len = _doc_length_map(eval_set)
    eligible = [d for d in eval_doc_ids if doc_len.get(d, 0) >= int(eval_longdoc_min_bytes)]

    if eval_split_strategy == "prefer_long_docs":
        ranked = sorted(eligible, key=lambda d: (doc_len.get(d, 0), d), reverse=True)
    else:
        ranked = sorted(eligible)
        rng = random.Random(seed + 909)
        rng.shuffle(ranked)

    selected_ids = set(ranked[: min(target_doc_count, len(ranked))])

    eval_regular: list[dict[str, Any]] = []
    eval_longdoc: list[dict[str, Any]] = []
    for doc_id in eval_doc_ids:
        target = eval_longdoc if doc_id in selected_ids else eval_regular
        target.extend(by_doc_eval[doc_id])

    for rec in train:
        rec["split"] = "train"
    for rec in eval_regular:
        rec["split"] = "eval"
    for rec in eval_longdoc:
        rec["split"] = "eval_longdoc"

    return train, eval_regular, eval_longdoc


def train_val_split(
    records: list[dict[str, Any]], train_ratio: float = 0.9, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backward-compatible split helper using shuffle-based doc-level split."""
    return split_records_by_doc_id(
        records,
        eval_fraction=1.0 - train_ratio,
        seed=seed,
        method="shuffle",
    )
