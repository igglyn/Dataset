import pathlib
import sys
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import distill_factory.pipeline.stage_a as stage_a_module


def _make_record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "doc_id": "doc-1",
        "chunk_index": 0,
        "byte_start": 0,
        "byte_end": 100,
        "raw_bytes": b"example chunk bytes",
        "split": "train",
        "top_k": 2,
        "enable_position_filtering": False,
        "selection_mode": "none",
        "entropy_threshold": None,
        "top1_gap_threshold": None,
        "selection_window_radius": 0,
        "extra_metadata": {},
    }
    base.update(overrides)
    return base


class _FakeTeacher:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self._outputs = outputs

    def prepare(self) -> None:
        return None

    def close(self) -> None:
        return None

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _ = records
        return self._outputs



def _patch_stage_a(monkeypatch, outputs: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(stage_a_module, "get_teacher", lambda _name: _FakeTeacher(outputs))
    monkeypatch.setattr(stage_a_module, "validate_teacher_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage_a_module, "log_stage_metrics", lambda _records, stage_name: {"stage": stage_name})



def test_stage_a_no_selection_keeps_dense_behavior(monkeypatch) -> None:
    output = {
        "top_k_ids": [[10, 11], [20, 21], [30, 31], [40, 41]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "entropy": 0.5,
        "per_token_entropy": [0.1, 0.2, 0.3, 0.4],
        "per_token_top1_gap": [0.5, 0.4, 0.3, 0.2],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(enable_position_filtering=False, selection_mode="selected_windows")
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 1
    assert out[0]["top_k_ids"] == output["top_k_ids"]
    assert out[0]["top_k_logprobs"] == output["top_k_logprobs"]
    assert "selected_window_start" not in out[0]["extra_metadata"]



def test_stage_a_position_mask_present_when_enabled(monkeypatch) -> None:
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "entropy": 0.4,
        "per_token_entropy": [0.1, 0.9, 0.1, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2, 0.9],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=True,
        selection_mode="position_mask",
        entropy_threshold=0.8,
        top1_gap_threshold=0.3,
        selection_window_radius=2,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 1
    meta = out[0]["extra_metadata"]
    assert "selected_position_mask" in meta
    assert "selected_positions" in meta
    assert meta["selected_positions"] == [1, 2]
    assert meta["selected_position_count"] == 2
    assert meta["selection_policy"]["selection_mode"] == "position_mask"



def test_stage_a_position_mask_length_aligns_with_positions(monkeypatch) -> None:
    output = {
        "top_k_ids": [[10, 11], [20, 21], [30, 31]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        "entropy": 0.5,
        "per_token_entropy": [0.2, 0.9, 0.4],
        "per_token_top1_gap": [0.8, 0.1, 0.7],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=True,
        selection_mode="position_mask",
        entropy_threshold=0.8,
        top1_gap_threshold=None,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 1
    mask = out[0]["extra_metadata"]["selected_position_mask"]
    assert len(mask) == len(out[0]["top_k_ids"])
    selected_positions = out[0]["extra_metadata"]["selected_positions"]
    assert selected_positions == [i for i, keep in enumerate(mask) if keep]



def test_stage_a_position_mask_dense_unchanged_when_disabled(monkeypatch) -> None:
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        "entropy": 0.4,
        "per_token_entropy": [0.1, 0.9, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=False,
        selection_mode="position_mask",
        entropy_threshold=0.8,
        top1_gap_threshold=0.3,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 1
    assert out[0]["top_k_ids"] == output["top_k_ids"]
    meta = out[0]["extra_metadata"]
    assert "selected_position_mask" not in meta
    assert "selected_positions" not in meta



def test_stage_a_entropy_threshold_selected_windows_reduces_export(monkeypatch) -> None:
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5], [-0.5, -0.6]],
        "entropy": 0.6,
        "per_token_entropy": [0.1, 0.9, 0.2, 0.8, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.9, 0.9, 0.9],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=True,
        selection_mode="selected_windows",
        entropy_threshold=0.75,
        top1_gap_threshold=None,
        selection_window_radius=0,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 2
    assert out[0]["top_k_ids"] == [[3, 4]]
    assert out[1]["top_k_ids"] == [[7, 8]]
    assert out[0]["extra_metadata"]["selected_window_start"] == 1
    assert out[0]["extra_metadata"]["selected_window_end"] == 1



def test_stage_a_gap_threshold_selected_windows_reduces_export(monkeypatch) -> None:
    output = {
        "top_k_ids": [[10, 11], [20, 21], [30, 31], [40, 41]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "entropy": 0.5,
        "per_token_entropy": [0.0, 0.0, 0.0, 0.0],
        "per_token_top1_gap": [0.7, 0.1, 0.8, 0.2],
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=True,
        selection_mode="selected_windows",
        entropy_threshold=None,
        top1_gap_threshold=0.2,
        selection_window_radius=0,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 2
    assert out[0]["top_k_ids"] == [[20, 21]]
    assert out[1]["top_k_ids"] == [[40, 41]]



def test_stage_a_combined_selection_defaults_to_union(monkeypatch) -> None:
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "entropy": 0.4,
        "per_token_entropy": [0.1, 0.9, 0.1, 0.1],  # selects position 1 for threshold 0.8
        "per_token_top1_gap": [0.9, 0.9, 0.2, 0.9],  # selects position 2 for threshold 0.3
    }
    _patch_stage_a(monkeypatch, [output])

    record = _make_record(
        enable_position_filtering=True,
        selection_mode="selected_windows",
        entropy_threshold=0.8,
        top1_gap_threshold=0.3,
        selection_window_radius=0,
    )
    out = stage_a_module.run_stage_a([record], teacher_name="dummy", mode="topk_logits", dry_run=False)

    assert len(out) == 2
    assert [r["extra_metadata"]["selected_window_start"] for r in out] == [1, 2]
    assert out[0]["extra_metadata"]["selection_policy"]["combine_mode"] == "union"


def test_stage_a_raises_when_teacher_output_count_mismatches_records(monkeypatch) -> None:
    output = {
        "top_k_ids": [[1, 2]],
        "top_k_logprobs": [[-0.1, -0.2]],
        "entropy": 0.5,
    }
    _patch_stage_a(monkeypatch, [output])

    record_a = _make_record(doc_id="doc-a")
    record_b = _make_record(doc_id="doc-b")

    with pytest.raises(RuntimeError, match="result count mismatch"):
        stage_a_module.run_stage_a([record_a, record_b], teacher_name="dummy", mode="topk_logits", dry_run=False)
