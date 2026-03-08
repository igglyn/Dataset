import pathlib
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import distill_factory.pipeline.stage_a as stage_a_module
import scripts.preview_selection_policy as preview_module


def _cfg(*, selection_mode: str, entropy_threshold: float | None, top1_gap_threshold: float | None, radius: int) -> Any:
    return SimpleNamespace(
        stage_a=SimpleNamespace(
            enable_position_filtering=True,
            selection_mode=selection_mode,
            entropy_threshold=entropy_threshold,
            top1_gap_threshold=top1_gap_threshold,
            selection_window_radius=radius,
            minimum_selected_positions_per_record=None,
        )
    )


def _base_stage_record() -> dict[str, Any]:
    return {
        "doc_id": "doc-1",
        "chunk_index": 0,
        "byte_start": 0,
        "byte_end": 4,
        "raw_bytes": b"abcd",
        "split": "train",
        "extra_metadata": {},
    }


def test_preview_position_mask_matches_stage_a_selection_metadata() -> None:
    cfg = _cfg(selection_mode="position_mask", entropy_threshold=0.8, top1_gap_threshold=0.3, radius=1)
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "per_token_entropy": [0.1, 0.9, 0.1, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2, 0.9],
    }

    selection_record = preview_module._stage_a_selection_record(cfg, output)
    preview_artifacts = stage_a_module.selection_artifacts_for_record(selection_record)

    stage_record = _base_stage_record()
    stage_record.update(selection_record)
    stage_out = stage_a_module._apply_position_aware_export([stage_record])

    assert len(stage_out) == 1
    assert stage_out[0]["extra_metadata"]["selected_positions"] == preview_artifacts["selected_positions"]
    assert stage_out[0]["extra_metadata"]["selection_policy"]["combine_mode"] == "union"


def test_preview_selected_windows_match_stage_a_window_boundaries() -> None:
    cfg = _cfg(selection_mode="selected_windows", entropy_threshold=0.75, top1_gap_threshold=None, radius=0)
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5], [-0.5, -0.6]],
        "per_token_entropy": [0.1, 0.9, 0.2, 0.8, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.9, 0.9, 0.9],
    }

    selection_record = preview_module._stage_a_selection_record(cfg, output)
    preview_artifacts = stage_a_module.selection_artifacts_for_record(selection_record)

    stage_record = _base_stage_record()
    stage_record.update(selection_record)
    stage_out = stage_a_module._apply_position_aware_export([stage_record])

    stage_windows = [
        (int(r["extra_metadata"]["selected_window_start"]), int(r["extra_metadata"]["selected_window_end"]))
        for r in stage_out
    ]
    assert stage_windows == preview_artifacts["windows"]


def test_preview_and_stage_a_share_combine_mode_threshold_handling() -> None:
    cfg = _cfg(selection_mode="position_mask", entropy_threshold=0.8, top1_gap_threshold=0.3, radius=0)
    output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "per_token_entropy": [0.1, 0.9, 0.1, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2, 0.9],
    }

    selection_record = preview_module._stage_a_selection_record(cfg, output)
    selection_record["selection_combine_mode"] = "intersection"
    preview_artifacts = stage_a_module.selection_artifacts_for_record(selection_record)

    stage_record = _base_stage_record()
    stage_record.update(selection_record)
    stage_out = stage_a_module._apply_position_aware_export([stage_record])

    assert len(stage_out) == 1
    assert stage_out[0]["extra_metadata"]["selected_positions"] == preview_artifacts["selected_positions"]
    assert stage_out[0]["extra_metadata"]["selection_policy"]["combine_mode"] == "intersection"
