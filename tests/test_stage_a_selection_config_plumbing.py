import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill_factory.pipeline import orchestrator as orch
import distill_factory.pipeline.stage_a as stage_a_module


@dataclass
class _StageAConfigStub:
    top_k: int
    extract_hidden_summary: bool
    enable_position_filtering: bool
    entropy_threshold: float | None
    top1_gap_threshold: float | None
    selection_window_radius: int
    selection_mode: str
    minimum_selected_positions_per_record: int | None

    def record_level_settings(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "extract_hidden_summary": self.extract_hidden_summary,
            "enable_position_filtering": self.enable_position_filtering,
            "entropy_threshold": self.entropy_threshold,
            "top1_gap_threshold": self.top1_gap_threshold,
            "selection_window_radius": self.selection_window_radius,
            "selection_mode": self.selection_mode,
            "minimum_selected_positions_per_record": self.minimum_selected_positions_per_record,
        }


class _CaptureTeacher:
    def __init__(self, output: dict[str, Any], captured_records: list[dict[str, Any]]) -> None:
        self._output = output
        self._captured_records = captured_records

    def prepare(self) -> None:
        return None

    def close(self) -> None:
        return None

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._captured_records.extend(dict(r) for r in records)
        return [self._output for _ in records]


def _cfg_for_stage_a_selection(
    *,
    enable_position_filtering: bool,
    selection_mode: str,
    entropy_threshold: float | None,
    top1_gap_threshold: float | None,
    selection_window_radius: int,
    minimum_selected_positions_per_record: int | None,
) -> Any:
    stage_a = _StageAConfigStub(
        top_k=2,
        extract_hidden_summary=False,
        enable_position_filtering=enable_position_filtering,
        entropy_threshold=entropy_threshold,
        top1_gap_threshold=top1_gap_threshold,
        selection_window_radius=selection_window_radius,
        selection_mode=selection_mode,
        minimum_selected_positions_per_record=minimum_selected_positions_per_record,
    )
    stage_b = SimpleNamespace(
        context_window=16,
        stride=1,
        max_teacher_context=16,
        window_policy="center_target",
        target_region_policy="preserve_full",
    )
    stage_c = SimpleNamespace(template_name="summarize_chunk", template_kwargs={}, deterministic=True)
    return SimpleNamespace(stage_a=stage_a, stage_b=stage_b, stage_c=stage_c)


def _base_record() -> dict[str, Any]:
    return {
        "doc_id": "doc-1",
        "chunk_index": 0,
        "byte_start": 0,
        "byte_end": 8,
        "raw_bytes": b"abcdefgh",
        "split": "train",
        "extra_metadata": {},
    }


def _run_orchestrated_stage_a(monkeypatch, cfg: Any, teacher_output: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captured_records: list[dict[str, Any]] = []
    teacher = _CaptureTeacher(teacher_output, captured_records)
    monkeypatch.setattr(stage_a_module, "get_teacher", lambda _name: teacher)
    monkeypatch.setattr(stage_a_module, "validate_teacher_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage_a_module, "log_stage_metrics", lambda _records, stage_name: {"stage": stage_name})

    out = orch._apply_stage_mixture(
        records=[_base_record()],
        stage_name="stage_a",
        mode="topk_logits",
        mixture=[SimpleNamespace(teacher_name="dummy", ratio=1.0)],
        cfg=cfg,
        seed=7,
        dry_run=False,
    )
    return out, captured_records


def test_orchestrated_stage_a_selection_disabled_stays_dense(monkeypatch) -> None:
    cfg = _cfg_for_stage_a_selection(
        enable_position_filtering=False,
        selection_mode="selected_windows",
        entropy_threshold=0.8,
        top1_gap_threshold=0.3,
        selection_window_radius=0,
        minimum_selected_positions_per_record=None,
    )
    teacher_output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        "entropy": 0.5,
        "per_token_entropy": [0.1, 0.9, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2],
    }

    out, captured = _run_orchestrated_stage_a(monkeypatch, cfg, teacher_output)

    assert len(out) == 1
    assert out[0]["top_k_ids"] == teacher_output["top_k_ids"]
    assert "selected_window_start" not in out[0]["extra_metadata"]
    assert captured[0]["enable_position_filtering"] is False
    assert captured[0]["selection_mode"] == "selected_windows"


def test_orchestrated_stage_a_position_mask_uses_threshold_config(monkeypatch) -> None:
    cfg = _cfg_for_stage_a_selection(
        enable_position_filtering=True,
        selection_mode="position_mask",
        entropy_threshold=0.8,
        top1_gap_threshold=0.3,
        selection_window_radius=2,
        minimum_selected_positions_per_record=1,
    )
    teacher_output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5]],
        "entropy": 0.4,
        "per_token_entropy": [0.1, 0.9, 0.1, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.2, 0.9],
    }

    out, captured = _run_orchestrated_stage_a(monkeypatch, cfg, teacher_output)

    assert len(out) == 1
    assert out[0]["extra_metadata"]["selected_positions"] == [1, 2]
    assert out[0]["extra_metadata"]["selection_policy"]["selection_mode"] == "position_mask"
    assert out[0]["extra_metadata"]["selection_policy"]["entropy_threshold"] == 0.8
    assert out[0]["extra_metadata"]["selection_policy"]["top1_gap_threshold"] == 0.3
    assert captured[0]["enable_position_filtering"] is True
    assert captured[0]["selection_mode"] == "position_mask"
    assert captured[0]["minimum_selected_positions_per_record"] == 1


def test_orchestrated_stage_a_selected_windows_uses_threshold_config(monkeypatch) -> None:
    cfg = _cfg_for_stage_a_selection(
        enable_position_filtering=True,
        selection_mode="selected_windows",
        entropy_threshold=0.75,
        top1_gap_threshold=None,
        selection_window_radius=0,
        minimum_selected_positions_per_record=None,
    )
    teacher_output = {
        "top_k_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        "top_k_logprobs": [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.4, -0.5], [-0.5, -0.6]],
        "entropy": 0.6,
        "per_token_entropy": [0.1, 0.9, 0.2, 0.8, 0.1],
        "per_token_top1_gap": [0.9, 0.9, 0.9, 0.9, 0.9],
    }

    out, captured = _run_orchestrated_stage_a(monkeypatch, cfg, teacher_output)

    assert len(out) == 1
    assert out[0]["extra_metadata"]["selected_window_start"] in {1, 3}
    assert out[0]["extra_metadata"]["selected_window_end"] in {1, 3}
    assert captured[0]["enable_position_filtering"] is True
    assert captured[0]["selection_mode"] == "selected_windows"
    assert captured[0]["entropy_threshold"] == 0.75
