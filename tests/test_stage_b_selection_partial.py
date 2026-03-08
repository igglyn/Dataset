import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill_factory.data.chunking import chunk_document_bytes
from distill_factory.pipeline.stage_b import run_stage_b


def _base_records() -> list[dict[str, object]]:
    doc = {"doc_id": "doc-1", "text": "abcdefghijklmno"}
    records = chunk_document_bytes(doc, chunk_bytes=5, overlap_bytes=0)
    return [dict(r, top_k=4) for r in records]


def test_stage_b_dense_mode_still_runs() -> None:
    records = _base_records()
    for r in records:
        r["enable_position_filtering"] = False
        r["selection_mode"] = "none"
        r["entropy_threshold"] = None
        r["top1_gap_threshold"] = None

    out = run_stage_b(
        records,
        teacher_name="dummy",
        mode="long_context",
        context_window=12,
        stride=1,
        max_teacher_context=12,
        dry_run=True,
    )

    assert len(out) == len(records)
    assert all("selected_window_start" not in (r.get("extra_metadata") or {}) for r in out)


def test_stage_b_selection_export_mode_fails_fast() -> None:
    records = _base_records()
    records[0]["enable_position_filtering"] = True
    records[0]["selection_mode"] = "selected_windows"
    records[0]["entropy_threshold"] = 0.8
    records[0]["top1_gap_threshold"] = None

    with pytest.raises(NotImplementedError, match="Stage B selection-aware export is not implemented"):
        run_stage_b(
            records,
            teacher_name="dummy",
            mode="long_context",
            context_window=12,
            stride=1,
            max_teacher_context=12,
            dry_run=True,
        )


def test_stage_b_position_mask_mode_fails_fast_even_if_flag_false() -> None:
    records = _base_records()
    records[0]["enable_position_filtering"] = False
    records[0]["selection_mode"] = "position_mask"
    records[0]["entropy_threshold"] = 0.8

    with pytest.raises(NotImplementedError, match="selection_mode='position_mask'"):
        run_stage_b(
            records,
            teacher_name="dummy",
            mode="long_context",
            context_window=12,
            stride=1,
            max_teacher_context=12,
            dry_run=True,
        )
