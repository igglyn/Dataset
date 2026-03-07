import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill_factory.data.chunking import chunk_document_bytes
from distill_factory.pipeline.stage_b import run_stage_b


def _run_stage_b(records, *, context_window, stride, max_teacher_context, window_policy="center_target", target_region_policy="preserve_full"):
    inputs = [dict(r, top_k=4) for r in records]
    return run_stage_b(
        inputs,
        teacher_name="dummy",
        mode="long_context",
        context_window=context_window,
        stride=stride,
        max_teacher_context=max_teacher_context,
        window_policy=window_policy,
        target_region_policy=target_region_policy,
        dry_run=True,
    )


def test_stage_b_target_inside_window_and_offsets_and_no_cross_doc_boundaries() -> None:
    doc_a = {"doc_id": "doc-a", "text": "AAAABBBBCCCCDDDD"}
    doc_b = {"doc_id": "doc-b", "text": "wxyzmnopQRST"}
    records = chunk_document_bytes(doc_a, chunk_bytes=4, overlap_bytes=0) + chunk_document_bytes(doc_b, chunk_bytes=4, overlap_bytes=0)

    out = _run_stage_b(records, context_window=8, stride=1, max_teacher_context=8)

    for rec in out:
        assert rec["target_doc_id"] == rec["doc_id"]
        assert rec["teacher_window_byte_start"] <= rec["target_byte_start"]
        assert rec["target_byte_end"] <= rec["teacher_window_byte_end"]

        expected_start = rec["target_byte_start"] - rec["teacher_window_byte_start"]
        expected_end = rec["target_byte_end"] - rec["teacher_window_byte_start"]
        assert rec["target_start_offset_within_window"] == expected_start
        assert rec["target_end_offset_within_window"] == expected_end

        text = rec["extra_metadata"]["stage_b_context"]["long_context_text"]
        if rec["doc_id"] == "doc-a":
            assert set(text) <= set("ABCD")
        else:
            assert not any(ch in text for ch in "ABCD")


def test_stage_b_near_start_near_end_and_short_document_cases() -> None:
    doc = {"doc_id": "edge-doc", "text": "abcdefghijklmnop"}
    short_doc = {"doc_id": "short-doc", "text": "tiny"}

    edge_chunks = chunk_document_bytes(doc, chunk_bytes=4, overlap_bytes=0)
    short_chunks = chunk_document_bytes(short_doc, chunk_bytes=16, overlap_bytes=0)

    # Include complete edge doc chunk set (chunk_index must be contiguous for current builder),
    # and one short-doc chunk.
    selected = edge_chunks + short_chunks
    out = _run_stage_b(selected, context_window=12, stride=1, max_teacher_context=12)

    edge_out = [r for r in out if r["doc_id"] == "edge-doc"]
    near_start = edge_out[0]
    near_end = edge_out[-1]
    short = [r for r in out if r["doc_id"] == "short-doc"][0]

    assert near_start["teacher_window_byte_start"] == 0
    assert near_start["target_start_offset_within_window"] == 0

    assert near_end["teacher_window_byte_end"] == near_end["target_byte_end"]
    assert near_end["target_end_offset_within_window"] == near_end["teacher_window_byte_end"] - near_end["teacher_window_byte_start"]

    short_text = short["extra_metadata"]["stage_b_context"]["long_context_text"]
    assert short_text == "tiny"
    assert short["teacher_window_byte_start"] == 0
    assert short["teacher_window_byte_end"] == len(short_doc["text"])


def test_stage_b_window_policy_placements_left_right_center() -> None:
    doc = {"doc_id": "policy-doc", "text": "abcdefghijklmnopqrst"}
    chunks = chunk_document_bytes(doc, chunk_bytes=4, overlap_bytes=0)

    left = _run_stage_b(chunks, context_window=20, stride=2, max_teacher_context=6, window_policy="left_biased")[2]
    right = _run_stage_b(chunks, context_window=20, stride=2, max_teacher_context=6, window_policy="right_biased")[2]
    center = _run_stage_b(chunks, context_window=20, stride=2, max_teacher_context=6, window_policy="center_target")[2]

    # For target [8,12) with max context 6 and full-window input, expected offsets are:
    # left_biased => target ends at right edge; right_biased => target starts at left edge;
    # center_target => approximately centered.
    assert (left["target_start_offset_within_window"], left["target_end_offset_within_window"]) == (2, 6)
    assert (right["target_start_offset_within_window"], right["target_end_offset_within_window"]) == (0, 4)
    assert (center["target_start_offset_within_window"], center["target_end_offset_within_window"]) == (1, 5)


def test_stage_b_truncation_policy_preserve_full_vs_truncate_if_needed() -> None:
    # Single large target chunk where target span is larger than max_teacher_context.
    records = [
        {
            "doc_id": "big-target",
            "chunk_index": 0,
            "byte_start": 0,
            "byte_end": 10,
            "raw_bytes": b"abcdefghij",
            "top_k": 4,
        }
    ]

    keep_full = _run_stage_b(
        records,
        context_window=20,
        stride=0,
        max_teacher_context=6,
        window_policy="center_target",
        target_region_policy="preserve_full",
    )[0]
    truncate = _run_stage_b(
        records,
        context_window=20,
        stride=0,
        max_teacher_context=6,
        window_policy="center_target",
        target_region_policy="truncate_if_needed",
    )[0]

    keep_meta = keep_full["extra_metadata"]["stage_b_context"]["truncation"]
    trunc_meta = truncate["extra_metadata"]["stage_b_context"]["truncation"]

    # preserve_full keeps whole target span even when it exceeds max_teacher_context.
    assert keep_full["target_start_offset_within_window"] == 0
    assert keep_full["target_end_offset_within_window"] == 10
    assert keep_meta["final_window_bytes"] == 10

    # truncate_if_needed keeps context budget and clips target span by policy.
    assert truncate["target_start_offset_within_window"] == 0
    assert truncate["target_end_offset_within_window"] == 6
    assert trunc_meta["final_window_bytes"] == 6
