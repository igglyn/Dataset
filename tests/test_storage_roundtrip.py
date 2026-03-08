from distill_factory.data.formats import DistilledSample, SCHEMA_VERSION
from distill_factory.storage.reader import read_jsonl, read_jsonl_records
from distill_factory.storage.writer import write_jsonl


def test_storage_roundtrip(tmp_path):
    records = [
        DistilledSample(
            doc_id="doc-1",
            chunk_index=0,
            byte_start=0,
            byte_end=4,
            raw_bytes=b"abcd",
            split="train",
            teacher_name="bulk_grounding_teacher",
            stage_name="stage_a",
            mode="topk_logits",
            top_k_ids=[[1, 2, 3], [4, 5, 6]],
            top_k_logprobs=[[-0.1, -0.2, -0.5], [-0.3, -0.7, -1.2]],
            entropy=1.23,
            per_token_entropy=[0.2, 0.4],
            per_token_top1_gap=[1.1, 0.7],
            per_token_token_ids=[101, 202],
            per_token_valid_mask=[True, False],
            hidden_summary=[0.1, 0.2, 0.3, 0.4],
            extra_metadata={"source_path": "data/raw/doc1.txt"},
        ),
        DistilledSample(
            doc_id="doc-2",
            chunk_index=1,
            byte_start=10,
            byte_end=14,
            raw_bytes=b"\xff\xfe\x00\x01",
            split="eval",
            teacher_name="refinement_teacher",
            stage_name="stage_c",
            mode="structured_outputs",
            top_k_ids=None,
            top_k_logprobs=None,
            entropy=None,
            structured_output={
                "task_type": "refinement",
                "prompt_text": "Refine this",
                "completion_text": "Refined output",
                "teacher_metadata": {"backend": "dummy"},
            },
            extra_metadata=None,
        ),
        DistilledSample(
            doc_id="doc-3",
            chunk_index=2,
            byte_start=20,
            byte_end=30,
            raw_bytes=b"stage-b-bytes",
            split="train",
            teacher_name="long_context_teacher",
            stage_name="stage_b",
            mode="topk_logits",
            target_doc_id="doc-3",
            target_chunk_index=2,
            target_byte_start=20,
            target_byte_end=30,
            teacher_window_byte_start=10,
            teacher_window_byte_end=80,
            target_start_offset_within_window=10,
            target_end_offset_within_window=20,
            top_k_ids=[[1, 2]],
            top_k_logprobs=[[-0.1, -0.2]],
            entropy=0.9,
        ),
    ]
    out = tmp_path / "records.jsonl"

    write_jsonl(records, out)
    raw = read_jsonl_records(out)
    loaded = read_jsonl(out)

    assert loaded[0].doc_id == records[0].doc_id
    assert loaded[0].top_k_ids == records[0].top_k_ids
    assert abs(loaded[0].top_k_logprobs[0][0] - records[0].top_k_logprobs[0][0]) < 1e-3
    assert loaded[0].schema_version == SCHEMA_VERSION
    assert loaded[0].extra_metadata == {"source_path": "data/raw/doc1.txt"}
    assert loaded[0].per_token_entropy is not None
    assert abs(loaded[0].per_token_entropy[0] - 0.2) < 1e-3
    assert loaded[0].per_token_top1_gap is not None
    assert abs(loaded[0].per_token_top1_gap[1] - 0.7) < 1e-3
    assert loaded[0].per_token_token_ids == [101, 202]
    assert loaded[0].per_token_valid_mask == [True, False]
    assert loaded[0].hidden_summary is not None
    assert len(loaded[0].hidden_summary) == 4
    assert loaded[1].raw_bytes == b"\xff\xfe\x00\x01"
    assert loaded[1].structured_output is not None
    assert loaded[1].structured_output["completion_text"] == "Refined output"
    assert loaded[2].stage_name == "stage_b"
    assert loaded[2].target_doc_id == "doc-3"
    assert loaded[2].target_chunk_index == 2
    assert loaded[2].target_byte_start == 20
    assert loaded[2].target_byte_end == 30
    assert loaded[2].teacher_window_byte_start == 10
    assert loaded[2].teacher_window_byte_end == 80
    assert loaded[2].target_start_offset_within_window == 10
    assert loaded[2].target_end_offset_within_window == 20


    # Compact wire-format assertions for storage efficiency.
    assert isinstance(raw[0]["top_k_ids"], dict)
    assert raw[0]["top_k_ids"]["encoding"] == "ndarray_b64"
    assert raw[0]["top_k_ids"]["dtype"] in {"uint8", "uint16", "uint32"}

    assert isinstance(raw[0]["top_k_logprobs"], dict)
    assert raw[0]["top_k_logprobs"]["encoding"] == "ndarray_b64"
    assert raw[0]["top_k_logprobs"]["dtype"] == "float16"

    assert isinstance(raw[0]["entropy"], float)
    assert isinstance(raw[0]["per_token_entropy"], dict)
    assert raw[0]["per_token_entropy"]["encoding"] == "ndarray_b64"
    assert raw[0]["per_token_entropy"]["dtype"] == "float16"
    assert isinstance(raw[0]["per_token_top1_gap"], dict)
    assert raw[0]["per_token_top1_gap"]["encoding"] == "ndarray_b64"
    assert raw[0]["per_token_top1_gap"]["dtype"] == "float16"
    assert isinstance(raw[0]["per_token_token_ids"], dict)
    assert raw[0]["per_token_token_ids"]["encoding"] == "ndarray_b64"
    assert raw[0]["per_token_token_ids"]["dtype"] in {"uint8", "uint16", "uint32"}
    assert isinstance(raw[0]["per_token_valid_mask"], dict)
    assert raw[0]["per_token_valid_mask"]["encoding"] == "ndarray_b64"
    assert raw[0]["per_token_valid_mask"]["dtype"] in {"uint8", "uint16", "uint32"}
    assert isinstance(raw[0]["hidden_summary"], dict)
    assert raw[0]["hidden_summary"]["encoding"] == "ndarray_b64"
    assert raw[0]["hidden_summary"]["dtype"] == "float16"

    assert raw[2]["target_doc_id"] == "doc-3"
    assert raw[2]["target_chunk_index"] == 2
    assert raw[2]["target_byte_start"] == 20
    assert raw[2]["target_byte_end"] == 30
    assert raw[2]["teacher_window_byte_start"] == 10
    assert raw[2]["teacher_window_byte_end"] == 80
    assert raw[2]["target_start_offset_within_window"] == 10
    assert raw[2]["target_end_offset_within_window"] == 20


def test_storage_writer_deduplicates_by_signature(tmp_path):
    sample = DistilledSample(
        doc_id="doc-dup",
        chunk_index=7,
        byte_start=70,
        byte_end=80,
        raw_bytes=b"duplicate",
        split="train",
        teacher_name="dummy_teacher",
        stage_name="stage_a",
        mode="topk_logits",
        top_k_ids=[[1, 2]],
        top_k_logprobs=[[-0.1, -0.2]],
        entropy=0.5,
    )

    # Same identity/signature fields => only one should be written.
    duplicate = DistilledSample(
        doc_id="doc-dup",
        chunk_index=7,
        byte_start=70,
        byte_end=80,
        raw_bytes=b"duplicate-but-same-identity",
        split="train",
        teacher_name="dummy_teacher",
        stage_name="stage_a",
        mode="topk_logits",
        top_k_ids=[[9, 9]],
        top_k_logprobs=[[-9.0, -9.0]],
        entropy=9.0,
    )

    out = tmp_path / "dupes.jsonl"
    write_jsonl([sample, duplicate], out)

    loaded = read_jsonl(out)
    raw = read_jsonl_records(out)

    assert len(loaded) == 1
    assert len(raw) == 1
    assert loaded[0].doc_id == "doc-dup"
    assert loaded[0].chunk_index == 7
