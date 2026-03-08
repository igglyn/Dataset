import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill_factory.pipeline import orchestrator as orch
from distill_factory.storage.reader import read_jsonl_records
from distill_factory.storage.writer import write_jsonl


def test_to_distilled_samples_preserves_per_token_selection_fields() -> None:
    stage_output = [
        {
            "doc_id": "doc-1",
            "chunk_index": 0,
            "byte_start": 0,
            "byte_end": 4,
            "raw_bytes": b"abcd",
            "split": "train",
            "teacher_name": "dummy",
            "stage_name": "stage_a",
            "mode": "topk_logits",
            "top_k_ids": [[1, 2]],
            "top_k_logprobs": [[-0.1, -0.2]],
            "entropy": 0.3,
            "per_token_entropy": [0.11, 0.22],
            "per_token_top1_gap": [0.33, 0.44],
            "per_token_token_ids": [101, 102],
            "per_token_valid_mask": [1, 0],
        }
    ]

    samples = orch._to_distilled_samples(stage_output)

    assert len(samples) == 1
    assert samples[0].per_token_entropy == [0.11, 0.22]
    assert samples[0].per_token_top1_gap == [0.33, 0.44]
    assert samples[0].per_token_token_ids == [101, 102]
    assert samples[0].per_token_valid_mask == [1, 0]


def test_orchestrator_serialization_path_keeps_per_token_selection_fields(tmp_path) -> None:
    stage_output = [
        {
            "doc_id": "doc-2",
            "chunk_index": 1,
            "byte_start": 10,
            "byte_end": 20,
            "raw_bytes": b"stage-output",
            "split": "eval",
            "teacher_name": "dummy",
            "stage_name": "stage_b",
            "mode": "topk_logits",
            "top_k_ids": [[3, 4]],
            "top_k_logprobs": [[-0.3, -0.4]],
            "entropy": 0.7,
            "per_token_entropy": [0.5, 0.6],
            "per_token_top1_gap": [0.2, 0.1],
            "per_token_token_ids": [203, 204],
            "per_token_valid_mask": [True, False],
        }
    ]

    samples = orch._to_distilled_samples(stage_output)
    out = tmp_path / "out.jsonl"
    write_jsonl(samples, out)
    records = read_jsonl_records(out)

    assert len(records) == 1
    assert records[0]["per_token_entropy"] is not None
    assert records[0]["per_token_top1_gap"] is not None
    assert records[0]["per_token_token_ids"] is not None
    assert records[0]["per_token_valid_mask"] is not None


def test_to_distilled_samples_backward_compatible_without_per_token_fields() -> None:
    legacy_stage_output = [
        {
            "doc_id": "doc-legacy",
            "chunk_index": 0,
            "byte_start": 0,
            "byte_end": 4,
            "raw_bytes": b"legacy",
            "split": "train",
            "teacher_name": "dummy",
            "stage_name": "stage_a",
            "mode": "topk_logits",
            "top_k_ids": [[1, 2]],
            "top_k_logprobs": [[-0.1, -0.2]],
            "entropy": 0.2,
        }
    ]

    samples = orch._to_distilled_samples(legacy_stage_output)

    assert len(samples) == 1
    assert samples[0].per_token_entropy is None
    assert samples[0].per_token_top1_gap is None
    assert samples[0].per_token_token_ids is None
    assert samples[0].per_token_valid_mask is None
