import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from distill_factory.pipeline.stage_c import _build_structured_prompt, run_stage_c
from scripts.preview_prompt import build_preview_prompt


def _base_record() -> dict:
    return {
        "doc_id": "doc-1",
        "chunk_index": 2,
        "byte_start": 0,
        "byte_end": 24,
        "raw_bytes": b"alpha beta gamma delta",
        "split": "train",
    }


def test_stage_c_template_selection_prompt_and_task_type_propagation() -> None:
    record = _base_record()
    kwargs = {"question": "What is the key message?"}

    out = run_stage_c(
        [record],
        teacher_name="dummy",
        mode="structured_outputs",
        template_name="answer_question_from_chunk",
        template_kwargs=kwargs,
        deterministic=True,
        dry_run=True,
    )[0]

    structured = out["structured_output"]
    assert structured is not None
    assert structured["task_type"] == "answer_question_from_chunk"
    assert "Question: What is the key message?" in structured["prompt_text"]
    assert structured["prompt_text"].startswith("Answer the question using only the provided chunk.")

    tpl_meta = out["extra_metadata"]["stage_c_template"]
    assert tpl_meta["template_name"] == "answer_question_from_chunk"
    assert tpl_meta["template_kwargs"] == kwargs
    assert tpl_meta["deterministic"] is True


def test_stage_c_structured_output_fields_present_and_deterministic_true_is_repeatable() -> None:
    rec1 = _base_record()
    rec2 = _base_record()

    a = run_stage_c(
        [rec1],
        teacher_name="dummy",
        mode="structured_outputs",
        template_name="summarize_chunk",
        template_kwargs={"max_words": 40},
        deterministic=True,
        dry_run=True,
    )[0]
    b = run_stage_c(
        [rec2],
        teacher_name="dummy",
        mode="structured_outputs",
        template_name="summarize_chunk",
        template_kwargs={"max_words": 40},
        deterministic=True,
        dry_run=True,
    )[0]

    for out in (a, b):
        structured = out["structured_output"]
        assert isinstance(structured, dict)
        assert set(structured.keys()) == {"task_type", "prompt_text", "completion_text", "teacher_metadata"}
        assert isinstance(structured["task_type"], str) and structured["task_type"]
        assert isinstance(structured["prompt_text"], str) and structured["prompt_text"]
        assert isinstance(structured["completion_text"], str)

    assert a["structured_output"]["prompt_text"] == b["structured_output"]["prompt_text"]
    assert a["structured_output"]["task_type"] == b["structured_output"]["task_type"]


def test_preview_prompt_and_stage_c_builder_match_exact_prompt_text() -> None:
    record = _base_record()
    template_name = "extract_key_points"
    kwargs = {"max_points": 3}

    stage_c_prompt = _build_structured_prompt(
        record,
        template_name=template_name,
        template_kwargs=kwargs,
        deterministic=True,
    )
    preview_prompt = build_preview_prompt(
        template_name=template_name,
        input_text=record["raw_bytes"].decode("utf-8"),
        template_kwargs=kwargs,
        doc_id=record["doc_id"],
        chunk_index=record["chunk_index"],
        deterministic=True,
    )

    assert stage_c_prompt["task_type"] == preview_prompt["task_type"]
    assert stage_c_prompt["prompt_text"] == preview_prompt["prompt_text"]


def test_stage_c_structured_backend_missing_support_is_explicit() -> None:
    with pytest.raises(ValueError, match="cannot run stage_c"):
        run_stage_c(
            [_base_record()],
            teacher_name="vllm_causal_lm",
            mode="structured_outputs",
            template_name="summarize_chunk",
            template_kwargs={},
            deterministic=True,
            dry_run=False,
        )
