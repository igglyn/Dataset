import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from distill_factory.config.schema import load_config
from distill_factory.data.formats import DistilledSample
from distill_factory.pipeline import orchestrator as orch
from distill_factory.pipeline.resume_state import (
    build_initial_resume_state,
    load_resume_state,
    validate_resume_state,
    write_resume_state,
)
from distill_factory.storage.reader import read_jsonl_records
from distill_factory.storage.writer import write_jsonl


def _write_docs(root: pathlib.Path, count: int = 5) -> pathlib.Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (raw / f"doc_{i}.txt").write_text(f"document {i}\nhello\n", encoding="utf-8")
    return raw


def _write_config(
    path: pathlib.Path,
    *,
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    resume: bool,
    resume_policy: str,
    stage_a_top_k: int = 4,
    max_records_per_shard: int = 0,
) -> None:
    path.write_text(
        f"""
[data]
input_path = "{input_path.as_posix()}"
file_glob = "*.txt"
encoding = "utf-8"
chunk_bytes = 64
overlap_bytes = 0
eval_fraction = 0.25
seed = 13

[input]
preserve_document_boundaries = true
normalize_newlines = true

[output]
output_dir = "{output_dir.as_posix()}"
format = "jsonl"
compression = null
max_records_per_shard = {max_records_per_shard}
shard_prefix = "shard"
resume = {str(resume).lower()}
resume_policy = "{resume_policy}"
dry_run = false
dry_run_max_records = 10
log_token_lengths = false
log_byte_lengths = false
stop_after_stage = "stage_a"

[stage_a]
enabled = true
teacher_name = "dummy"
mode = "topk_logits"
top_k = {stage_a_top_k}
temperature = 1.0

[stage_b]
enabled = false
teacher_name = "dummy"
mode = "long_context"
top_k = 4
temperature = 1.0
context_window = 64
stride = 1

[stage_c]
enabled = false
teacher_name = "dummy"
mode = "structured_outputs"
top_k = 4
temperature = 0.7
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_initial_run_then_resume_does_not_duplicate_records(tmp_path: pathlib.Path) -> None:
    raw = _write_docs(tmp_path, count=6)
    cfg = tmp_path / "config.toml"
    out = tmp_path / "out"
    _write_config(cfg, input_path=raw, output_dir=out, resume=True, resume_policy="strict")

    first = orch.run_pipeline(str(cfg))
    first_total = len(read_jsonl_records(first["train_path"])) + len(read_jsonl_records(first["eval_path"]))
    assert first_total > 0

    second = orch.run_pipeline(str(cfg))
    second_total = len(read_jsonl_records(second["train_path"])) + len(read_jsonl_records(second["eval_path"]))

    assert second_total == first_total
    assert second["train_count"] == first["train_count"]
    assert second["eval_count"] == first["eval_count"]


def test_resume_policy_strict_and_best_effort_validation(tmp_path: pathlib.Path) -> None:
    raw = _write_docs(tmp_path, count=4)

    cfg_base_path = tmp_path / "base.toml"
    cfg_noncritical_path = tmp_path / "noncritical.toml"
    cfg_critical_path = tmp_path / "critical.toml"
    out = tmp_path / "out"

    _write_config(cfg_base_path, input_path=raw, output_dir=out, resume=True, resume_policy="strict", stage_a_top_k=4, max_records_per_shard=0)
    _write_config(cfg_noncritical_path, input_path=raw, output_dir=out, resume=True, resume_policy="best_effort", stage_a_top_k=4, max_records_per_shard=2)
    _write_config(cfg_critical_path, input_path=raw, output_dir=out, resume=True, resume_policy="strict", stage_a_top_k=8, max_records_per_shard=0)

    cfg_base = load_config(cfg_base_path)
    state = build_initial_resume_state(cfg_base, str(cfg_base_path), {"stage_a": ["dummy"], "stage_b": [], "stage_c": []})

    ok_strict, msg_strict = validate_resume_state(state, load_config(cfg_critical_path), resume_policy="strict")
    assert not ok_strict
    assert msg_strict is not None and "strict mode" in msg_strict

    ok_best, msg_best = validate_resume_state(state, load_config(cfg_noncritical_path), resume_policy="best_effort")
    assert ok_best
    assert msg_best is not None and "non-critical" in msg_best


def test_dedupe_signatures_prevent_duplicates_across_repeated_appends(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "records.jsonl"

    a = DistilledSample(
        doc_id="doc-1",
        chunk_index=0,
        byte_start=0,
        byte_end=4,
        raw_bytes=b"aaaa",
        split="train",
        teacher_name="dummy",
        stage_name="stage_a",
        mode="topk_logits",
        top_k_ids=[[1, 2]],
        top_k_logprobs=[[-0.1, -0.2]],
        entropy=0.1,
    )
    b = DistilledSample(
        doc_id="doc-2",
        chunk_index=0,
        byte_start=0,
        byte_end=4,
        raw_bytes=b"bbbb",
        split="train",
        teacher_name="dummy",
        stage_name="stage_a",
        mode="topk_logits",
        top_k_ids=[[1, 2]],
        top_k_logprobs=[[-0.1, -0.2]],
        entropy=0.1,
    )

    write_jsonl([a], out, append=False)
    write_jsonl([a, a, b], out, append=True)

    records = read_jsonl_records(out)
    assert len(records) == 2


def test_simulated_interruption_resume_continues_without_duplication(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _write_docs(tmp_path, count=8)

    interrupted_cfg = tmp_path / "interrupted.toml"
    interrupted_out = tmp_path / "interrupted_out"
    _write_config(interrupted_cfg, input_path=raw, output_dir=interrupted_out, resume=True, resume_policy="strict")

    baseline_cfg = tmp_path / "baseline.toml"
    baseline_out = tmp_path / "baseline_out"
    _write_config(baseline_cfg, input_path=raw, output_dir=baseline_out, resume=False, resume_policy="strict")

    baseline = orch.run_pipeline(str(baseline_cfg))
    baseline_total = len(read_jsonl_records(baseline["train_path"])) + len(read_jsonl_records(baseline["eval_path"]))

    real_write_split = orch._write_split
    call_count = {"n": 0}

    def flaky_write_split(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated interruption during eval write")
        return real_write_split(*args, **kwargs)

    monkeypatch.setattr(orch, "_write_split", flaky_write_split)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        orch.run_pipeline(str(interrupted_cfg))

    state = load_resume_state(interrupted_out)
    assert state is not None
    assert state.get("split_progress", {}).get("train", {}).get("completed") is True
    assert state.get("split_progress", {}).get("eval", {}).get("completed", False) is not True

    monkeypatch.setattr(orch, "_write_split", real_write_split)
    resumed = orch.run_pipeline(str(interrupted_cfg))
    resumed_total = len(read_jsonl_records(resumed["train_path"])) + len(read_jsonl_records(resumed["eval_path"]))

    assert resumed_total == baseline_total
