#!/usr/bin/env python3
"""Preview stage-C prompt templates without running teacher backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.pipeline.prompts import build_prompt_record, list_prompt_template_names


def _load_input_text(inline_text: str | None, input_file: str | None) -> str:
    if bool(inline_text) == bool(input_file):
        raise ValueError("Provide exactly one of --input-text or --input-file")
    if inline_text is not None:
        return inline_text
    return Path(str(input_file)).read_text(encoding="utf-8")


def _parse_template_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--template-kwargs-json must decode to a JSON object")
    return parsed




def build_preview_prompt(
    *,
    template_name: str,
    input_text: str,
    template_kwargs: dict[str, Any] | None = None,
    doc_id: str = "preview_doc",
    chunk_index: int = 0,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Build preview payload using same prompt path as Stage C."""
    record = {
        "doc_id": str(doc_id),
        "chunk_index": int(chunk_index),
        "raw_bytes": input_text.encode("utf-8"),
    }
    return build_prompt_record(
        record=record,
        template_name=str(template_name),
        template_kwargs=dict(template_kwargs or {}),
        deterministic=bool(deterministic),
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a stage-C structured-output prompt template.")
    parser.add_argument("--template-name", required=True, choices=list_prompt_template_names())
    parser.add_argument(
        "--template-kwargs-json",
        default="{}",
        help="Template kwargs as JSON object, e.g. '{\"max_words\":80}'",
    )
    parser.add_argument("--input-text", help="Inline chunk text used as template input")
    parser.add_argument("--input-file", help="Path to text file used as template input")
    parser.add_argument("--doc-id", default="preview_doc")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--non-deterministic", action="store_true", help="Preview with deterministic=false")
    args = parser.parse_args()

    text = _load_input_text(args.input_text, args.input_file)
    kwargs = _parse_template_kwargs(args.template_kwargs_json)
    deterministic = False if args.non_deterministic else bool(args.deterministic)

    built = build_preview_prompt(
        template_name=str(args.template_name),
        input_text=text,
        template_kwargs=kwargs,
        doc_id=str(args.doc_id),
        chunk_index=int(args.chunk_index),
        deterministic=deterministic,
    )

    print("=== Prompt Preview ===")
    print(f"template_name: {args.template_name}")
    print(f"task_type: {built['task_type']}")
    print(f"deterministic: {built['deterministic']}")
    print("prompt_text:")
    print(built["prompt_text"])


if __name__ == "__main__":
    main()
