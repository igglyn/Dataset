#!/usr/bin/env python3
"""Inspect Hugging Face tokenizer padding expectations for a model.

Examples:
  python scripts/check_hf_padding_expectation.py --model distilgpt2
  python scripts/check_hf_padding_expectation.py --model distilgpt2 --hf-pad-token-id 0
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def summarize_padding_expectation(tokenizer: Any, configured_pad_token_id: int | None = None) -> dict[str, Any]:
    vocab_size_raw = getattr(tokenizer, "vocab_size", None)
    vocab_size = int(vocab_size_raw) if isinstance(vocab_size_raw, int) and vocab_size_raw > 0 else None

    tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    tokenizer_pad_token = getattr(tokenizer, "pad_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)

    configured_in_vocab_range: bool | None = None
    if configured_pad_token_id is not None and vocab_size is not None:
        configured_in_vocab_range = 0 <= int(configured_pad_token_id) < int(vocab_size)

    return {
        "tokenizer_pad_token_id": tokenizer_pad_token_id,
        "tokenizer_pad_token": tokenizer_pad_token,
        "eos_token_id": eos_token_id,
        "eos_token": eos_token,
        "unk_token_id": unk_token_id,
        "unk_token": unk_token,
        "vocab_size": vocab_size,
        "configured_hf_pad_token_id": configured_pad_token_id,
        "configured_hf_pad_token_id_in_vocab_range": configured_in_vocab_range,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect tokenizer padding expectations for an HF model/tokenizer")
    parser.add_argument("--model", required=True, help="Hugging Face model/tokenizer name or local path")
    parser.add_argument(
        "--hf-pad-token-id",
        type=int,
        default=None,
        help="Optional pad token id from config to validate against tokenizer vocab range",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("transformers is required for this script") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    summary = summarize_padding_expectation(tokenizer, configured_pad_token_id=args.hf_pad_token_id)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
