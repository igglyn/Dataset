#!/usr/bin/env python3
"""Build a mixed text corpus from cached source datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.corpus.mix import build_corpus_mixture_from_cache
from distill_factory.corpus.schema import load_corpus_mixture_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to corpus mixture TOML config")
    parser.add_argument("--mixture-name", required=True, help="Output mixture name")
    parser.add_argument("--cache-root", default="data/sources", help="Cached source root directory")
    parser.add_argument("--output-root", default="data/corpora", help="Output corpus root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show output path without building")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_corpus_mixture_config(args.config)

    out_dir = build_corpus_mixture_from_cache(
        cfg,
        mixture_name=args.mixture_name,
        cache_root=args.cache_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Dry-run: would build mixture '{args.mixture_name}' at {out_dir}")
    else:
        manifest_path = Path(out_dir) / "manifest.json"
        print(f"Built mixture '{args.mixture_name}' at {out_dir}")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("composition_deviation"):
                print(
                    "WARNING: realized composition deviates from requested percentages/target due to "
                    "insufficient cached source data. See manifest warnings."
                )
            for warning in manifest.get("warnings", []):
                print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
