#!/usr/bin/env python3
"""Extract one named source dataset into persistent cached text documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_factory.corpus.extract import extract_source_to_cache, inspect_source_cache_state
from distill_factory.corpus.schema import load_corpus_mixture_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to corpus mixture TOML config")
    parser.add_argument("--source-name", required=True, help="source_name to extract")
    parser.add_argument(
        "--cache-root",
        default="data/sources",
        help="Root directory for source caches (default: data/sources)",
    )
    parser.add_argument(
        "--refresh",
        "--overwrite",
        action="store_true",
        help="Force refresh existing cache if present (deletes and rebuilds source cache directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cache state/action without extracting data",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_corpus_mixture_config(args.config)

    source = next((s for s in cfg.source_extraction.datasets if s.source_name == args.source_name), None)
    if source is None:
        available = ", ".join(sorted(s.source_name for s in cfg.source_extraction.datasets))
        raise SystemExit(
            f"Unknown source-name '{args.source_name}'. Available in config: [{available}]"
        )

    cache_root = Path(args.cache_root)
    state = inspect_source_cache_state(source=source, cache_root=cache_root)

    if args.dry_run:
        print(f"Dry-run: source='{source.source_name}' state='{state['state']}' cache_root='{cache_root}'")
        if state.get("reason"):
            print(f"Reason: {state['reason']}")
        if state.get("existing_config_fingerprint"):
            print(f"Existing fingerprint: {state['existing_config_fingerprint']}")
            print(f"Requested fingerprint: {state['requested_config_fingerprint']}")
        if state["state"] == "ready" and not args.refresh:
            print("Action: skip extraction (cache is reusable and matches config)")
        elif args.refresh:
            print("Action: refresh extraction (overwrite enabled)")
        else:
            print("Action: extract if no conflicting cache state")
        return 0

    out_dir = extract_source_to_cache(
        source=source,
        cache_root=cache_root,
        refresh=args.refresh,
    )

    if state["state"] == "ready" and not args.refresh:
        print(f"Cache already matches requested config, skipped extraction: {out_dir}")
    else:
        print(f"Extracted source '{source.source_name}' to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
