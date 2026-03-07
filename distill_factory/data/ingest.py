"""Raw document ingestion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _stable_doc_id(path: Path) -> str:
    """Return a stable document id derived from normalized file path."""
    return hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()


def ingest_documents(
    input_path: str,
    file_glob: str,
    encoding: str = "utf-8",
    normalize_newlines: bool = True,
) -> list[dict]:
    """Read source text files and return document records with stable ids."""
    root = Path(input_path)
    files = sorted(root.glob(file_glob), key=lambda p: p.as_posix())

    docs: list[dict] = []
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding=encoding)
        if normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        rel_path = path.relative_to(root)
        docs.append(
            {
                "doc_id": _stable_doc_id(rel_path),
                "text": text,
                "metadata": {"source_path": path.as_posix()},
            }
        )

    return docs


def ingest_text_file(path: str | Path) -> str:
    """Backward-compatible single-file helper."""
    return Path(path).read_text(encoding="utf-8")
