"""Simple stage-C prompt template registry."""

from __future__ import annotations

from typing import Any, Callable

TemplateBuilder = Callable[[dict[str, Any], dict[str, Any], bool], dict[str, Any]]


def _chunk_text(record: dict[str, Any]) -> str:
    raw = record.get("raw_bytes", b"")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _summarize_chunk(record: dict[str, Any], kwargs: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    max_words = int(kwargs.get("max_words", 120))
    text = _chunk_text(record)
    prompt = (
        f"Summarize the following chunk in at most {max_words} words.\n"
        f"Document: {record.get('doc_id')}\n"
        f"Chunk: {record.get('chunk_index')}\n\n"
        f"Chunk text:\n{text}\n"
    )
    return {
        "task_type": "summarize_chunk",
        "prompt_text": prompt,
        "template_metadata": {"max_words": max_words, "deterministic": deterministic},
    }


def _answer_question_from_chunk(record: dict[str, Any], kwargs: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    question = str(kwargs.get("question", "What is the most important information in this chunk?"))
    text = _chunk_text(record)
    prompt = (
        "Answer the question using only the provided chunk.\n"
        f"Question: {question}\n"
        f"Document: {record.get('doc_id')}\n"
        f"Chunk: {record.get('chunk_index')}\n\n"
        f"Chunk text:\n{text}\n"
    )
    return {
        "task_type": "answer_question_from_chunk",
        "prompt_text": prompt,
        "template_metadata": {"question": question, "deterministic": deterministic},
    }


def _continue_document(record: dict[str, Any], kwargs: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    continuation_tokens = int(kwargs.get("continuation_tokens", 128))
    text = _chunk_text(record)
    prompt = (
        f"Continue the document naturally for about {continuation_tokens} tokens.\n"
        f"Document: {record.get('doc_id')}\n"
        f"Chunk: {record.get('chunk_index')}\n\n"
        "Current excerpt:\n"
        f"{text}\n"
    )
    return {
        "task_type": "continue_document",
        "prompt_text": prompt,
        "template_metadata": {"continuation_tokens": continuation_tokens, "deterministic": deterministic},
    }


def _extract_key_points(record: dict[str, Any], kwargs: dict[str, Any], deterministic: bool) -> dict[str, Any]:
    max_points = int(kwargs.get("max_points", 5))
    text = _chunk_text(record)
    prompt = (
        f"Extract up to {max_points} key points from the chunk as concise bullet points.\n"
        f"Document: {record.get('doc_id')}\n"
        f"Chunk: {record.get('chunk_index')}\n\n"
        f"Chunk text:\n{text}\n"
    )
    return {
        "task_type": "extract_key_points",
        "prompt_text": prompt,
        "template_metadata": {"max_points": max_points, "deterministic": deterministic},
    }


_TEMPLATE_REGISTRY: dict[str, TemplateBuilder] = {
    "summarize_chunk": _summarize_chunk,
    "answer_question_from_chunk": _answer_question_from_chunk,
    "continue_document": _continue_document,
    "extract_key_points": _extract_key_points,
}




def list_prompt_template_names() -> list[str]:
    """Return available stage-C prompt template names."""
    return sorted(_TEMPLATE_REGISTRY.keys())

def get_prompt_template(name: str) -> TemplateBuilder:
    if name not in _TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown stage_c template_name: {name}. Available: {', '.join(list_prompt_template_names())}"
        )
    return _TEMPLATE_REGISTRY[name]


def build_prompt_record(
    record: dict[str, Any],
    template_name: str,
    template_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    kwargs = dict(template_kwargs or {})
    builder = get_prompt_template(template_name)
    built = builder(record, kwargs, deterministic)

    raw = record.get("raw_bytes", b"")
    raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")

    return {
        "task_type": str(built["task_type"]),
        "prompt_text": str(built["prompt_text"]),
        "raw_bytes": raw_bytes,
        "template_name": template_name,
        "template_kwargs": kwargs,
        "template_metadata": built.get("template_metadata") if isinstance(built.get("template_metadata"), dict) else {},
        "deterministic": bool(deterministic),
    }
