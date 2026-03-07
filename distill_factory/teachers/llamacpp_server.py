"""llama.cpp server-backed teacher runtime.

This backend talks to a user-managed llama.cpp HTTP server and is intentionally
minimal for architectural bring-up.

Assumptions (easy to adjust):
- Endpoint availability differs across llama.cpp server versions/builds.
- Startup check probes a small endpoint catalog and succeeds when at least one
  known endpoint responds successfully.
- Top-k distillation currently uses OpenAI-compatible `/v1/completions` prompt
  logprob payloads. If required fields are unavailable, inference fails
  explicitly instead of guessing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any
from urllib import error, parse, request

from .base import Teacher
from .long_context import prepare_long_context_teacher_input
from .runtime_base import RuntimeCapabilities, TeacherRuntime


# Endpoint catalog is explicit so variants can be adjusted in one place.
_PROBE_ENDPOINTS: tuple[str, ...] = (
    "/health",
    "/v1/models",
    "/props",
)

# OpenAI-compatible completion payload field variants observed across servers.
_TOKEN_IDS_FIELDS: tuple[str, ...] = ("prompt_token_ids", "token_ids", "tokens_ids")
_TOP_LOGPROBS_FIELDS: tuple[str, ...] = ("prompt_top_logprobs", "top_logprobs")
_TOKENIZE_ENDPOINTS: tuple[str, ...] = ("/tokenize", "/v1/tokenize")


@dataclass(slots=True)
class _ProbeResult:
    endpoint: str
    ok: bool
    status_code: int | None
    detail: str
    body: Any


class LlamaCppServerTeacher(Teacher, TeacherRuntime):
    """Teacher that validates and targets an external llama.cpp HTTP server."""

    def __init__(
        self,
        base_url: str,
        model_hint: str | None = None,
        request_timeout: float = 30.0,
        max_context: int = 2048,
        default_top_k: int = 5,
        default_temperature: float = 0.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model_hint = None if model_hint is None else str(model_hint)
        self.request_timeout = float(request_timeout)
        self.max_context = max(1, int(max_context))
        self.default_top_k = max(1, int(default_top_k))
        self.default_temperature = float(default_temperature)
        self._prepared = False
        self._server_metadata: dict[str, Any] = {}
        self._supports_tokenizer_diagnostics = False
        self._tokenizer_diagnostics_detail: str | None = None

    def supports_topk(self) -> bool:
        return True

    def supports_structured(self) -> bool:
        return False

    def supports_long_context(self) -> bool:
        return True

    def supports_hidden_summary(self) -> bool:
        return False

    def supports_tokenizer_diagnostics(self) -> bool:
        return bool(self._supports_tokenizer_diagnostics)

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            backend_type="llamacpp_server",
            supports_topk=True,
            supports_structured=False,
            supports_tokenizer_diagnostics=self.supports_tokenizer_diagnostics(),
        )

    def _http_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        url = parse.urljoin(f"{self.base_url}/", endpoint.lstrip("/"))
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, method=method.upper(), headers=headers, data=data)
        with request.urlopen(req, timeout=self.request_timeout) as resp:
            status_code = int(getattr(resp, "status", 200))
            raw = resp.read()
        if not raw:
            return status_code, None
        try:
            return status_code, json.loads(raw.decode("utf-8"))
        except Exception:
            return status_code, raw.decode("utf-8", errors="replace")

    def _probe(self, endpoint: str) -> _ProbeResult:
        try:
            status_code, body = self._http_json("GET", endpoint)
            return _ProbeResult(
                endpoint=endpoint,
                ok=(200 <= status_code < 300),
                status_code=status_code,
                detail=str(body)[:300],
                body=body,
            )
        except error.HTTPError as exc:
            return _ProbeResult(endpoint=endpoint, ok=False, status_code=int(exc.code), detail=str(exc), body=None)
        except Exception as exc:
            return _ProbeResult(endpoint=endpoint, ok=False, status_code=None, detail=str(exc), body=None)

    def _extract_from_mapping(self, body: Any, keys: tuple[str, ...]) -> Any:
        if not isinstance(body, dict):
            return None
        for key in keys:
            if key in body and body[key] is not None:
                return body[key]
        return None

    def _discover_server_metadata(self, probes: list[_ProbeResult]) -> dict[str, Any]:
        ok_probes = [p for p in probes if p.ok]
        available_endpoints = [p.endpoint for p in ok_probes]

        version: str | None = None
        build: str | None = None
        context_length: int | None = None
        model_metadata: Any = None

        for probe in ok_probes:
            body = probe.body
            if version is None:
                raw_version = self._extract_from_mapping(body, ("version", "server_version", "llama_cpp_version"))
                if raw_version is not None:
                    version = str(raw_version)

            if build is None:
                raw_build = self._extract_from_mapping(body, ("build", "build_info", "build_commit"))
                if raw_build is not None:
                    build = str(raw_build)

            if context_length is None:
                raw_ctx = self._extract_from_mapping(body, ("n_ctx", "context_length", "max_context_length"))
                try:
                    if raw_ctx is not None:
                        context_length = int(raw_ctx)
                except Exception:
                    context_length = None

            if model_metadata is None:
                if probe.endpoint == "/v1/models" and isinstance(body, dict):
                    model_metadata = body.get("data", body)
                elif probe.endpoint == "/props":
                    model_metadata = body

        return {
            "available_endpoints": available_endpoints,
            "version": version,
            "build": build,
            "model_metadata": model_metadata,
            "reported_context_length": context_length,
        }

    def _detect_tokenizer_diagnostics_support(self) -> tuple[bool, str]:
        try:
            self._tokenize_via_endpoint("diagnostics_probe")
            return True, "tokenization endpoint available"
        except NotImplementedError as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover
            return False, str(exc)

    def startup_self_check(self, requested_top_k: int | None = None) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("llama.cpp server teacher misconfiguration: base_url must be non-empty.")
        if self.request_timeout <= 0:
            raise ValueError("llama.cpp server teacher misconfiguration: request_timeout must be > 0.")
        if self.max_context < 1:
            raise ValueError("llama.cpp server teacher misconfiguration: max_context must be >= 1.")

        effective_top_k = self.default_top_k if requested_top_k is None else int(requested_top_k)
        if effective_top_k < 1:
            raise ValueError("llama.cpp server teacher misconfiguration: requested top_k must be >= 1.")

        probes = [self._probe(endpoint) for endpoint in _PROBE_ENDPOINTS]
        if not any(p.ok for p in probes):
            summary = "; ".join(
                f"{p.endpoint}: status={p.status_code}, detail={p.detail}" for p in probes
            )
            raise RuntimeError(
                "llama.cpp server startup check failed: server unreachable or unsupported endpoint surface. "
                f"Tried {summary}"
            )

        discovered = self._discover_server_metadata(probes)
        tokenizer_diag_supported, tokenizer_diag_detail = self._detect_tokenizer_diagnostics_support()
        self._supports_tokenizer_diagnostics = bool(tokenizer_diag_supported)
        self._tokenizer_diagnostics_detail = tokenizer_diag_detail
        discovered["supports_tokenizer_diagnostics"] = bool(tokenizer_diag_supported)
        discovered["tokenizer_diagnostics_detail"] = tokenizer_diag_detail
        self._server_metadata = discovered

        return {
            "backend": "llamacpp_server",
            "base_url": self.base_url,
            "model_hint": self.model_hint,
            "max_context": int(self.max_context),
            "requested_top_k": int(effective_top_k),
            "temperature": float(self.default_temperature),
            "discovered": discovered,
            "ok": True,
        }

    def _prepare_stage_b_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record.get("window_raw_bytes"), bytes):
            return record

        prepared = prepare_long_context_teacher_input(
            window_raw_bytes=record["window_raw_bytes"],
            target_start_offset=int(record.get("target_start_offset", 0)),
            target_end_offset=int(record.get("target_end_offset", 0)),
            max_teacher_context=int(record.get("max_teacher_context", self.max_context)),
            window_policy=str(record.get("window_policy", "center_target")),
            target_region_policy=str(record.get("target_region_policy", "preserve_full")),
        )

        out = dict(record)
        out["teacher_input_bytes"] = prepared["teacher_input_bytes"]
        out["teacher_input_text"] = prepared["teacher_input_text"]
        out["target_start_offset"] = prepared["target_start_offset"]
        out["target_end_offset"] = prepared["target_end_offset"]
        out["long_context_truncation"] = prepared["truncation_metadata"]
        out["raw_bytes"] = prepared["teacher_input_bytes"]
        return out

    def _extract_text(self, record: dict[str, Any]) -> str:
        explicit_text = record.get("teacher_input_text")
        if isinstance(explicit_text, str):
            return explicit_text

        explicit_bytes = record.get("teacher_input_bytes")
        if isinstance(explicit_bytes, bytes):
            return explicit_bytes.decode("utf-8", errors="replace")

        raw = record.get("raw_bytes", b"")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
        return str(raw)

    def _extract_completion_logprobs(self, body: Any) -> tuple[list[int], list[dict[str, Any] | None]]:
        if not isinstance(body, dict):
            raise RuntimeError("llama.cpp top-k inference failed: expected JSON object response.")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp top-k inference failed: /v1/completions response missing choices[0].")

        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("llama.cpp top-k inference failed: /v1/completions choices[0] must be an object.")

        logprobs = choice0.get("logprobs")
        if not isinstance(logprobs, dict):
            raise RuntimeError("llama.cpp top-k inference failed: /v1/completions must include choices[0].logprobs.")

        token_ids_raw = self._extract_from_mapping(logprobs, _TOKEN_IDS_FIELDS)
        if not isinstance(token_ids_raw, list) or not token_ids_raw:
            raise RuntimeError(
                "llama.cpp top-k inference failed: /v1/completions logprobs must expose token ids "
                f"via one of {_TOKEN_IDS_FIELDS}."
            )

        token_ids: list[int] = []
        for tok in token_ids_raw:
            try:
                token_ids.append(int(tok))
            except Exception as exc:
                raise RuntimeError("llama.cpp top-k inference failed: token_ids entries must be integer-like.") from exc

        top_logprobs_raw = self._extract_from_mapping(logprobs, _TOP_LOGPROBS_FIELDS)
        if not isinstance(top_logprobs_raw, list):
            raise RuntimeError(
                "llama.cpp top-k inference failed: /v1/completions logprobs must expose per-position top logprobs "
                f"via one of {_TOP_LOGPROBS_FIELDS}."
            )

        return token_ids, top_logprobs_raw

    def _extract_row_ids_logprobs(self, row: dict[str, Any], top_k: int) -> tuple[list[int], list[float]]:
        pairs: list[tuple[int, float]] = []
        for key, value in row.items():
            try:
                token_id = int(key)
                logprob = float(value)
            except Exception:
                continue
            if not math.isfinite(logprob):
                continue
            pairs.append((token_id, logprob))

        if not pairs:
            raise RuntimeError(
                "llama.cpp top-k inference failed: top_logprobs entries must provide numeric token-id keys. "
                "String token-only maps are not supported for distillation ids."
            )

        pairs.sort(key=lambda item: item[1], reverse=True)
        pairs = pairs[: max(1, int(top_k))]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def _pooled_entropy(self, per_pos_logprobs: list[list[float]]) -> float:
        entropies: list[float] = []
        for row in per_pos_logprobs:
            if not row:
                continue
            max_lp = max(row)
            exps = [math.exp(lp - max_lp) for lp in row]
            z = sum(exps)
            if z <= 0:
                continue
            probs = [v / z for v in exps]
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
            entropies.append(entropy)
        if not entropies:
            return 0.0
        return float(sum(entropies) / len(entropies))

    def _infer_topk_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt_text = self._extract_text(self._prepare_stage_b_record(record))
        top_k = int(record.get("top_k", self.default_top_k))
        if top_k < 1:
            raise ValueError("llama.cpp top-k inference misconfiguration: top_k must be >= 1.")

        payload = {
            "model": self.model_hint,
            "prompt": prompt_text,
            "max_tokens": 0,
            "temperature": float(record.get("temperature", self.default_temperature)),
            "logprobs": int(top_k),
            "echo": True,
        }
        if payload["model"] is None:
            payload.pop("model")

        _, body = self._http_json("POST", "/v1/completions", payload=payload)
        token_ids, per_position_maps = self._extract_completion_logprobs(body)

        per_pos_ids: list[list[int]] = []
        per_pos_lps: list[list[float]] = []
        for row in per_position_maps:
            if row is None:
                continue
            if not isinstance(row, dict):
                raise RuntimeError("llama.cpp top-k inference failed: per-position top_logprobs row must be an object or null.")
            ids, lps = self._extract_row_ids_logprobs(row, top_k=top_k)
            per_pos_ids.append(ids)
            per_pos_lps.append(lps)

        token_length = len(token_ids)
        expected_max_positions = max(token_length - 1, 0)
        if len(per_pos_ids) > expected_max_positions:
            raise RuntimeError(
                "llama.cpp top-k inference failed: prompt logprob rows exceed expected token positions "
                f"({len(per_pos_ids)} > {expected_max_positions})."
            )

        return {
            "top_k_ids": per_pos_ids,
            "top_k_logprobs": per_pos_lps,
            "entropy": self._pooled_entropy(per_pos_lps),
            "teacher_input_token_length": token_length,
            "teacher_input_byte_length": len(prompt_text.encode("utf-8", errors="replace")),
        }

    def _extract_token_count(self, body: Any) -> int | None:
        if isinstance(body, dict):
            for key in ("count", "n_tokens", "token_count"):
                value = body.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        pass
            for key in ("tokens", "token_ids"):
                value = body.get(key)
                if isinstance(value, list):
                    return len(value)
        if isinstance(body, list):
            return len(body)
        return None

    def _tokenize_via_endpoint(self, text: str) -> int:
        payload_variants = (
            {"content": text},
            {"text": text},
            {"prompt": text},
        )
        errors_seen: list[str] = []
        for endpoint in _TOKENIZE_ENDPOINTS:
            for payload in payload_variants:
                try:
                    _, body = self._http_json("POST", endpoint, payload=payload)
                except Exception as exc:
                    errors_seen.append(f"{endpoint}:{exc}")
                    continue
                token_count = self._extract_token_count(body)
                if token_count is not None:
                    return min(max(0, int(token_count)), self.max_context)
                errors_seen.append(f"{endpoint}:unrecognized response shape")

        raise NotImplementedError(
            "llama.cpp tokenization diagnostics are unsupported by this server deployment. "
            "Expected a working /tokenize or /v1/tokenize endpoint returning token ids/count. "
            f"Attempts: {'; '.join(errors_seen[:6])}"
        )

    def token_lengths(self, texts: list[str]) -> list[int]:
        lengths: list[int] = []
        for text in texts:
            lengths.append(self._tokenize_via_endpoint(str(text)))
        return lengths

    def prepare(self) -> None:
        self.startup_self_check()
        self._prepared = True

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._prepared:
            raise RuntimeError("Teacher must be prepared before inference.")

        outputs: list[dict[str, Any]] = []
        for record in records:
            outputs.append(self._infer_topk_for_record(record))
        return outputs

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError("Structured outputs are not implemented for llama.cpp server teacher.")

    def tokenizer_diagnostics(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared_records = [self._prepare_stage_b_record(r) for r in records]
        texts = [self._extract_text(r) for r in prepared_records]
        if not self.supports_tokenizer_diagnostics():
            detail = self._tokenizer_diagnostics_detail or "tokenization endpoint unsupported"
            raise NotImplementedError(
                f"Tokenizer diagnostics are unsupported for this llama.cpp server: {detail}"
            )
        lengths = self.token_lengths(texts)
        out: list[dict[str, Any]] = []
        for text, token_length in zip(texts, lengths):
            out.append(
                {
                    "teacher_input_token_length": int(token_length),
                    "teacher_input_byte_length": len(text.encode("utf-8", errors="replace")),
                }
            )
        return out

    def close(self) -> None:
        self._prepared = False
