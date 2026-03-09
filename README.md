# distill-factory

`distill-factory` is a portable Python repository focused on **teacher-data generation** and **dataset building** for distillation workflows.

This is **not** a model-training repository.

## Pipeline stages

- **Stage A**: bulk grounding teacher
- **Stage B**: long-context structure teacher
- **Stage C**: refinement teacher

## Reusable source-cache + corpus-mixture workflow

The repository now supports a **two-layer corpus design** so dataset extraction is decoupled from final corpus composition:

1. **Source extraction layer** (`[source_extraction]`)
   - define each reusable source dataset once (`source_name`, `source_type`, Hugging Face dataset/config, text field, split mapping, grouping controls)
   - extract and normalize into a persistent cache directory (`cache_dir`)
   - optionally cap extraction volume per split (`max_docs_per_split`)
   - optional source byte filters at extraction time (`min_bytes`, `max_bytes`)

2. **Mixture build layer** (`[mixture_build]`)
   - define semantic groups (for example: `code`, `general_knowledge`, `narrative`)
   - assign cached datasets to each group
   - set a percentage per group and a required `target_documents`
   - set `random_seed` for reproducible sampling/remixing

This lets you cache expensive source downloads/extraction once, then rebuild many corpus variants without re-downloading or re-extracting.

Schema entry points:
- `distill_factory/corpus/schema.py`
- `distill_factory/corpus/__init__.py`
- `distill_factory/config/schema.py` (re-exported schema types/loader)

Example config:
- `configs/examples/corpus_mixture.toml`

Validation guarantees in the corpus-mixture schema:
- `target_documents` is required and must be `> 0`
- all group percentages must sum to exactly `100`
- mixture groups can only reference datasets declared in the source cache section

## Source dataset extraction cache layout

Source extraction now supports deterministic cached document materialization under `data/sources/<source_name>/`.

Output layout:

```text
data/sources/<source_name>/
  train/
  eval/
  validation/
  manifest.json
```

Each split stores grouped text docs and sidecar metadata:

- `doc_00000001.txt`
- `doc_00000001.meta.json`

Sidecar metadata includes provenance and reproducibility fields such as:
`source_name`, `source_type`, `hf_dataset`, `hf_config`, `split`, `upstream_split`, `row_ordinal_start`, `row_ordinal_end`, `text_field`, `group_size`, and extraction timestamp.

Current extractor scope:
- Hugging Face streaming datasets (`source_type = "huggingface"`)
- deterministic row grouping by `group_size`
- split-specific extraction with clear failure when a requested split mapping cannot be loaded
- manifest-based cache safety for reuse across many future mixture builds

`manifest.json` is now the cache contract for reuse and includes at least:
- source identity/type (`source_name`, `source_type`)
- dataset identifier/config (`hf_dataset`, `hf_config`)
- split mapping used
- text extraction settings (`text_field`, `group_size`, optional `max_docs_per_split`)
- `extracted_doc_counts` per split
- extraction timestamp
- config fingerprint (`config_fingerprint`)

Re-run behavior:
- if existing cache fingerprint matches config, extraction is skipped (safe reuse)
- if fingerprint differs, extraction fails clearly unless `--refresh` / `--overwrite` is set
- if `manifest.json` is missing in an existing cache dir, extraction fails clearly unless refresh is explicit

Run extraction for one source from config:

```bash
python scripts/extract_source_dataset.py \
  --config configs/examples/corpus_mixture.toml \
  --source-name the_stack_python \
  --cache-root data/sources
```

Inspect planned action without extracting:

```bash
python scripts/extract_source_dataset.py \
  --config configs/examples/corpus_mixture.toml \
  --source-name the_stack_python \
  --cache-root data/sources \
  --dry-run
```

Force overwrite/refresh when config changes:

```bash
python scripts/extract_source_dataset.py \
  --config configs/examples/corpus_mixture.toml \
  --source-name the_stack_python \
  --cache-root data/sources \
  --refresh
```

## Build mixed corpora from cached sources

After sources are cached under `data/sources/<source_name>/...`, you can build new corpora without re-downloading or re-extracting datasets.

Mixture builder characteristics:
- consumes cached docs only (no Hugging Face download/extraction path)
- supports config-defined groups/datasets, percentage shares, `target_documents`, and `random_seed`
- validates group percentages sum to `100`
- deterministic sampling per split (`train`/`eval`/`validation`) from cached docs
- writes per-output-doc provenance sidecars with source dataset/split/group and original cached doc path/id
- records requested vs realized composition in output `manifest.json` for auditing and future remixes

Output layout:

```text
data/corpora/<mixture_name>/
  train/
  eval/
  validation/
  manifest.json
```

Group expression in config (explicit and auditable):

```toml
[mixture_build]
target_documents = 1000000
random_seed = 42
min_bytes = 128
max_bytes = 180000
depletion_policy = "rebalance" # "rebalance" | "strict" | "record_only"

[[mixture_build.groups]]
group_name = "code"
percentage = 35
dataset_names = ["the_stack_python"]

[[mixture_build.groups]]
group_name = "general_knowledge"
percentage = 40
dataset_names = ["wikipedia_en"]

[[mixture_build.groups]]
group_name = "narrative"
percentage = 25
dataset_names = ["bookcorpus"]
```

Filtering order (explicit):
1. **Source extraction filtering** (`source_extraction.datasets[].min_bytes/max_bytes`) is applied first, before grouped docs are written to `data/sources/...`.
2. **Mixture filtering** (`mixture_build.min_bytes/max_bytes`) is applied later, when selecting cached docs for `data/corpora/...`.

This keeps filtering deterministic and auditable at both layers.

Validation behavior:
- group percentages must sum to exactly `100`
- each `dataset_names` entry must exist in `[source_extraction].datasets`
- `target_documents` must be `> 0`
- split availability is checked at build time for each referenced cached source split

Dataset overlap across groups:
- overlap is allowed intentionally (a dataset may appear in multiple groups), but this should be done deliberately because it changes sampling behavior and effective source weighting.

Output manifest (`data/corpora/<mixture_name>/manifest.json`) includes at least:
- `mixture_name`
- `target_documents`
- `random_seed`
- requested group percentages (`requested_group_percentages`)
- configured depletion policy (`depletion_policy`)
- realized document counts by group (`realized_document_counts_by_group`)
- realized document counts by source dataset (`realized_document_counts_by_source_dataset`)
- realized split counts (`realized_split_counts`)
- applied mixture byte filters (`mixture_min_bytes`, `mixture_max_bytes`)
- source cache manifests/references used (`source_cache_references`)
- split-level filtered-doc counters when filters apply (`filtered_below_min_bytes`, `filtered_above_max_bytes`)
- creation timestamp (`created_at`)

Deterministic sampling inputs:
- mixture config (group percentages, group datasets, target documents)
- `random_seed`
- cached source file contents/layout (`doc_*.txt` under each split)

Depletion policy is configurable via `mixture_build.depletion_policy`:
- `"rebalance"` (default):
  - sample each group to its requested share first
  - deterministically fill shortfall from remaining groups when possible
  - if total cache is still insufficient, record explicit shortfall warnings
- `"strict"`:
  - fail fast if any split/group cannot satisfy requested proportions
  - no partial build output is produced
- `"record_only"`:
  - do not rebalance across groups
  - build only what each group can provide and record explicit shortfall

Determinism is preserved under all policies (config + seed + cached docs determine outcomes).

If cached source data is insufficient for requested shares, the builder does **not** silently hide this:
- manifest records configured `depletion_policy`
- manifest sets `composition_deviation = true` when realized output is short
- manifest stores explicit warning entries in `warnings`
- manifest includes requested vs realized totals and per-split realized counts
- CLI prints warnings after build so deviations are visible immediately

Build command:

```bash
python scripts/build_text_corpus.py \
  --config configs/examples/corpus_mixture.toml \
  --mixture-name base_mix_v1 \
  --cache-root data/sources \
  --output-root data/corpora
```

Dry-run (path/info only):

```bash
python scripts/build_text_corpus.py \
  --config configs/examples/corpus_mixture.toml \
  --mixture-name base_mix_v1 \
  --dry-run
```

## Inspect cached sources and built corpora

Use the terminal inspector to quickly sanity-check either:
- a cached source dataset directory (`data/sources/<source_name>/`), or
- a built mixed corpus directory (`data/corpora/<mixture_name>/`).

Inspector command:

```bash
python scripts/inspect_text_corpus.py data/sources/the_stack_python
python scripts/inspect_text_corpus.py data/corpora/base_mix_v1
```

What it prints:
- manifest summary (core fields + key list)
- split document counts (`train`/`eval`/`validation`)
- document counts by `source_name` (if sidecar metadata has it)
- document counts by `group_name` (if sidecar metadata has it)
- one sample text document preview + its metadata sidecar

Optional text preview length:

```bash
python scripts/inspect_text_corpus.py data/corpora/base_mix_v1 --preview-chars 800
```

## End-to-end: from reusable corpus cache to distillation pipeline

You can connect the new corpus cache/mixture workflow directly to the existing distillation pipeline with this sequence:

1. **Extract source datasets once** (cached under `data/sources/<source_name>/...`):

```bash
python scripts/extract_source_dataset.py   --config configs/examples/corpus_mixture.toml   --source-name the_stack_python   --cache-root data/sources
```

2. **Build a mixed corpus** (materialized under `data/corpora/<mixture_name>/...`):

```bash
python scripts/build_text_corpus.py   --config configs/examples/corpus_mixture.toml   --mixture-name base_mix_v1   --cache-root data/sources   --output-root data/corpora
```

3. **Point the distillation pipeline input at the built corpus split** (typically `train/`):
   - `input_path = "data/corpora/base_mix_v1/train"`
   - `file_glob = "*.txt"`

4. **Run existing pipeline tooling**.

Example: Stage A validation against a built corpus:

```bash
python scripts/validate_stage_a.py --config configs/examples/default.toml
```

Example: dataset build entrypoint against the same corpus input:

```bash
python scripts/build_dataset.py --config configs/examples/default.toml
```

This keeps source extraction reusable and lets you regenerate new corpus mixtures while reusing existing distillation pipeline entrypoints unchanged.

## Current status

- Repository scaffolding and lightweight data pipeline modules are included.
- Real Stage A backends exist:
  - `hf_causal_lm` (Transformers + Torch)
  - `vllm_causal_lm` (vLLM)
- vLLM is top-k distillation only in this repository (no structured output generation).
- Stage C structured generation backends are intentionally not implemented yet.

## Teacher capability checks (fail-fast)

Stage runners validate teacher capabilities before expensive generation begins.

Capability flags used by the pipeline:

- `supports_topk`
- `supports_structured`
- `supports_hidden_summary`
- `supports_long_context`
- `supports_per_token_entropy`
- `supports_per_token_top1_gap`

Stage requirements:

- Stage A requires `supports_topk`.
- Stage B requires `supports_topk` and `supports_long_context`.
- Stage C in `structured_outputs` mode requires `supports_structured`.
- Hidden-summary extraction requires `supports_hidden_summary` in the selected stage mode.
- Stage A selection-aware export modes (`position_mask` / `selected_windows`) require per-token signal capabilities when corresponding thresholds are configured:
  - entropy threshold => `supports_per_token_entropy`
  - top1-gap threshold => `supports_per_token_top1_gap`
- Stage B currently does **not** implement selection-aware export modes; requesting them fails fast with an explicit error.

If a teacher does not satisfy required capabilities, the run fails early with a clear error that includes teacher name, stage, mode, and missing capabilities.

## Stage A backend selection

Choose backend via config:

```toml
[stage_a]
teacher_name = "hf_causal_lm"     # or "vllm_causal_lm" / "llamacpp_server"
backend_type = "hf"               # or "vllm" / "llamacpp_server"
mode = "topk_logits"
```

For complete examples, start from `configs/examples/default.toml` or one of the backend-specific examples in `configs/examples/*_backend.toml`.

### Optional position-aware selection policy config (schema support)

Stage A and Stage B can now declare position-aware export selection policy fields (kept conservative/disabled by default):

```toml
[stage_a]
enable_position_filtering = false
entropy_threshold = 1.5
top1_gap_threshold = 0.2
selection_window_radius = 0
selection_mode = "none" # "none" | "position_mask" | "selected_windows"
minimum_selected_positions_per_record = 1
```

The same fields are available in `[stage_b]` for schema compatibility.

Current Stage B status: selection fields are parsed/validated, but Stage-B-specific selection export (`position_mask` / `selected_windows`) is not implemented and now fails fast when requested.

Validation rules:

- if `enable_position_filtering = true`, `selection_mode` must not be `"none"`
- if `enable_position_filtering = true`, at least one threshold must be set (`entropy_threshold` and/or `top1_gap_threshold`)
- `entropy_threshold >= 0` and `top1_gap_threshold >= 0` when provided
- `selection_window_radius >= 0`
- `minimum_selected_positions_per_record >= 0` when provided

### Position-selection utility semantics (reusable layer)

Core pure helpers now live in `distill_factory/data/selection.py` for export/filtering pipelines:

- entropy selection: keep positions where `per_token_entropy >= entropy_threshold`
- gap selection: keep positions where `per_token_top1_gap <= top1_gap_threshold`
- combined selection default: **union** (OR); intersection (AND) is explicitly supported
- selected positions can be expanded into fixed-radius windows and overlapping windows are merged
- optional minimum-selected-position enforcement is available for sparse selections

These utilities are intentionally pure (no I/O), so they can be reused by future export/filtering steps and tested independently.

### Stage A selected-window export mode (first conservative implementation)

When `enable_position_filtering = true`, Stage A supports two explicit export modes:

- `selection_mode = "position_mask"`
  - keeps dense full records/chunks unchanged
  - computes informative positions from per-token signals
  - attaches reusable selection metadata in `extra_metadata`:
    - `selected_position_mask`
    - `selected_positions`
    - `selected_position_count`
    - `selection_policy`

- `selection_mode = "selected_windows"`
  - builds the same position mask from available per-position teacher signals:
    - entropy rule: `per_token_entropy >= entropy_threshold`
    - gap rule: `per_token_top1_gap <= top1_gap_threshold`
  - if both thresholds are set, the default combined rule is **union**
  - expands selected positions by `selection_window_radius` and merges overlapping windows
  - emits only those selected local windows (instead of one dense full-position record)
  - attaches window metadata in `extra_metadata`:
    - `selected_window_start`
    - `selected_window_end`
    - `selected_position_count`
    - `selection_policy`

Dense behavior remains unchanged by default (selection disabled).

> Stage B note: selection-aware export modes are currently unsupported in Stage B and fail fast if requested.

Quick policy sanity-check before expensive runs:

```bash
python scripts/preview_selection_policy.py   --config configs/examples/default.toml   --sample-records 4
```

This diagnostic runs a tiny Stage A teacher pass and prints:

- selected position counts per sampled record
- selected window ranges
- average compression ratio vs dense chunks
- whether selection is driven by entropy, gap, or both

Use it to quickly detect policies that are too aggressive (over-pruning) or too weak before full dataset generation.

Tiny end-to-end Stage A selection harness (normal pipeline path, Stage A only):

```bash
python scripts/validate_stage_a_selection.py --config configs/examples/default.toml --mode selected_windows --records 8
```


## Teacher abstraction vs runtime backend

The pipeline now separates two concepts explicitly:

- **Teacher abstraction**: stage-level semantics used by dataset construction (top-k distillation, structured outputs, long-context behavior).
- **Runtime backend**: execution transport/engine that actually serves model inference.

This allows one teacher contract to run on different runtime types:

- `hf` (local Python Hugging Face runtime)
- `vllm` (local Python vLLM runtime)
- `llamacpp_server` (external server runtime, reserved for integrations)

Each stage config can set `backend_type` independently from `teacher_name`.

Example:

```toml
[stage_a]
teacher_name = "hf_causal_lm"
backend_type = "hf"
mode = "topk_logits"

[stage_b]
teacher_name = "vllm_causal_lm"
backend_type = "vllm"
mode = "long_context"
```

If `backend_type` is omitted for known built-in teachers (`hf_causal_lm`, `vllm_causal_lm`), it is inferred automatically for backward compatibility.



## Choosing a runtime backend (HF vs vLLM vs llama.cpp server)

Why explicit runtime backends now:

- backend choice is now a **first-class config decision** (`backend_type`) instead of implicit behavior tied only to teacher naming
- this makes runs reproducible, easier to audit, and safer to migrate across environments
- it also keeps teacher semantics stable while letting you swap execution runtimes

`llama.cpp` is separate from `vllm`:

- `vllm` is a Python runtime backend inside this process
- `llamacpp_server` is an **external HTTP runtime** (user-managed server process)
- choose `llamacpp_server` if you want to use a custom local llama.cpp build (compiler flags, quant kernels, patches) without changing pipeline code

When to choose each backend:

- **HF (`backend_type = "hf"`)**
  - best for widest compatibility and structured Stage C support
  - good default for bring-up and CPU/single-GPU experimentation
- **vLLM (`backend_type = "vllm"`)**
  - best for high-throughput top-k distillation on supported GPU setups
  - Stage C structured mode is not supported
- **llama.cpp server (`backend_type = "llamacpp_server"`)**
  - best when you operate a separate llama.cpp server, especially with custom local builds
  - supports top-k distillation subset via server API; unsupported features fail early and explicitly

Ready-to-use backend example configs:

- `configs/examples/hf_backend.toml`
- `configs/examples/vllm_backend.toml`
- `configs/examples/llamacpp_server_backend.toml`

Migration notes (from earlier implicit backend structure):

1. Set `backend_type` explicitly in each enabled stage (`stage_a`, `stage_b`, `stage_c`).
2. Use a teacher/runtime pair that matches:
   - `hf_causal_lm` + `hf`
   - `vllm_causal_lm` + `vllm`
   - `llamacpp_server` + `llamacpp_server`
3. If you previously relied on `teacher_name` alone, keep the same teacher but add `backend_type` for clarity and future-proofing.
4. For llama.cpp runs, move runtime details into config (`llama_base_url`, optional `llama_model_hint`, timeout) and treat the server as an external dependency.

## Backend capability matrix (current)

| Backend | supports_topk | supports_structured | supports_hidden_summary | supports_long_context | supports_per_token_entropy | supports_per_token_top1_gap | supports_tokenizer_diagnostics |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hf` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `vllm` | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `llamacpp_server` | ✅ | ❌ | ❌ | ✅ | ✅* | ✅* | ⚠️ depends on server tokenization endpoints |

Notes:
- Stage runners fail early if a requested feature is unsupported (for example structured mode in Stage C, hidden summaries, per-token selection signals, or token diagnostics).
- For `llamacpp_server`, tokenizer diagnostics require compatible `/tokenize` or `/v1/tokenize` support.
- `llamacpp_server` per-token signal support (`✅*`) depends on response shape and can fail explicitly for malformed/insufficient top-logprob rows.

## Hugging Face teacher setup (Stage A)

Install runtime dependencies:

```bash
pip install transformers torch
```

HF-related stage A config fields:

- `model_name_or_path`
- `device_map`
- `torch_dtype`
- `max_context`
- `batch_size`
- `emit_per_token_entropy` (optional, default `false`)
- `emit_per_token_top1_gap` (optional, default `false`)

When enabled, HF `infer_topk` emits per-position arrays aligned to the same effective prompt positions as `top_k_ids`/`top_k_logprobs`.
`per_token_top1_gap` is defined as `logprob(top1) - logprob(top2)` at each position; if only one candidate is requested (`top_k = 1`), the emitted gap is `0.0` by convention.

Policy note: `raw_bytes` are decoded as UTF-8 with replacement (`errors="replace"`) before tokenization.

## vLLM teacher setup (Stage A)

Install runtime dependency:

```bash
pip install vllm
```

vLLM-related stage A config fields:

- `model_name_or_path`
- `tensor_parallel_size`
- `dtype`
- `max_context`
- `batch_size`
- `gpu_memory_utilization`
- `trust_remote_code`
- `emit_per_token_entropy` (optional, default `false`)
- `emit_per_token_top1_gap` (optional, default `false`)

Current vLLM backend behavior mirrors HF output semantics as closely as practical:

- emits `top_k_ids` and `top_k_logprobs` per token position
- emits pooled mean entropy per record
- when enabled, emits per-position arrays aligned with emitted vLLM prompt-logprob rows:
  - `per_token_entropy`
  - `per_token_top1_gap`
- `per_token_top1_gap` uses `logprob(top1) - logprob(top2)` when two candidates are available; if a row only has one candidate, gap is `0.0` by convention
- decodes `raw_bytes` as UTF-8 with replacement before tokenization

vLLM alignment note vs HF:

- vLLM prompt logprobs may omit some prompt positions depending on runtime behavior, so per-position array lengths can be `<= teacher_input_token_length - 1`.
- Entropy for vLLM per-position/pooled outputs is computed from the returned top-k candidates (renormalized over available mass), not the full vocabulary distribution.




## llama.cpp server backend (external runtime)

You can run distillation against a **user-managed llama.cpp HTTP server** instead of a local Python runtime.

Why this backend:

- treat llama.cpp as an external runtime backend (`backend_type = "llamacpp_server"`)
- reuse locally compiled llama.cpp optimizations/patches without changing Python runtime code
- keep teacher semantics stable while swapping execution transport

Example stage config:

```toml
[stage_a]
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "topk_logits"

# llama.cpp server settings
llama_base_url = "http://127.0.0.1:8080"
llama_model_hint = "qwen2.5-7b-instruct-q4_k_m" # optional
llama_request_timeout = 30.0

# existing stage knobs still apply
top_k = 5
temperature = 0.0
max_context = 2048
```

Startup check configuration example (minimal preflight):

```toml
[stage_a]
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "topk_logits"
llama_base_url = "http://127.0.0.1:8080"
llama_request_timeout = 10.0
max_context = 2048
top_k = 5
temperature = 0.0
```

Current status:

- startup/self-check is implemented via HTTP probes to common endpoints and metadata discovery
- if endpoint metadata is missing, the backend keeps minimal safe assumptions and reports what it did discover
- top-k extraction is implemented for OpenAI-compatible `/v1/completions` prompt logprobs responses
- optional per-position outputs are supported when explicitly enabled (`emit_per_token_entropy`, `emit_per_token_top1_gap`) and when response rows include enough numeric candidates
- `per_token_top1_gap` is defined as `logprob(top1) - logprob(top2)` and fails explicitly if any emitted row has fewer than 2 candidates
- the backend requires numeric token-id keys in `top_logprobs` maps to produce `top_k_ids`; if unavailable, it fails explicitly
- structured generation is not implemented for this backend in the current prompt
- hidden-summary extraction is not supported for this backend

Compared with HF/vLLM:

- llama.cpp per-position outputs depend on server response shape; unsupported/malformed rows fail explicitly instead of using fallback values.
- entropy is computed from returned top-k candidates (renormalized over available mass), not full-vocabulary logits.

Note: endpoint support varies across llama.cpp versions/builds; adjust the endpoint catalog and response field adapters in `distill_factory/teachers/llamacpp_server.py` to match your deployment.

## Teacher startup self-checks (preflight)

Before ingestion/chunking and stage execution, the orchestrator runs a fast teacher preflight check for each enabled stage/teacher combination.

The self-check validates:

- model path/name is present
- requested `top_k` is valid (`>= 1`)
- requested context length is sensible (`max_context >= 1`)
- required backend libraries are importable (`transformers`/`torch` for HF, `vllm` for vLLM)
- backend initialization succeeds (fast `prepare`/`close` cycle)

If any check fails, the pipeline exits early with a readable stage+teacher error so misconfiguration is caught before long generation runs.

## Optional byte/token length diagnostics

You can optionally log aggregated teacher-input length diagnostics to estimate tokenization cost and long-context feasibility.

Output config fields:

- `log_token_lengths` (`true`/`false`)
- `log_byte_lengths` (`true`/`false`)

When enabled, stage metrics include aggregated fields such as:

- average teacher input byte length
- average teacher input token length
- min/max teacher input token length
- average byte-to-token ratio

These are aggregated only (not per-sample dumps), and help answer questions like:

- whether your corpus tokenizes densely/sparsely for a given tokenizer
- how close Stage B windows are to context limits
- whether a teacher backend/configuration is likely to be cost-feasible

## Teacher sanity metrics (why entropy matters)

During Stage A and Stage B, the pipeline logs simple teacher-output sanity metrics:

- average entropy
- min/max entropy
- average chunk length
- average top-k width actually populated
- number of records per teacher
- number of records per stage

Entropy is a practical quality signal for distillation:

- very low entropy everywhere can indicate overconfident/peaky teacher outputs
- very high entropy everywhere can indicate noisy or uninformative supervision
- tracking entropy ranges helps compare teacher backends/settings and catch regressions early

The dataset inspector also prints these metrics from stored records, and prints embedded stage summaries when present.

## Tokenization cost quick-check

Estimate tokenizer context cost on a small sample before large runs:

```bash
python scripts/measure_tokenization_cost.py --config configs/examples/default.toml --backend hf --sample-records 32
```

Use `--backend vllm` to estimate using the vLLM teacher tokenizer path.

Use `--backend llamacpp_server` to estimate using a user-managed llama.cpp server **when tokenization endpoints are available** (`/tokenize` or `/v1/tokenize`).
If those endpoints are not exposed by your server version/build, the command exits with an explicit unsupported-diagnostics error.

## Build preflight checklist convenience

Before non-dry-run builds, `scripts/build_dataset.py` now prints a short preflight checklist
(config path, enabled stages/teachers, output/resume/shard settings, dry-run/stop-after-stage,
and longdoc-eval status) to reduce accidental mis-launches on large runs.

## Pipeline benchmark (small validation run)

Use this benchmark script to measure where time is spent (ingestion/chunking/splitting/stages/writing):

```bash
python scripts/benchmark_pipeline.py --config configs/examples/default.toml --records 16
```

## Stage A bring-up

Run a tiny, reproducible Stage A validation before large jobs:

```bash
python scripts/validate_stage_a.py --config configs/examples/default.toml --records 16
```

## Inspecting generated datasets

Use the inspection CLI to quickly sanity-check output JSONL files:

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl
```

The inspector prints:

- schema summary (versions + observed fields)
- selection-aware export indicators when present:
  - count of selected-window records
  - average selected window length
  - average selected position count
  - whether position masks are present
  - proportion of records using selection filtering

Dense vs selection-aware exports:

- **Dense export**: full records with no selection markers in `extra_metadata`.
- **Position-mask export**: full records retained, with `selected_position_mask` / `selected_positions` metadata for later weighting/post-processing.
- **Selected-windows export**: records represent selected local windows and include `selected_window_start` / `selected_window_end` metadata.

Canonical JSONL records are backward-compatible: older records without optional per-token fields remain valid. When present, `per_token_entropy` and `per_token_top1_gap` can be used by downstream selection/filtering steps, and `per_token_token_ids` / `per_token_valid_mask` provide alignment helpers for position-level semantics.
- counts by stage
- counts by teacher
- average chunk length in bytes
- whether top-k fields and structured fields exist
- one sample record

You can also inspect eval output:

```bash
python scripts/inspect_dataset.py --path data/processed/eval.jsonl
```


Export a few representative records for manual inspection (with optional stage/teacher filters):

```bash
python scripts/inspect_dataset.py \
  --path data/processed/train.jsonl \
  --sample-count 5 \
  --sample-stage stage_b \
  --sample-teacher hf_causal_lm \
  --export-samples data/processed/sample_preview.txt \
  --sample-format text
```

Use `--sample-format json` to write a JSON preview file instead.


## Backend parity sanity check (recommended)

When introducing or switching a backend (especially `llamacpp_server`), run a tiny parity sanity check before large dataset builds:

```bash
python scripts/validate_backend_parity.py   --left-config configs/examples/hf_backend.toml   --right-config configs/examples/llamacpp_server_backend.toml   --sample-records 8
```

For selection-aware Stage A runs (`position_mask` / `selected_windows`), run this check before switching backends. The tool now prints PASS/WARN/FAIL structural signals for:
- per-token field presence (`per_token_entropy`, `per_token_top1_gap`)
- plausible per-token lengths relative to top-k position count
- selection-metadata feasibility when selection mode is enabled

It intentionally does **not** require exact numeric equality across backends.

What it compares (without requiring exact logit equality):

- record count
- top-k field presence
- token-length stats (when available)
- entropy ranges (when available)

Use this as a quick compatibility/sanity signal, not as a performance benchmark or numerical-equivalence test.

## Comparing two dataset runs

Use the comparison utility to quickly spot run-to-run differences:

```bash
python scripts/compare_datasets.py --left data/run_a --right data/run_b
```

The comparison prints terminal-friendly diffs for:

- total record count
- record count by stage
- record count by teacher
- shard count
- average chunk length
- average entropy (if present)
- manifest metadata differences
- schema version differences

This is intended as a lightweight change check (not deep statistical analysis).

## Merging dataset shards

Use the merge utility to consolidate a sharded dataset into a new stable layout:

```bash
python scripts/merge_dataset_shards.py \
  --input-dir data/processed \
  --output-dir data/processed_merged
```

Optional re-sharding with a new deterministic shard size:

```bash
python scripts/merge_dataset_shards.py \
  --input-dir data/processed \
  --output-dir data/processed_resharded \
  --max-records-per-shard 50000 \
  --shard-prefix shard
```

Behavior:

- Reads shard files in deterministic sorted order.
- Preserves record lines exactly (no JSON rewriting by default).
- Does not apply extra deduplication during merge.
- Regenerates `dataset_manifest.json` in the output directory and preserves existing manifest metadata where available.

## Stop-after-stage controls (phased generation)

You can stop pipeline execution after a specific stage without editing code.

Output config field:

- `stop_after_stage` = `null` | `"stage_a"` | `"stage_b"` | `"stage_c"`

Behavior:

- Enabled stages still execute in order.
- The run stops cleanly once the requested stage completes.
- Outputs, manifests, metrics exports, and resume state are still written for the completed portion.

This is useful for phased generation, debugging intermediate supervision, and inspecting stage-specific outputs before running later stages.

## Dry-run mode (pipeline wiring validation)

Use dry-run mode to validate end-to-end pipeline wiring before spending GPU time.

Output config fields:

- `dry_run` (`true`/`false`)
- `dry_run_max_records` (int)

Behavior in dry-run mode:

- ingestion, chunking, split logic, long-context assembly (Stage B), prompt templating (Stage C), and storage all run
- teacher inference is skipped and replaced with deterministic placeholder outputs
- records include `extra_metadata.dry_run = true` and a `dry_run_note`
- applies to Stage A, Stage B, and Stage C
- only up to `dry_run_max_records` records per split are processed to keep runs cheap

Use this mode to check dataset shape, stage routing, prompt formatting, and output writing quickly.

## Failure logging and skipped-record tracking

The pipeline writes record-level failures to `<output_dir>/record_failures.jsonl` so failures are auditable instead of silently disappearing.

Each failure entry includes at least:

- `timestamp`
- `stage_name`
- `teacher_name`
- `doc_id` (when available)
- `chunk_index` (when available)
- `error_message`

Behavior:

- **Skipped record**: one record fails stage processing, is logged, and the run continues when safe.
- **Failed run**: a fatal stage-level condition (e.g., all records fail in a stage) aborts the run with a clear error.
- **Resumed run**: run restarts from resume state checkpoints and continues writing output while preserving prior progress.

Manifest includes `skipped_record_count` so you can track skipped totals across writes.

## Resumable pipeline execution

Long-running dataset builds can resume safely after interruption.

Output config fields:

- `resume` (`true`/`false`)
- `resume_policy` (`"strict"` or `"best_effort"`)

Behavior:

- When `resume = false`, pipeline behavior is unchanged.
- When `resume = true`, the pipeline loads `<output_dir>/resume_state.json` and continues from the last safe export point.
- Strict mode fails if the saved config fingerprint differs.
- Best-effort mode allows non-critical config differences (with a warning) but still fails on critical config differences.
- Resume writes only new records and avoids duplicating already written records.

Resume state file (`resume_state.json`) includes:

- `schema_version`
- `config_fingerprint` (+ `critical_config_fingerprint`)
- `completed_stages`
- split progress with completed shard ids and record counts
- `teacher_names`
- `last_update_timestamp`

## Dataset manifest (provenance)

Each dataset output directory now includes a machine-parseable manifest file:

- `dataset_manifest.json`

The manifest is updated when JSONL outputs are written and records:

- `schema_version`
- `creation_timestamp` and `updated_timestamp`
- `config_path` or `config_snapshot` (when provided)
- `enabled_stages`
- `teacher_names`
- `shard_count`
- `total_record_count` (when known)
- `format`, `compression`, and `format_settings`

This helps make dataset generation reproducible and auditable across runs.

`inspect_dataset.py` prints manifest contents (when present):

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl
```

## Stage C prompt templates

Stage C structured-output generation supports plain Python prompt templates (no Jinja):

- `summarize_chunk`
- `answer_question_from_chunk`
- `continue_document`
- `extract_key_points`

Stage C config fields:

- `template_name`
- `template_kwargs` (dict/table)
- `deterministic` (bool)

Example:

```toml
[stage_c]
mode = "structured_outputs"
template_name = "extract_key_points"
template_kwargs = { max_points = 6 }
deterministic = true
```

Template reference example file:

- `configs/examples/stage_c_templates.toml`

Preview a prompt before expensive generation:

```bash
python scripts/preview_prompt.py \
  --template-name summarize_chunk \
  --template-kwargs-json '{"max_words": 60}' \
  --input-text "This is a sample chunk to preview formatting."
```

The preview prints the generated `task_type` and exact `prompt_text` using the same Stage C template registry used by the pipeline.

## Teacher-quality metrics export

The pipeline supports exporting a machine-readable teacher-quality summary JSON:

- `teacher_quality_metrics.json`

Export schema includes:

- `schema_version`
- `record_count`
- `entropy_histogram` (deterministic equal-width bins)
- `average_entropy_per_stage`
- `average_entropy_per_teacher`
- `record_counts_per_stage`
- `record_counts_per_teacher`

Inspection CLI behavior:

- prints metrics export location and whether the file is present
- can print entropy histogram bins in terminal with `--show-histogram`

Example:

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl --show-histogram
```

## Long-document evaluation split (`eval_longdoc`)

For context-sensitive evaluation, you can reserve a dedicated split of longer documents instead of relying only on random held-out docs.

Data config fields:

- `eval_longdoc_min_bytes`
- `eval_longdoc_fraction`
- `eval_split_strategy` = `"random_docs" | "prefer_long_docs"`

Behavior:

- The normal doc-level train/eval split is still produced.
- If `eval_longdoc_fraction > 0`, a deterministic subset of eval documents is reassigned to `eval_longdoc`.
- With `eval_split_strategy = "prefer_long_docs"`, candidate docs meeting `eval_longdoc_min_bytes` are ranked by length and selected first.
- With `eval_split_strategy = "random_docs"`, eligible docs are selected deterministically via seeded shuffle.

Records keep explicit `split` metadata values: `train`, `eval`, or `eval_longdoc`.

## Static curriculum / replay stage mixing

The pipeline supports a **static curriculum/replay mix** across enabled stages for final export (not dynamic online scheduling).

Data config controls:

- `replay_stage_a_fraction`
- `replay_stage_b_fraction`
- `replay_stage_c_fraction`

Optional split-specific overrides:

- `train_replay_stage_a_fraction`
- `train_replay_stage_b_fraction`
- `train_replay_stage_c_fraction`
- `eval_replay_stage_a_fraction`
- `eval_replay_stage_b_fraction`
- `eval_replay_stage_c_fraction`

Behavior:

- When multiple stages are enabled, the orchestrator keeps per-stage record snapshots.
- Final `train` and `eval` exports are built by deterministic subsampling from those stage snapshots according to configured ratios.
- `stage_name` is preserved on all records so replay composition remains auditable.
- If only one stage is enabled, output behavior is unchanged.
