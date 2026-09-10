"""Teacher-generated synthetic tasks, verified by the execution oracle.

The template families (templates.py) are in-distribution but narrow. The
teacher LLM widens the corpus with NOVEL workloads — but a teacher's guess
about its own code's footprint would be exactly the labeling bias this
project exists to remove. So the labels never come from the teacher:

    teacher writes code → AST safety screen → harness pod EXECUTES it
    with generous resources and measures peak RSS + avg CPU (the oracle)
    → only tasks that ran clean, in-bounds, become corpus records.

Mirrors basic-model-factory's oracle-verified synthetic station: only
verifiable samples survive curation.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap

ALLOWED_IMPORTS = {
    "numpy", "pandas", "sklearn", "torch", "time", "math", "random",
    "itertools", "collections", "functools", "json", "string", "statistics",
    # envelope widening (round 12): I/O-phase realism via sandboxed temp
    # files, sparse/columnar data, and the text-mangling real ETL does.
    "tempfile", "io", "csv", "gzip", "scipy", "pyarrow", "re", "array",
    "heapq", "bisect", "datetime",
    # ── round 13: the stack axis ────────────────────────────────────────
    # A corpus that only knows pandas/torch teaches pandas/torch physics.
    # These libraries differ in exactly the dimension the policy learns:
    # Arrow-backed columnar (polars), out-of-core with disk spill
    # (duckdb, dask), partitioned graphs (dask), their own histogram
    # structures (xgboost/lightgbm), separate allocators (jax), and
    # agent/RAG shapes where memory lives in documents, embeddings and
    # vector indexes rather than frames.
    "polars", "duckdb", "dask", "ibis",
    "xgboost", "lightgbm", "statsmodels", "networkx",
    "jax", "jaxlib", "lightning", "transformers", "tokenizers", "safetensors",
    "faiss", "langchain_core", "langchain_text_splitters", "langgraph",
    "llama_index", "pydantic", "typing", "dataclasses", "uuid", "hashlib",
    "textwrap", "operator", "enum",
}
FORBIDDEN_NAMES = {
    # `open` is ALLOWED since round 12: the prompt invites tempfile-staged
    # phases (real ETL is I/O-phased), and writing a temp file without
    # open() is contortion — rejections for it cost ~15% of archetypes.
    # This screen curates our own teacher's code in an ephemeral isolated
    # pod; it is not a security sandbox (see validate_task_code).
    "exec", "eval", "compile", "__import__", "input", "breakpoint",
}
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "http", "multiprocessing", "ctypes", "pickle", "importlib",
}

GENERATION_PROMPT = """\
You generate realistic Python workloads for benchmarking a workflow \
system's resource estimation. Write ONE self-contained Python module that:

1. defines `def run() -> dict:` doing realistic {family_hint} work \
(construct synthetic in-memory data — no files, no network),
2. holds a significant, roughly steady memory footprint for at least \
{duration_s} seconds by repeating its core work in a timed loop \
(`time.monotonic()` deadline pattern),
3. returns a small dict of result stats,
4. imports only from: {allowed},
5. targets roughly {target_mib} MiB of peak memory (choose data sizes \
accordingly) and about {target_cores} CPU core(s).

Also provide a one-line human description of the workload's input profile.

Respond with ONLY this JSON (no code fences):
{{"description": "<one line>", "code": "<the module source, \\n-escaped>"}}"""

FAMILY_HINTS = [
    ("data_engineering", "tabular ETL (joins/groupbys/window ops)"),
    ("data_science", "statistical/tabular model fitting"),
    ("ml_training", "a small neural-net training loop on CPU"),
    ("batch_inference", "vectorized scoring/embedding math"),
    ("etl", "record parsing and aggregation"),
    # Round 13: the workloads people actually deploy on Flyte today —
    # RAG/agent pipelines whose footprint is documents, chunks, embedding
    # matrices, vector indexes and accumulated graph state.
    ("agent_pipeline", "an LLM-agent / RAG pipeline stage with every model "
                       "call STUBBED (the footprint is documents, chunking, "
                       "embedding matrices, a vector index and graph state)"),
]

# GPU families: the generated code must move work to CUDA behind a
# `torch.cuda.is_available()` guard (calibration pods have a T4).
GPU_FAMILY_HINTS = [
    ("gpu_batch_inference", "batched tensor inference on CUDA (torch, fp16)"),
    ("gpu_training", "training-step loop on CUDA (torch: forward/backward/optim)"),
]

# ── scenario grid (round 12): diversity by construction, not temperature ─
SCENARIO_DOMAINS = [
    "ad-click attribution", "genomics variant records", "web server logs",
    "financial tick data", "geospatial trajectories", "recommender events",
    "IoT sensor streams", "e-commerce orders", "call-center transcripts",
    "clinical lab results", "network flow records", "game telemetry",
    "supply-chain shipments", "energy meter readings", "fraud signals",
]
SCENARIO_SHAPES = [
    "wide numeric table (hundreds of float columns)",
    "long narrow event table (few columns, many rows)",
    "string-heavy records (ids, categories, free text)",
    "sparse features (mostly zeros / sparse matrices)",
    "timestamped series needing resample/window ops",
    "nested/denormalized records flattened before use",
]
SCENARIO_PATTERNS = [
    "single steady transformation loop",
    "multi-phase: load/generate a raw form, transform it, then aggregate "
    "(phases may have DIFFERENT memory footprints; the peak phase must "
    "hold for the required duration)",
    "bursty: repeatedly build a large intermediate, reduce it, release it",
    "stage data through a temporary file (tempfile) between phases",
]

# ── the stack axis (round 13) ───────────────────────────────────────────
# Library choice IS a footprint decision: pandas materializes, polars is
# Arrow-columnar, duckdb streams and spills to disk, dask partitions,
# jax has its own allocator, and agent/RAG stacks hold memory in
# documents/embeddings/indexes. The teacher is TOLD which stack to use so
# coverage is deterministic instead of luck.
STACKS_BY_FAMILY: dict[str, list[tuple[str, str]]] = {
    "data_engineering": [
        ("pandas", "pandas DataFrames (materialized joins/groupbys/windows)"),
        ("polars", "polars (Arrow-backed, lazy or eager) — use pl.DataFrame/LazyFrame"),
        ("duckdb", "duckdb in-process SQL over Arrow/pandas relations "
                   "(duckdb.connect(); it streams and may spill to disk)"),
        ("dask", "dask.dataframe partitioned over many pandas partitions "
                 "(scheduler='threads', compute() at the end)"),
        ("pyarrow", "pyarrow Tables/compute kernels directly (no pandas)"),
    ],
    "data_science": [
        ("sklearn", "scikit-learn estimators + preprocessing pipelines"),
        ("xgboost", "xgboost.train on a DMatrix (its own histogram structures)"),
        ("lightgbm", "lightgbm.train on a Dataset (histogram binning)"),
        ("statsmodels", "statsmodels OLS/GLM/time-series fits over numpy arrays"),
        ("scipy", "scipy sparse matrices + linalg/stats routines"),
    ],
    "ml_training": [
        ("torch", "plain torch training loop (nn.Module + optimizer)"),
        ("lightning", "lightning.pytorch LightningModule + Trainer "
                      "(fast_dev_run=False, accelerator='cpu', logger=False)"),
        ("jax", "jax + jax.numpy (jit-compiled update step, its own allocator)"),
        ("transformers", "transformers built OFFLINE from a config: "
                         "AutoConfig.for_model(...) / a small XxxConfig then "
                         "AutoModel.from_config(cfg) — NEVER from_pretrained"),
    ],
    "batch_inference": [
        ("numpy", "numpy vectorized scoring (matmul/einsum over batches)"),
        ("torch", "torch.no_grad() batched forward passes"),
        ("transformers", "a from_config transformers model scoring batches offline"),
        ("faiss", "faiss in-memory index (IndexFlatIP/IVF) built then queried"),
    ],
    "etl": [
        ("stdlib", "pure stdlib: csv/json/collections/re over generated records"),
        ("polars", "polars streaming over generated CSV/parquet in a tempdir"),
        ("duckdb", "duckdb reading generated CSV/parquet files from a tempdir"),
        ("networkx", "networkx graph build + traversal over generated edges"),
    ],
    # Agent/RAG work: memory lives in documents, chunks, embedding
    # matrices, vector indexes and accumulated graph state — all of which
    # are exercisable with the LLM calls stubbed out.
    "agent_pipeline": [
        ("langgraph", "langgraph StateGraph whose nodes mutate an accumulating "
                      "state dict (messages//scratchpad grow each iteration); "
                      "any 'model call' is a local stub function"),
        ("langchain_core", "langchain_core Documents + "
                           "langchain_text_splitters RecursiveCharacterTextSplitter "
                           "chunking a generated corpus, with a stub embedder"),
        ("llama_index", "llama_index.core Document/TextNode objects and a "
                        "node parser over a generated corpus (no service context, "
                        "no network)"),
        ("faiss", "a RAG retrieval layer: numpy embedding matrix + faiss index "
                  "built over generated chunks, then batched similarity queries"),
    ],
    "gpu_batch_inference": [
        ("torch", "torch fp16 tensors/modules on CUDA"),
        ("transformers", "a from_config transformers model moved to CUDA (fp16)"),
    ],
    "gpu_training": [
        ("torch", "torch training step on CUDA (forward/backward/optimizer)"),
        ("lightning", "lightning.pytorch Trainer with accelerator='gpu', devices=1"),
    ],
}

# ── per-stack API contracts ─────────────────────────────────────────────
# Round 13 admitted a dozen new libraries with a one-line hint each, and
# the dominant failure of the 1M release (run upnz22h5) was `curated out:
# execution failed` — 462 pods running code that IMPORTED fine and then
# raised: renamed methods, removed kwargs, wrong dtypes, constructors that
# want the network. A one-line hint names a library; it does not pin the
# handful of call signatures that actually decide whether the module runs.
#
# These notes are deliberately narrow: only the pitfalls observed to crash
# oracle pods, stated as the calls to USE. Keep them short — they compete
# with the rest of the prompt for the teacher's attention.
STACK_PITFALLS: dict[str, str] = {
    "polars": (
        "polars>=1.0 API: pl.DataFrame({...}); group_by (NOT groupby); "
        "with_columns (NOT with_column); map_elements (NOT apply); "
        "LazyFrame work must end in .collect()."
    ),
    "duckdb": (
        "duckdb>=1.1 API: con = duckdb.connect(); con.execute(sql).fetchdf() "
        "or con.sql(sql).df(). A pandas/Arrow object in a local variable is "
        "queryable BY ITS VARIABLE NAME in the SQL string."
    ),
    "dask": (
        "dask API: import dask.dataframe as dd; dd.from_pandas(df, "
        "npartitions=N); finish with .compute(scheduler='threads'). Every "
        "dask result is lazy until compute() — an uncomputed graph measures "
        "no memory."
    ),
    "ibis": (
        "ibis>=9 API: t = ibis.memtable(df); build the expression, then "
        "expr.execute() (the duckdb backend is the default in-process one)."
    ),
    "xgboost": (
        "xgboost>=2.1 API: d = xgb.DMatrix(X, label=y); xgb.train(params, d, "
        "num_boost_round=N) with params={'objective': ..., 'tree_method': "
        "'hist'}. early_stopping_rounds REQUIRES an evals list."
    ),
    "lightgbm": (
        "lightgbm>=4.5 API: ds = lgb.Dataset(X, label=y); lgb.train(params, "
        "ds, num_boost_round=N) with params={'objective': ..., 'verbose': -1}. "
        "verbose_eval was REMOVED in 4.x — passing it raises."
    ),
    "statsmodels": (
        "statsmodels API: sm.OLS(y, sm.add_constant(X)).fit() over float64 "
        "arrays; ARIMA lives at statsmodels.tsa.arima.model.ARIMA."
    ),
    "networkx": (
        "networkx>=3 API: nx.Graph()/nx.DiGraph() + add_edges_from(...); "
        "from_numpy_array (from_numpy_matrix was REMOVED in 3.x)."
    ),
    "jax": (
        "jax API: import jax.numpy as jnp; randomness needs an explicit key "
        "(key = jax.random.PRNGKey(0); jax.random.normal(key, shape)); jax "
        "arrays are IMMUTABLE — update via x.at[idx].set(v); call "
        "jax.block_until_ready(out) before measuring, since jax is async."
    ),
    "lightning": (
        "lightning>=2.4 API: import lightning as L; subclass L.LightningModule "
        "with training_step(self, batch, batch_idx) + configure_optimizers; "
        "L.Trainer(max_epochs=1, accelerator='cpu', logger=False, "
        "enable_checkpointing=False, enable_progress_bar=False, "
        "default_root_dir=<a tempdir>) — the defaults WRITE TO DISK in the "
        "working directory and fail. Feed it a torch DataLoader."
    ),
    "transformers": (
        "transformers OFFLINE only: cfg = AutoConfig.for_model('bert', "
        "vocab_size=V, hidden_size=H, num_hidden_layers=L, "
        "num_attention_heads=A, intermediate_size=4*H); model = "
        "AutoModel.from_config(cfg). H MUST be divisible by A. Feed "
        "torch.randint(0, V, (batch, seq)) as input_ids — there is no "
        "tokenizer and from_pretrained will fail with no network."
    ),
    "faiss": (
        "faiss API: index = faiss.IndexFlatIP(d); index.add(x) where x is a "
        "C-CONTIGUOUS float32 numpy array (np.ascontiguousarray(x, "
        "dtype='float32')) — any other dtype raises. An IVF index must be "
        "train()ed before add(), with nlist well under the vector count."
    ),
    "langchain_core": (
        "langchain_core only (the `langchain` package is NOT installed): "
        "from langchain_core.documents import Document → Document("
        "page_content=..., metadata={}); from langchain_text_splitters import "
        "RecursiveCharacterTextSplitter → .split_documents(docs). Embedders "
        "and chat models must be local stub callables."
    ),
    "langgraph": (
        "langgraph>=0.2 API: from langgraph.graph import StateGraph, START, "
        "END; state is a TypedDict; g = StateGraph(State); g.add_node(name, "
        "fn); g.add_edge(START, name); g.add_edge(name, END); "
        "app = g.compile(); app.invoke(state). Any loop MUST terminate — pass "
        "config={'recursion_limit': N} and bound the iteration count in the "
        "state, or the graph raises instead of finishing."
    ),
    "llama_index": (
        "llama_index.core only: from llama_index.core.schema import Document, "
        "TextNode; from llama_index.core.node_parser import SentenceSplitter → "
        ".get_nodes_from_documents(docs). Do NOT build a VectorStoreIndex or "
        "anything taking an embed_model — those reach for the network."
    ),
}

# Never let the teacher reach for the network: these pods have none.
OFFLINE_RULE = (
    "The runner has NO NETWORK and NO pre-downloaded model weights: never "
    "call from_pretrained, hf_hub_download, tiktoken.get_encoding, or any "
    "API/client. Build models from config objects and synthesize all data "
    "in-process."
)


def pick_stack(family: str, rng) -> tuple[str, str]:
    """(stack name, usage guidance) for a family — the axis that makes the
    corpus span libraries instead of re-teaching pandas."""
    options = STACKS_BY_FAMILY.get(family) or [("numpy", "numpy arrays")]
    return rng.choice(options)


def build_scenario(rng, family: str | None = None) -> str:
    """One sampled scenario clause for the archetype prompt."""
    parts = [
        f"Domain: {rng.choice(SCENARIO_DOMAINS)}.",
        f"Data shape: {rng.choice(SCENARIO_SHAPES)}.",
        f"Structure: {rng.choice(SCENARIO_PATTERNS)}.",
    ]
    if family:
        name, guidance = pick_stack(family, rng)
        parts.append(f"Stack: use {guidance} as the primary library.")
        pitfall = STACK_PITFALLS.get(name)
        if pitfall:
            # On its own line: this is a contract to satisfy, not scenery.
            parts.append(f"\nAPI contract for that stack — {pitfall}")
    return " ".join(parts)


class RejectedTask(ValueError):
    """Why a teacher sample failed the safety/shape screen."""


def parse_teacher_response(text: str) -> tuple[str, str]:
    """Teacher completion → (description, code). Raises RejectedTask."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RejectedTask("no JSON object in teacher response")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise RejectedTask(f"bad JSON: {e}")
    code, desc = obj.get("code"), obj.get("description")
    if not code or not isinstance(code, str):
        raise RejectedTask("missing code")
    return (str(desc or "teacher-generated workload"), textwrap.dedent(code))


def validate_task_code(code: str) -> None:
    """AST screen: parseable, defines run(), imports only from the
    allowlist, touches no filesystem/network/process surface. Raises
    RejectedTask. This is a curation filter for OUR OWN teacher's output
    running in an isolated pod — not a general sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RejectedTask(f"syntax error: {e}")
    has_run = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "run"
        for n in tree.body
    )
    if not has_run:
        raise RejectedTask("no top-level def run()")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    raise RejectedTask(f"forbidden import {root!r}")
                if root not in ALLOWED_IMPORTS:
                    raise RejectedTask(f"import {root!r} not in allowlist")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise RejectedTask(f"forbidden builtin {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in ("system", "popen", "fork"):
            raise RejectedTask(f"forbidden attribute .{node.attr}")


def curate_measurement(measured: dict, min_mib: float = 96, max_mib: float = 12288,
                       min_s: float = 30, max_s: float = 400) -> str | None:
    """Oracle result → rejection reason, or None if the sample survives.

    Bounds keep the corpus schedulable (max) and the metrics pipeline
    honest (min duration: attempts under ~30s have no pod-metric samples).
    """
    if not measured.get("ok"):
        return f"execution failed: {measured.get('error', '')[:200]}"
    if not (min_mib <= measured["peak_rss_mib"] <= max_mib):
        return f"peak {measured['peak_rss_mib']:.0f}MiB out of bounds"
    if not (min_s <= measured["duration_s"] <= max_s):
        return f"duration {measured['duration_s']:.0f}s out of bounds"
    return None


def synthetic_record(
    task_id: str,
    family: str,
    code: str,
    description: str,
    measured: dict,
    generator: str = "teacher",
) -> dict:
    """Oracle-labeled corpus row (same schema as template records — the
    policy cannot tell them apart, which is the point)."""
    return {
        "task_id": task_id,
        "family": family,
        "source_code": code,  # policy sees the plain module: cold-start UC2
        "harness_code": code,
        "input_profile": description,
        "params_json": json.dumps({"synthetic": True}),
        "generator": generator,  # which teacher model wrote this task
        "prior_json": "",  # teacher tasks are cold-start by construction
        "history_json": "",
        "true_peak_memory_mib": float(measured["peak_rss_mib"]),
        "true_cpu_cores": max(1.0, round(float(measured.get("cpu_avg_cores", 1.0)), 1)),
        "true_gpu_mem_mib": 0.0,  # oracle pods are CPU-only
        "duration_s": int(measured["duration_s"]),
        "split": "train",
    }
