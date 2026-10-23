"""RightSizeCPU workload families: real code whose peak CPU the code decides.

Same shape as RightSizeMem's workloads (``resource_tuner.bench.workloads``:
code + inputs + hidden knobs, rendered as a Flyte task for arms to read and
as a harness that executes), with the knobs that decide *parallelism*:

- explicit pool sizes: ``OPENBLAS_NUM_THREADS``, ``POLARS_MAX_THREADS``,
  ``SET threads``, ``n_jobs``, ``torch.set_num_threads``,
  ``threadpool_limits``, ``pa.set_cpu_count``, ``max_workers``;
- what a pool can actually run in parallel: Python threads on pure-Python
  work hold the GIL (one core however many threads), while ``zlib``,
  ``hashlib`` and numpy release it; processes do not share a GIL;
- work that caps a pool below its size (4 trees on 8 jobs, 3 chunks on 6
  processes), small batches that do not parallelize, serial libraries
  (pandas), and I/O waits that leave cores idle.

The measurement environment pins BLAS/OpenMP to one thread unless the code
says otherwise (``measure._base_env``): an unsized pool is one thread here,
and every prompt says so.

Every workload loops until ``TIME_BUDGET_S`` (60-90 s), so a run spans
several 15-second platform windows and its peak is a sustained window, not
an import-time spike. ``driver`` names what an arm had to understand:

- ``threads``    an explicit pool size, honored by the library
- ``gil``        a thread pool whose parallelism depends on the GIL
- ``processes``  a process pool, possibly capped by the number of chunks
- ``serial``     single-threaded code, whatever the input
- ``io``         waits dominate; peak well under one core
- ``phases``     a serial phase then a parallel one: peak != mean
"""

from __future__ import annotations

import hashlib
from random import Random

from ..bench.workloads import (
    Family,
    Task,
    _floats,
    _ind,
    _log_uniform,
    array,
    col,
    table,
)

#: Pool sizes a task asks for. Weighted toward small pools, the common case.
_POOLS = (1, 2, 2, 3, 4, 4, 6, 8)
_BUDGETS = (60, 75, 90)
#: Inputs vary 4x across a task's runs (RightSizeMem: 16x). Peak CPU barely
#: moves with size; prep time and pod memory do.
_SIZES = (0.5, 2.0)


def _budget(rng: Random) -> str:
    return f'TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "{rng.choice(_BUDGETS)}"))'


def _loop(lines: list[str]) -> list[str]:
    """`lines` repeated until the time budget is spent."""
    return ["t0 = time.monotonic()", "passes = 0", "while time.monotonic() - t0 < TIME_BUDGET_S:",
            *[f"    {line}" for line in lines], "    passes += 1"]


# ── 1. numpy BLAS ───────────────────────────────────────────────────────


def _numpy_blas(rng: Random) -> dict:
    k = rng.choice(_POOLS + (None,))  # None: the code never sizes BLAS
    n = int(_log_uniform(rng, 2e4, 1.5e5))
    d = rng.choice([64, 128, 256, 512])
    dtype = rng.choice(["float32", "float64"])
    op = rng.choice(["gram", "project", "svd_sample", "solve"])
    preamble = ""
    if k is not None:
        preamble = f'import os\nos.environ["OPENBLAS_NUM_THREADS"] = "{k}"\nos.environ["OMP_NUM_THREADS"] = "{k}"'
    work = {
        "gram": ["G = X.T @ X"],
        "project": ["Y = X @ W", "Y = np.tanh(Y, out=Y)"],
        "svd_sample": ["idx = rng.integers(0, len(X), 4096)", "_ = np.linalg.svd(X[idx], full_matrices=False)"],
        "solve": ["G = X[:8192].T @ X[:8192] + np.eye(X.shape[1], dtype=X.dtype)", "_ = np.linalg.solve(G, X[:64].T)"],
    }[op]
    body = [_budget(rng), "X = np.load(features_path)", "rng = np.random.default_rng(0)",
            "W = rng.standard_normal((X.shape[1], X.shape[1])).astype(X.dtype)"] + _loop(work) + [
            'return {"passes": passes}']
    return {"func_name": "embed_features", "imports": "import os\nimport time\nimport numpy as np",
            "body": "\n".join(body), "inputs": [array("features", [n, d], dtype)],
            "knobs": {"threads": k, "op": op, "d": d}, "preamble": preamble}


# ── 2. polars group-by ──────────────────────────────────────────────────


def _polars_groupby(rng: Random) -> dict:
    k = rng.choice(_POOLS)
    rows = int(_log_uniform(rng, 1e6, 6e6))
    keys = rng.choice([100, 10_000, 1_000_000])
    n_float = rng.choice([4, 8, 16])
    op = rng.choice(["agg", "window", "sort_agg"])
    cols = [col("key", "i64", distinct=keys), col("region", "cat", distinct=20)] + _floats("x", n_float)
    q = {
        "agg": 'lf.group_by("key").agg(pl.col("x0").mean(), pl.col("x1").std(), pl.len())',
        "window": 'lf.with_columns((pl.col("x0") - pl.col("x0").mean().over("key")).alias("z")).select(pl.col("z").sum())',
        "sort_agg": 'lf.sort("x0").group_by("region").agg(pl.col("x1").quantile(0.9))',
    }[op]
    body = [_budget(rng), "lf = pl.scan_parquet(events_path)"] + _loop([f"out = ({q}).collect()"]) + [
        'return {"passes": passes, "rows": out.height}']
    return {"func_name": "summarize_events", "imports": "import os\nimport time\nimport polars as pl",
            "body": "\n".join(body), "inputs": [table("events", "parquet", rows, cols)],
            "knobs": {"threads": k, "op": op, "keys": keys},
            "preamble": f'import os\nos.environ["POLARS_MAX_THREADS"] = "{k}"'}


# ── 3. duckdb (held out) ────────────────────────────────────────────────


def _duckdb_sql(rng: Random) -> dict:
    k = rng.choice(_POOLS)
    rows = int(_log_uniform(rng, 1e6, 6e6))
    kind = rng.choice(["aggregate", "window", "join_self"])
    cols = [col("account_id", "i64", distinct=rng.choice([1000, 100_000])), col("kind", "cat", distinct=12)] + _floats("amt", 6)
    sql = {
        "aggregate": "SELECT account_id, sum(amt0), avg(amt1), count(*) FROM read_parquet('{p}') GROUP BY account_id",
        "window": "SELECT account_id, amt0 - avg(amt0) OVER (PARTITION BY account_id) AS d FROM read_parquet('{p}') ORDER BY d DESC LIMIT 100",
        "join_self": "SELECT a.kind, count(*) FROM read_parquet('{p}') a JOIN (SELECT DISTINCT account_id, kind FROM read_parquet('{p}')) b USING (account_id) GROUP BY a.kind",
    }[kind]
    body = [_budget(rng), "con = duckdb.connect()", f'con.execute("SET threads = {k}")',
            f'sql = "{sql}".format(p=ledger_path)'] + _loop(["out = con.execute(sql).fetchall()"]) + [
            'return {"passes": passes}']
    return {"func_name": "ledger_report", "imports": "import os\nimport time\nimport duckdb",
            "body": "\n".join(body), "inputs": [table("ledger", "parquet", rows, cols)],
            "knobs": {"threads": k, "kind": kind}}


# ── 4. scikit-learn ensembles ───────────────────────────────────────────


def _sklearn_ensemble(rng: Random) -> dict:
    k = rng.choice(_POOLS)
    rows = int(_log_uniform(rng, 2e4, 2e5))
    n_feat = rng.choice([10, 20, 40])
    model = rng.choice(["forest", "forest", "extra", "hgb"])
    trees = rng.choice([4, 16, 64, 200])
    cols = _floats("f", n_feat) + [col("label", "i32", distinct=2)]
    if model == "hgb":
        imports = "import os\nimport time\nimport pandas as pd\nfrom sklearn.ensemble import HistGradientBoostingClassifier\nfrom threadpoolctl import threadpool_limits"
        fit = [f"with threadpool_limits(limits={k}):",
               "    HistGradientBoostingClassifier(max_iter=100).fit(X, y)"]
    else:
        cls = "RandomForestClassifier" if model == "forest" else "ExtraTreesClassifier"
        imports = f"import os\nimport time\nimport pandas as pd\nfrom sklearn.ensemble import {cls}"
        fit = [f"{cls}(n_estimators={trees}, max_depth=12, n_jobs={k}).fit(X, y)"]
    body = [_budget(rng), "df = pd.read_parquet(train_path)", 'X, y = df.drop(columns="label").to_numpy(), df["label"].to_numpy()'] + _loop(fit) + [
        'return {"fits": passes}']
    return {"func_name": "train_classifier", "imports": imports, "body": "\n".join(body),
            "inputs": [table("train", "parquet", rows, cols)],
            "knobs": {"threads": k, "model": model, "trees": trees if model != "hgb" else None}}


# ── 5. pandas (serial) ──────────────────────────────────────────────────


def _pandas_serial(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 5e5, 3e6))
    op = rng.choice(["groupby", "merge", "apply_rows", "string_ops"])
    cols = [col("customer_id", "i64", distinct=rng.choice([1000, 50_000])), col("city", "cat", distinct=200),
            col("note", "str", len=16)] + _floats("v", 6)
    work = {
        "groupby": ['out = df.groupby("customer_id").agg(v0=("v0", "mean"), v1=("v1", "max"))'],
        "merge": ['out = df.merge(df.groupby("city", observed=True)["v0"].mean().rename("city_mean"), on="city")'],
        "apply_rows": ['out = df.head(20_000).apply(lambda r: r["v0"] * 2 + r["v1"], axis=1)'],
        "string_ops": ['out = df["note"].str.upper().str.slice(0, 4).value_counts()'],
    }[op]
    body = [_budget(rng), "df = pd.read_parquet(orders_path)"] + _loop(work) + ['return {"passes": passes}']
    return {"func_name": "customer_rollup", "imports": "import os\nimport time\nimport pandas as pd",
            "body": "\n".join(body), "inputs": [table("orders", "parquet", rows, cols)],
            "knobs": {"op": op}}


# ── 6. thread pools and the GIL ─────────────────────────────────────────


def _python_threadpool(rng: Random) -> dict:
    k = rng.choice(_POOLS[1:])
    chunks = int(_log_uniform(rng, 200, 2000))
    work = rng.choice(["pure_python", "pure_python", "zlib", "hashlib", "numpy_sort"])
    fn = {
        "pure_python": ["def work(chunk):", "    total = 0", "    for b in chunk[:20_000].tolist():", "        total += (b * b) % 7",
                        "    return total"],
        "zlib": ["def work(chunk):", "    return len(zlib.compress(chunk.tobytes(), 6))"],
        "hashlib": ["def work(chunk):", "    h = hashlib.sha256()", "    for _ in range(8):", "        h.update(chunk.tobytes())",
                    "    return h.hexdigest()"],
        "numpy_sort": ["def work(chunk):", "    return int(np.sort(chunk.astype(np.int32))[len(chunk) // 2])"],
    }[work]
    body = [_budget(rng), "blob = np.load(blob_path)", *fn,
            f"with ThreadPoolExecutor(max_workers={k}) as pool:"] + [
            f"    {line}" for line in _loop(["results = list(pool.map(work, blob))"])] + ['return {"passes": passes}']
    imports = "import hashlib\nimport os\nimport time\nimport zlib\nfrom concurrent.futures import ThreadPoolExecutor\nimport numpy as np"
    return {"func_name": "digest_blobs", "imports": imports, "body": "\n".join(body),
            "inputs": [array("blob", [chunks, 65_536], "uint8")], "knobs": {"threads": k, "work": work}}


# ── 7. process pools ────────────────────────────────────────────────────


def _process_pool(rng: Random) -> dict:
    k = rng.choice(_POOLS[1:])
    n_chunks = rng.choice([2, 3, 4, 8, 16, 64])
    rows = int(_log_uniform(rng, 2e5, 2e6))
    cols = [col("id", "i64"), col("text", "text", words=20)]
    body = [_budget(rng), "with open(docs_path) as f:", "    lines = f.read().splitlines()[1:]",
            f"parts = [lines[i::{n_chunks}] for i in range({n_chunks})]",
            f"with ProcessPoolExecutor(max_workers={k}) as pool:"] + [
            f"    {line}" for line in _loop(["counts = list(pool.map(count_words, parts))"])] + [
            'return {"passes": passes, "words": sum(counts)}']
    helper = ("def count_words(part):\n    n = 0\n    for line in part:\n        for w in line.split():\n"
              "            n += len(w) % 3\n    return n")
    return {"func_name": "count_tokens", "imports": "import os\nimport time\nfrom concurrent.futures import ProcessPoolExecutor",
            "body": "\n".join(body), "inputs": [table("docs", "csv", rows, cols)],
            "knobs": {"processes": k, "chunks": n_chunks}, "helpers": helper}


# ── 8. torch on CPU ─────────────────────────────────────────────────────


def _torch_cpu(rng: Random) -> dict:
    k = rng.choice(_POOLS)
    rows = int(_log_uniform(rng, 2e4, 2e5))
    n_feat = rng.choice([16, 64, 128])
    width = rng.choice([64, 256, 1024])
    batch = rng.choice([32, 256, 4096])
    cols = _floats("f", n_feat) + [col("y", "f32")]
    body = [_budget(rng), f"torch.set_num_threads({k})", "df = pd.read_parquet(train_path)",
            'X = torch.tensor(df.drop(columns="y").to_numpy(), dtype=torch.float32)',
            'y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)[:, None]',
            f"model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], {width}), torch.nn.ReLU(), "
            f"torch.nn.Linear({width}, {width}), torch.nn.ReLU(), torch.nn.Linear({width}, 1))",
            "opt = torch.optim.Adam(model.parameters(), lr=1e-3)"] + _loop([
            f"for i in range(0, len(X), {batch}):",
            f"    loss = torch.nn.functional.mse_loss(model(X[i : i + {batch}]), y[i : i + {batch}])",
            "    opt.zero_grad()", "    loss.backward()", "    opt.step()",
            "    if time.monotonic() - t0 > TIME_BUDGET_S:", "        break"]) + ['return {"epochs": passes}']
    return {"func_name": "fit_ranker", "imports": "import os\nimport time\nimport pandas as pd\nimport torch",
            "body": "\n".join(body), "inputs": [table("train", "parquet", rows, cols)],
            "knobs": {"threads": k, "width": width, "batch": batch}}


# ── 9. pyarrow CSV (held out) ───────────────────────────────────────────


def _pyarrow_csv(rng: Random) -> dict:
    k = rng.choice(_POOLS)
    use_threads = rng.random() < 0.75
    rows = int(_log_uniform(rng, 5e5, 3e6))
    cols = [col("sensor", "i64", distinct=5000)] + _floats("r", rng.choice([4, 8, 12]))
    body = [_budget(rng), f"pa.set_cpu_count({k})",
            f"opts = pacsv.ReadOptions(use_threads={use_threads}, block_size=1 << 22)"] + _loop([
            "t = pacsv.read_csv(readings_path, read_options=opts)",
            'm = pc.mean(t["r0"]).as_py()']) + ['return {"passes": passes, "rows": t.num_rows}']
    imports = "import os\nimport time\nimport pyarrow as pa\nimport pyarrow.compute as pc\nimport pyarrow.csv as pacsv"
    return {"func_name": "parse_readings", "imports": imports, "body": "\n".join(body),
            "inputs": [table("readings", "csv", rows, cols)], "knobs": {"threads": k, "use_threads": use_threads}}


# ── 10. I/O-bound fetch-and-parse ───────────────────────────────────────


def _io_bound(rng: Random) -> dict:
    k = rng.choice((1, 2, 4, 8, 16))
    latency_ms = rng.choice([5, 20, 50, 200])
    rows = int(_log_uniform(rng, 5e4, 5e5))
    cols = [col("user", "i64"), col("event", "cat", distinct=30), col("payload", "str", len=64)]
    body = [_budget(rng), "with open(events_path) as f:", "    records = [json.loads(line) for line in f]",
            "batches = [records[i : i + 500] for i in range(0, len(records), 500)]",
            "def fetch_and_score(batch):",
            f"    time.sleep({latency_ms / 1000})  # the enrichment API's round trip",
            "    return sum(len(json.dumps(r)) for r in batch)",
            f"with ThreadPoolExecutor(max_workers={k}) as pool:"] + [
            f"    {line}" for line in _loop(["scores = list(pool.map(fetch_and_score, batches))"])] + [
            'return {"passes": passes}']
    return {"func_name": "enrich_events", "imports": "import json\nimport os\nimport time\nfrom concurrent.futures import ThreadPoolExecutor",
            "body": "\n".join(body), "inputs": [table("events", "jsonl", rows, cols)],
            "knobs": {"threads": k, "latency_ms": latency_ms}}


# ── 11. serial phase, then a parallel one ───────────────────────────────


def _mixed_phases(rng: Random) -> dict:
    k = rng.choice(_POOLS[1:])
    serial_frac = rng.choice([0.2, 0.5, 0.8])
    n = int(_log_uniform(rng, 5e4, 3e5))
    preamble = f'import os\nos.environ["OPENBLAS_NUM_THREADS"] = "{k}"'
    body = [_budget(rng), "X = np.load(matrix_path)",
            "# phase 1: per-row Python cleanup (one core)",
            "t0 = time.monotonic()", "rows_seen = 0",
            f"while time.monotonic() - t0 < {serial_frac} * TIME_BUDGET_S:",
            "    for row in X[:2000].tolist():", "        rows_seen += sum(1 for v in row if v > 0)",
            "# phase 2: blocked Gram matrix (BLAS)", "passes = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:", "    G = X.T @ X", "    passes += 1",
            'return {"rows": rows_seen, "passes": passes}']
    return {"func_name": "clean_and_factor", "imports": "import os\nimport time\nimport numpy as np",
            "body": "\n".join(body), "inputs": [array("matrix", [n, 256], "float64")],
            "knobs": {"threads": k, "serial_frac": serial_frac}, "preamble": preamble}


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family("numpy_blas", "threads", _numpy_blas, ("numpy",), size_range=_SIZES),
        Family("polars_groupby", "threads", _polars_groupby, ("polars",), size_range=_SIZES),
        Family("duckdb_sql", "threads", _duckdb_sql, ("duckdb",), size_range=_SIZES),
        Family("sklearn_ensemble", "threads", _sklearn_ensemble, ("scikit-learn", "pandas", "pyarrow"), size_range=_SIZES),
        Family("pandas_serial", "serial", _pandas_serial, ("pandas", "pyarrow"), size_range=_SIZES),
        Family("python_threadpool", "gil", _python_threadpool, ("numpy",), size_range=_SIZES),
        Family("process_pool", "processes", _process_pool, (), size_range=_SIZES),
        Family("torch_cpu", "threads", _torch_cpu, ("torch", "pandas", "pyarrow"), size_range=_SIZES),
        Family("pyarrow_csv", "threads", _pyarrow_csv, ("pyarrow",), size_range=_SIZES),
        Family("io_bound", "io", _io_bound, (), size_range=_SIZES),
        Family("mixed_phases", "phases", _mixed_phases, ("numpy",), size_range=_SIZES),
    )
}

#: Never trained on: an engine (duckdb) and a reader (pyarrow's CSV) whose
#: pool-size APIs no training family uses.
HELDOUT_FAMILIES: tuple[str, ...] = ("duckdb_sql", "pyarrow_csv")


def make_task(family: str, seed: int, n_runs: int = 2) -> Task:
    fam = FAMILIES[family]
    rng = Random(seed)
    spec = fam.make(rng)
    lo, hi = fam.size_range
    factors = sorted(_log_uniform(rng, lo, hi) for _ in range(n_runs))
    rng.shuffle(factors)
    digest = hashlib.sha1(f"cpu:{family}:{seed}".encode()).hexdigest()[:10]
    body = spec["body"]
    if spec.get("helpers"):
        # module-level helpers (a process pool pickles its function by name)
        spec["imports"] = spec["imports"] + "\n\n\n" + spec["helpers"]
    return Task(task_id=f"{family}-{digest}", family=family, memory_driver=fam.memory_driver,
                func_name=spec["func_name"], imports=spec["imports"], body=body, inputs=spec["inputs"],
                knobs=spec["knobs"], size_factors=factors, packages=fam.packages, preamble=spec.get("preamble", ""))


__all__ = ["FAMILIES", "HELDOUT_FAMILIES", "make_task", "_ind"]
