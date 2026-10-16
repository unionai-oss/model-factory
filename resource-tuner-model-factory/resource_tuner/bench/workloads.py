"""Workload families: real code whose memory is driven by inputs AND by code.

The previous corpus rendered code with its sizes baked in as literals and
labelled it with analytic formulas. This one is built around the question
the product actually faces: *a task is submitted with some inputs -- what
should it ask for?* So every workload is three things:

- **code** -- a real task body (pandas, polars, duckdb, numpy, sklearn,
  torch, plain Python). The text an arm reads is the text that executes;
  only the Flyte wrapper differs (``render_task`` vs ``render_harness``).
- **inputs** -- files the task reads, described the way a scheduler can
  see them at submit time: format, bytes on disk, and whatever the format's
  own metadata exposes (a parquet footer has rows and a schema; a CSV has
  only its size and header line; an ``.npy`` header has shape and dtype).
- **knobs** -- the code-level choices that decide how input bytes turn into
  resident memory: column projection, ``dtype_backend``, float32
  downcasts, streaming vs eager engines, ``chunksize``, ``mmap_mode``,
  ``.toarray()``, dedup-before-join, Python objects vs columnar. Knobs are
  hidden from every arm; they are recorded only so results can be sliced
  by *what an arm had to understand* to get the answer right.

Each family's ``memory_driver`` names that boundary:

- ``size``   input bytes predict the peak; the code is a constant multiplier
- ``code``   the peak is (nearly) independent of input size -- chunked,
             streaming, mmap'd, or sized by constants in the code
- ``both``   the code decides the multiplier *and* the multiplier is large

A *task* is one code instance; a *run* is that task on one set of input
sizes. Tasks are run at several sizes, which is what gives an arm a run
history to read.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from random import Random
from typing import Callable

# ── input specs ─────────────────────────────────────────────────────────
# Plain dicts (they ride in parquet rows as JSON):
#   table: {"name", "fmt": parquet|csv|jsonl, "rows", "scales": bool,
#           "columns": [{"name", "kind": f64|f32|i64|i32|cat|str|text,
#                        "distinct"?, "len"?, "words"?}]}
#   array: {"name", "fmt": "npy", "shape": [n, ...], "dtype", "scales": bool}


def col(name: str, kind: str, **extra) -> dict:
    return {"name": name, "kind": kind, **extra}


def table(name: str, fmt: str, rows: int, columns: list[dict], scales: bool = True) -> dict:
    return {"name": name, "fmt": fmt, "rows": int(rows), "columns": columns, "scales": scales}


def array(name: str, shape: list[int], dtype: str, scales: bool = True) -> dict:
    return {"name": name, "fmt": "npy", "shape": [int(s) for s in shape], "dtype": dtype, "scales": scales}


def scale_input(spec: dict, factor: float) -> dict:
    """The same input at a different size (only the leading dimension)."""
    if not spec.get("scales", True):
        return dict(spec)
    out = json.loads(json.dumps(spec))
    if spec["fmt"] == "npy":
        out["shape"][0] = max(16, int(spec["shape"][0] * factor))
    else:
        out["rows"] = max(64, int(spec["rows"] * factor))
    return out


# ── tasks ───────────────────────────────────────────────────────────────


@dataclass
class Task:
    """One code instance. ``inputs`` are at the family's base size; runs
    rescale them by ``size_factors``."""

    task_id: str
    family: str
    memory_driver: str
    func_name: str
    imports: str
    body: str
    inputs: list[dict]
    knobs: dict
    size_factors: list[float]
    packages: tuple[str, ...] = ()
    preamble: str = ""  # module-level lines before imports (thread caps)

    def run_inputs(self, k: int) -> list[dict]:
        return [scale_input(s, self.size_factors[k]) for s in self.inputs]


@dataclass(frozen=True)
class Family:
    name: str
    memory_driver: str
    make: Callable[[Random], dict]
    packages: tuple[str, ...]
    # Size factors relative to the family's base rows; runs draw from this
    # log-range so one task spans small and large inputs.
    size_range: tuple[float, float] = (0.25, 4.0)
    notes: str = ""


def _ind(code: str, n: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(code).strip("\n"), " " * n)


def _floats(prefix: str, n: int, kind: str = "f64") -> list[dict]:
    return [col(f"{prefix}{i}", kind) for i in range(n)]


def _strs(prefix: str, n: int, length: int) -> list[dict]:
    return [col(f"{prefix}{i}", "str", len=length) for i in range(n)]


def _log_uniform(rng: Random, lo: float, hi: float) -> float:
    import math

    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


# ── 1. pandas groupby / feature pipeline ────────────────────────────────


def _pandas_groupby(rng: Random) -> dict:
    n_float = rng.choice([4, 8, 16, 24, 40])
    n_str = rng.choice([0, 0, 1, 2, 4])
    str_len = rng.choice([8, 16, 32])
    rows = int(_log_uniform(rng, 4e5, 4e6))
    customers = rng.choice([1_000, 50_000, 500_000])
    project = rng.random() < 0.45
    backend = rng.choice(["numpy", "numpy", "pyarrow"])
    downcast = rng.random() < 0.35
    op = rng.choice(["agg", "transform", "rank", "describe"])
    sort = rng.random() < 0.4
    write = rng.random() < 0.5
    cols = [col("customer_id", "i64", distinct=customers), col("region", "cat", distinct=20)]
    cols += _floats("f", n_float) + _strs("s", n_str, str_len)
    keep = ["customer_id", "region", "f0", "f1", "f2"] if project else None
    read_args = f", columns={keep!r}" if keep else ""
    if backend == "pyarrow":
        read_args += ', dtype_backend="pyarrow"'
    lines = [f"df = pd.read_parquet(events_path{read_args})"]
    if downcast:
        lines.append('floats = df.select_dtypes("number").columns.drop("customer_id")')
        lines.append('df[floats] = df[floats].astype("float32")')
    if op == "agg":
        lines.append('out = df.groupby("customer_id").agg(f0_mean=("f0", "mean"), f1_max=("f1", "max"), n=("f2", "size"))')
    elif op == "transform":
        lines.append('g = df.groupby("customer_id")["f0"]')
        lines.append('df["f0_z"] = (df["f0"] - g.transform("mean")) / (g.transform("std") + 1e-9)')
        lines.append('df["f1_share"] = df["f1"] / df.groupby("region")["f1"].transform("sum")')
        lines.append("out = df")
    elif op == "rank":
        lines.append('df["f1_rank"] = df.groupby("region")["f1"].rank(pct=True)')
        lines.append('out = df[df["f1_rank"] > 0.5]')
    else:
        lines.append('out = df.describe(include="all") if len(df.columns) < 12 else df.describe()')
    if sort:
        lines.append('out = out.sort_values(out.columns[0])')
    if write:
        lines.append('out.to_parquet("features.parquet")')
    lines.append('return {"rows": int(len(out)), "cols": int(out.shape[1])}')
    return {
        "func_name": "build_customer_features",
        "imports": "import pandas as pd",
        "body": "\n".join(lines),
        "inputs": [table("events", "parquet", rows, cols)],
        "knobs": {"project": project, "backend": backend, "downcast": downcast, "op": op,
                  "sort": sort, "write": write, "n_float": n_float, "n_str": n_str},
    }


# ── 2. pandas join (fact x dimension) ───────────────────────────────────


def _pandas_join(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 3e5, 3e6))
    skus = rng.choice([10_000, 100_000, 400_000])
    dup = rng.choice([1, 1, 2, 4, 8])  # rows per sku in the dimension table
    n_dim_cols = rng.choice([2, 6, 12])
    dedup = dup > 1 and rng.random() < 0.5
    how = rng.choice(["inner", "left"])
    project_after = rng.random() < 0.4
    fact_cols = [col("order_id", "i64", distinct=rows), col("sku", "i64", distinct=skus),
                 col("qty", "i32"), col("price", "f64"), col("ts", "i64")]
    dim_cols = [col("sku", "i64", distinct=skus)] + _floats("attr", n_dim_cols) + [col("brand", "cat", distinct=500)]
    lines = [
        "orders = pd.read_parquet(orders_path)",
        "catalog = pd.read_parquet(catalog_path)",
    ]
    if dedup:
        lines.append('catalog = catalog.drop_duplicates("sku", keep="last")')
    lines.append(f'joined = orders.merge(catalog, on="sku", how="{how}")')
    if project_after:
        lines.append('joined = joined[["order_id", "sku", "qty", "price", "brand"]]')
    lines.append('joined["revenue"] = joined["qty"] * joined["price"]')
    lines.append('by_brand = joined.groupby("brand", observed=True)["revenue"].sum()')
    lines.append('return {"rows": int(len(joined)), "brands": int(len(by_brand))}')
    return {
        "func_name": "join_orders_catalog",
        "imports": "import pandas as pd",
        "body": "\n".join(lines),
        "inputs": [
            table("orders", "parquet", rows, fact_cols),
            table("catalog", "parquet", skus * dup, dim_cols, scales=False),
        ],
        "knobs": {"dup_factor": dup, "dedup": dedup, "how": how, "project_after": project_after},
    }


# ── 3. polars: eager vs lazy vs streaming ───────────────────────────────


def _polars_pipeline(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 1e6, 1.2e7))
    n_float = rng.choice([4, 8, 16, 32])
    users = rng.choice([1_000, 100_000, 1_000_000])
    mode = rng.choice(["eager", "lazy", "streaming", "streaming"])
    threads = rng.choice([1, 2, 3])
    cols = [col("user_id", "i64", distinct=users), col("event", "cat", distinct=12)] + _floats("v", n_float)
    agg = '.group_by("user_id").agg(pl.col("v0").mean().alias("v0_mean"), pl.col("v1").max(), pl.len().alias("n"))'
    filt = '.filter(pl.col("v2") > 0)'
    if mode == "eager":
        body = [f"df = pl.read_parquet(events_path)", f"out = df{filt}{agg}"]
    elif mode == "lazy":
        body = [f"out = pl.scan_parquet(events_path){filt}{agg}.collect()"]
    else:
        body = [f'out = pl.scan_parquet(events_path){filt}{agg}.collect(engine="streaming")']
    body.append('return {"groups": out.height}')
    return {
        "func_name": "user_rollup",
        "preamble": f'import os\nos.environ["POLARS_MAX_THREADS"] = "{threads}"',
        "imports": "import polars as pl",
        "body": "\n".join(body),
        "inputs": [table("events", "parquet", rows, cols)],
        "knobs": {"mode": mode, "threads": threads, "n_float": n_float, "users": users},
    }


# ── 4. duckdb SQL ───────────────────────────────────────────────────────


def _duckdb_sql(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 1e6, 1.5e7))
    n_float = rng.choice([4, 8, 16, 32])
    keys = rng.choice([1_000, 200_000, 2_000_000])
    limit = rng.choice([None, None, "512MB", "1GB", "2GB"])
    threads = rng.choice([1, 2, 3])
    query = rng.choice(["aggregate", "filter_fetch", "window", "aggregate"])
    selectivity = rng.choice([0.01, 0.1, 0.5])
    cols = [col("account_id", "i64", distinct=keys), col("kind", "cat", distinct=8)] + _floats("amt", n_float)
    setup = [f"con = duckdb.connect()", f'con.execute("SET threads = {threads}")']
    if limit:
        setup.append(f"con.execute(\"SET memory_limit = '{limit}'\")")
    src = "read_parquet('{events_path}')"
    if query == "aggregate":
        sql = f"SELECT account_id, sum(amt0) AS total, count(*) AS n FROM {src} GROUP BY account_id"
    elif query == "filter_fetch":
        sql = f"SELECT * FROM {src} WHERE amt1 > {round(_norm_ppf(1 - selectivity), 4)}"
    else:
        sql = (f"SELECT account_id, amt0, avg(amt0) OVER (PARTITION BY account_id) AS acct_avg "
               f"FROM {src} WHERE amt2 > {round(_norm_ppf(1 - selectivity), 4)}")
    body = setup + [f'sql = f"{sql}"', "out = con.execute(sql).df()",
                    'return {"rows": int(len(out))}']
    return {
        "func_name": "ledger_query",
        "imports": "import duckdb",
        "body": "\n".join(body),
        "inputs": [table("events", "parquet", rows, cols)],
        "knobs": {"memory_limit": limit, "threads": threads, "query": query, "selectivity": selectivity},
    }


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


# ── 5. CSV: whole file vs chunked ───────────────────────────────────────


def _csv_chunked(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 3e5, 4e6))
    n_float = rng.choice([4, 8, 16, 32])
    n_str = rng.choice([0, 1, 3])
    chunked = rng.random() < 0.5
    chunk = rng.choice([50_000, 200_000, 1_000_000])
    usecols = rng.random() < 0.4
    f32 = rng.random() < 0.3
    cols = [col("device", "i64", distinct=10_000)] + _floats("x", n_float) + _strs("tag", n_str, 12)
    args = ""
    if usecols:
        args += ', usecols=["device", "x0", "x1"]'
    if f32:
        args += ', dtype={"x0": "float32", "x1": "float32"}'
    if chunked:
        body = [
            "totals = None",
            f"for chunk in pd.read_csv(readings_path, chunksize={chunk:_}{args}):",
            '    part = chunk.groupby("device")[["x0", "x1"]].sum()',
            "    totals = part if totals is None else totals.add(part, fill_value=0)",
            'return {"devices": int(len(totals))}',
        ]
    else:
        body = [
            f"df = pd.read_csv(readings_path{args})",
            'totals = df.groupby("device")[["x0", "x1"]].sum()',
            'return {"devices": int(len(totals))}',
        ]
    return {
        "func_name": "device_totals",
        "imports": "import pandas as pd",
        "body": "\n".join(body),
        "inputs": [table("readings", "csv", rows, cols)],
        "knobs": {"chunked": chunked, "chunk": chunk if chunked else None, "usecols": usecols,
                  "f32": f32, "n_float": n_float, "n_str": n_str},
    }


# ── 6. JSON lines: python objects vs streaming vs pandas ────────────────


def _jsonl_records(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 2e5, 3e6))
    n_float = rng.choice([2, 6, 12])
    n_str = rng.choice([1, 2, 3])
    mode = rng.choice(["list", "stream", "pandas", "pandas_chunked"])
    cols = [col("user", "i64", distinct=50_000)] + _floats("score", n_float) + _strs("field", n_str, 16)
    if mode == "list":
        body = [
            "with open(logs_path) as f:",
            "    records = [json.loads(line) for line in f]",
            'by_user = {}',
            "for r in records:",
            '    by_user[r["user"]] = by_user.get(r["user"], 0.0) + r["score0"]',
            'return {"users": len(by_user), "records": len(records)}',
        ]
    elif mode == "stream":
        body = [
            "by_user = {}",
            "with open(logs_path) as f:",
            "    for line in f:",
            "        r = json.loads(line)",
            '        by_user[r["user"]] = by_user.get(r["user"], 0.0) + r["score0"]',
            'return {"users": len(by_user)}',
        ]
    elif mode == "pandas":
        body = [
            "df = pd.read_json(logs_path, lines=True)",
            'by_user = df.groupby("user")["score0"].sum()',
            'return {"users": int(len(by_user))}',
        ]
    else:
        body = [
            "parts = []",
            "for chunk in pd.read_json(logs_path, lines=True, chunksize=100_000):",
            '    parts.append(chunk.groupby("user")["score0"].sum())',
            'by_user = pd.concat(parts).groupby(level=0).sum()',
            'return {"users": int(len(by_user))}',
        ]
    return {
        "func_name": "score_users",
        "imports": "import json\nimport pandas as pd",
        "body": "\n".join(body),
        "inputs": [table("logs", "jsonl", rows, cols)],
        "knobs": {"mode": mode, "n_float": n_float, "n_str": n_str},
    }


# ── 7. numpy linear algebra ─────────────────────────────────────────────


def _numpy_linalg(rng: Random) -> dict:
    d = rng.choice([16, 64, 128, 256, 512])
    dtype = rng.choice(["float64", "float64", "float32"])
    n = int(_log_uniform(rng, 2e5, 4e6) * 64 / d)
    op = rng.choice(["gram", "svd", "standardize", "pairwise_sample", "chunked_moments"])
    mmap = op == "chunked_moments" or rng.random() < 0.2
    sample = rng.choice([2_000, 8_000, 20_000])
    chunk = rng.choice([10_000, 100_000])
    load = 'np.load(matrix_path, mmap_mode="r")' if mmap else "np.load(matrix_path)"
    lines = [f"X = {load}"]
    if op == "gram":
        lines += ["G = X.T @ X", 'return {"trace": float(np.trace(G))}']
    elif op == "svd":
        lines += ["U, s, Vt = np.linalg.svd(X, full_matrices=False)", 'return {"top": float(s[0])}']
    elif op == "standardize":
        lines += ["Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)", "C = np.cov(Z, rowvar=False)",
                  'return {"c00": float(C[0, 0])}']
    elif op == "pairwise_sample":
        lines += [f"S = np.asarray(X[:{sample:_}])", "sq = (S * S).sum(axis=1)",
                  "D = sq[:, None] + sq[None, :] - 2 * S @ S.T", 'return {"mean_d": float(D.mean())}']
    else:
        lines += ["total = np.zeros(X.shape[1])", "sq = np.zeros(X.shape[1])",
                  f"for i in range(0, X.shape[0], {chunk:_}):",
                  f"    block = np.asarray(X[i:i + {chunk:_}], dtype=np.float64)",
                  "    total += block.sum(axis=0)", "    sq += (block * block).sum(axis=0)",
                  'return {"var0": float(sq[0] / X.shape[0] - (total[0] / X.shape[0]) ** 2)}']
    return {
        "func_name": "matrix_stats",
        "preamble": 'import os\nos.environ["OMP_NUM_THREADS"] = "2"',
        "imports": "import numpy as np",
        "body": "\n".join(lines),
        "inputs": [array("matrix", [n, d], dtype)],
        "knobs": {"op": op, "mmap": mmap, "dtype": dtype, "d": d,
                  "sample": sample if op == "pairwise_sample" else None},
    }


# ── 8. sklearn model fit ────────────────────────────────────────────────


def _sklearn_fit(rng: Random) -> dict:
    n_feat = rng.choice([8, 16, 32, 64])
    model = rng.choice(["rf", "hgb", "logreg", "kmeans", "knn"])
    rows = int(_log_uniform(rng, 2e4, 4e5) if model in ("rf", "knn") else _log_uniform(rng, 1e5, 2e6))
    n_jobs = rng.choice([1, 2, 3])
    cols = _floats("x", n_feat) + [col("label", "i32", distinct=2)]
    fit = {
        "rf": f"RandomForestClassifier(n_estimators={rng.choice([20, 50, 100])}, "
              f"max_depth={rng.choice(['None', '12', '20'])}, n_jobs={n_jobs}, random_state=0)",
        "hgb": f"HistGradientBoostingClassifier(max_iter={rng.choice([50, 100, 200])})",
        "logreg": "LogisticRegression(max_iter=200)",
        "kmeans": f"KMeans(n_clusters={rng.choice([8, 64, 256])}, n_init=1, random_state=0)",
        "knn": f"KNeighborsClassifier(n_neighbors=15, n_jobs={n_jobs})",
    }[model]
    imp = {
        "rf": "from sklearn.ensemble import RandomForestClassifier",
        "hgb": "from sklearn.ensemble import HistGradientBoostingClassifier",
        "logreg": "from sklearn.linear_model import LogisticRegression",
        "kmeans": "from sklearn.cluster import KMeans",
        "knn": "from sklearn.neighbors import KNeighborsClassifier",
    }[model]
    lines = [
        "df = pd.read_parquet(train_path)",
        'X = df.drop(columns=["label"]).to_numpy()',
        'y = df["label"].to_numpy()',
        f"model = {fit}",
    ]
    if model == "kmeans":
        lines += ["model.fit(X)", 'return {"inertia": float(model.inertia_)}']
    elif model == "knn":
        lines += ["model.fit(X, y)", "pred = model.predict(X[:20_000])", 'return {"acc": float((pred == y[:20_000]).mean())}']
    else:
        lines += ["model.fit(X, y)", 'return {"acc": float(model.score(X[:50_000], y[:50_000]))}']
    return {
        "func_name": "train_classifier",
        "preamble": 'import os\nos.environ["OMP_NUM_THREADS"] = "2"',
        "imports": f"import pandas as pd\n{imp}",
        "body": "\n".join(lines),
        "inputs": [table("train", "parquet", rows, cols)],
        "knobs": {"model": model, "n_jobs": n_jobs, "n_feat": n_feat, "spec": fit},
    }


# ── 9. TF-IDF: sparse vs .toarray() ─────────────────────────────────────


def _tfidf_text(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 2e4, 4e5))
    words = rng.choice([20, 60, 150])
    max_features = rng.choice([2_000, 10_000, 50_000])
    ngram = rng.choice([(1, 1), (1, 2)])
    dense = rng.random() < 0.4
    svd = not dense and rng.random() < 0.5
    cols = [col("doc_id", "i64", distinct=rows), col("text", "text", words=words, vocab=30_000)]
    lines = [
        'docs = pd.read_parquet(docs_path)["text"]',
        f"vec = TfidfVectorizer(max_features={max_features:_}, ngram_range={ngram})",
        "M = vec.fit_transform(docs)",
    ]
    if dense:
        lines += ["M = M.toarray()", 'return {"nnz_mean": float((M > 0).sum(axis=1).mean())}']
    elif svd:
        lines += ["emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(M)",
                  'return {"dim": int(emb.shape[1])}']
    else:
        lines += ['return {"nnz": int(M.nnz)}']
    imports = "import pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer"
    if svd:
        imports += "\nfrom sklearn.decomposition import TruncatedSVD"
    return {
        "func_name": "embed_documents",
        "imports": imports,
        "body": "\n".join(lines),
        "inputs": [table("docs", "parquet", rows, cols)],
        "knobs": {"dense": dense, "svd": svd, "max_features": max_features, "ngram": list(ngram), "words": words},
    }


# ── 10. torch training on CPU ───────────────────────────────────────────


def _torch_train(rng: Random) -> dict:
    n_feat = rng.choice([16, 64, 128])
    rows = int(_log_uniform(rng, 1e5, 2e6))
    hidden = rng.choice([256, 1024, 2048, 4096])
    depth = rng.choice([2, 4, 6])
    vocab = rng.choice([0, 0, 100_000, 1_000_000, 4_000_000])
    emb = rng.choice([32, 64, 128])
    batch = rng.choice([256, 2048, 16384])
    full_tensor = rng.random() < 0.6
    cols = _floats("x", n_feat) + [col("item", "i64", distinct=max(vocab, 1000)), col("y", "f32")]
    width = n_feat + (emb if vocab else 0)
    model = [
        "layers = []",
        f"width = {width}",
        f"for _ in range({depth}):",
        f"    layers += [torch.nn.Linear(width, {hidden}), torch.nn.ReLU()]",
        f"    width = {hidden}",
        "layers.append(torch.nn.Linear(width, 1))",
        "mlp = torch.nn.Sequential(*layers)",
    ]
    if vocab:
        model += [f"embed = torch.nn.Embedding({vocab:_}, {emb})",
                  "opt = torch.optim.AdamW([*mlp.parameters(), *embed.parameters()], lr=1e-3)"]
    else:
        model += ["embed = None", "opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)"]
    item_expr = f'df["item"].to_numpy() % {vocab:_}' if vocab else 'df["item"].to_numpy() * 0'
    if full_tensor:
        data = [
            "df = pd.read_parquet(train_path)",
            'X = torch.tensor(df[[c for c in df.columns if c.startswith("x")]].to_numpy(), dtype=torch.float32)',
            f"items = torch.tensor({item_expr})",
            'y = torch.tensor(df["y"].to_numpy())',
            "del df",
            "batches = [(X[i:i + B], items[i:i + B], y[i:i + B]) for i in range(0, min(len(X), 6 * B), B)]",
        ]
    else:
        data = [
            "pf = pq.ParquetFile(train_path)",
            "def gen():",
            "    for rb in pf.iter_batches(batch_size=B):",
            "        d = rb.to_pandas()",
            '        xs = torch.tensor(d[[c for c in d.columns if c.startswith("x")]].to_numpy(), dtype=torch.float32)',
            f'        yield xs, torch.tensor({item_expr.replace("df[", "d[")}), torch.tensor(d["y"].to_numpy())',
            "batches = itertools.islice(gen(), 6)",
        ]
    loop = [
        "for xb, ib, yb in batches:",
        "    feats = torch.cat([xb, embed(ib)], dim=1) if embed is not None else xb",
        "    loss = torch.nn.functional.mse_loss(mlp(feats).squeeze(-1), yb)",
        "    opt.zero_grad()",
        "    loss.backward()",
        "    opt.step()",
        'return {"loss": float(loss.detach())}',
    ]
    return {
        "func_name": "train_ranker",
        "preamble": 'import os\nos.environ["OMP_NUM_THREADS"] = "2"',
        "imports": "import itertools\nimport pandas as pd\nimport pyarrow.parquet as pq\nimport torch",
        "body": "\n".join([f"B = {batch}", "torch.set_num_threads(2)"] + model + data + loop),
        "inputs": [table("train", "parquet", rows, cols)],
        "knobs": {"hidden": hidden, "depth": depth, "vocab": vocab, "emb": emb, "batch": batch,
                  "full_tensor": full_tensor},
    }


# ── 11. image batches: uint8 -> float32 blowup ──────────────────────────


def _image_batch(rng: Random) -> dict:
    side = rng.choice([32, 64, 128, 224])
    n = int(_log_uniform(rng, 2e8, 3e9) / (side * side * 3))
    mode = rng.choice(["full_float", "batched", "mmap_batched", "full_uint8"])
    batch = rng.choice([256, 1024, 4096])
    load = 'np.load(images_path, mmap_mode="r")' if mode == "mmap_batched" else "np.load(images_path)"
    lines = [f"imgs = {load}"]
    if mode == "full_float":
        lines += ["x = imgs.astype(np.float32) / 255.0", "x -= x.mean(axis=(0, 1, 2))",
                  "feats = x.mean(axis=(1, 2))", 'return {"n": int(len(feats))}']
    elif mode == "full_uint8":
        lines += ["hist = np.stack([np.bincount(imgs[..., c].ravel(), minlength=256) for c in range(3)])",
                  'return {"peak": int(hist.argmax())}']
    else:
        lines += ["feats = []", f"for i in range(0, len(imgs), {batch}):",
                  f"    x = np.asarray(imgs[i:i + {batch}], dtype=np.float32) / 255.0",
                  "    feats.append(x.mean(axis=(1, 2)))", "feats = np.concatenate(feats)",
                  'return {"n": int(len(feats))}']
    return {
        "func_name": "image_features",
        "preamble": 'import os\nos.environ["OMP_NUM_THREADS"] = "2"',
        "imports": "import numpy as np",
        "body": "\n".join(lines),
        "inputs": [array("images", [n, side, side, 3], "uint8")],
        "knobs": {"mode": mode, "side": side, "batch": batch if "batched" in mode else None},
    }


# ── 12. plain Python grouping ───────────────────────────────────────────


def _python_grouping(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 2e5, 3e6))
    keys = rng.choice([100, 10_000, 1_000_000])
    n_str = rng.choice([1, 3])
    mode = rng.choice(["counter", "group_rows", "group_values"])
    cols = [col("key", "i64", distinct=keys), col("value", "f64")] + _strs("note", n_str, 20)
    if mode == "counter":
        inner = ["        counts[row[0]] += 1"]
        init = ["counts = Counter()"]
        ret = 'return {"keys": len(counts)}'
    elif mode == "group_rows":
        inner = ["        groups[row[0]].append(row)"]
        init = ["groups = defaultdict(list)"]
        ret = 'return {"keys": len(groups), "largest": max(len(v) for v in groups.values())}'
    else:
        inner = ["        groups[row[0]].append(float(row[1]))"]
        init = ["groups = defaultdict(list)"]
        ret = 'return {"keys": len(groups)}'
    body = init + [
        "with open(rows_path, newline='') as f:",
        "    reader = csv.reader(f)",
        "    next(reader)",
        "    for row in reader:",
    ] + inner + [ret]
    return {
        "func_name": "group_rows",
        "imports": "import csv\nfrom collections import Counter, defaultdict",
        "body": "\n".join(body),
        "inputs": [table("rows", "csv", rows, cols)],
        "knobs": {"mode": mode, "keys": keys, "n_str": n_str},
    }


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family("pandas_groupby", "both", _pandas_groupby, ("pandas", "pyarrow")),
        Family("pandas_join", "both", _pandas_join, ("pandas", "pyarrow")),
        Family("polars_pipeline", "code", _polars_pipeline, ("polars",)),
        Family("duckdb_sql", "code", _duckdb_sql, ("duckdb", "pandas")),
        Family("csv_chunked", "code", _csv_chunked, ("pandas",)),
        Family("jsonl_records", "both", _jsonl_records, ("pandas",)),
        Family("numpy_linalg", "both", _numpy_linalg, ("numpy",)),
        Family("sklearn_fit", "both", _sklearn_fit, ("scikit-learn", "pandas", "pyarrow")),
        Family("tfidf_text", "both", _tfidf_text, ("scikit-learn", "pandas", "pyarrow")),
        Family("torch_train", "both", _torch_train, ("torch", "pandas", "pyarrow")),
        Family("image_batch", "both", _image_batch, ("numpy",)),
        Family("python_grouping", "both", _python_grouping, ()),
    )
}

#: Families no arm trains on: the out-of-distribution test. Chosen because
#: the skill they need (reading an engine's memory semantics, spotting a
#: dtype blowup) is one the in-distribution families also teach.
HELDOUT_FAMILIES: tuple[str, ...] = ("duckdb_sql", "image_batch")


def make_task(family: str, seed: int, n_runs: int = 3) -> Task:
    fam = FAMILIES[family]
    rng = Random(seed)
    spec = fam.make(rng)
    lo, hi = fam.size_range
    factors = sorted(_log_uniform(rng, lo, hi) for _ in range(n_runs))
    rng.shuffle(factors)  # run order is not size order: history can be bigger or smaller
    digest = hashlib.sha1(f"{family}:{seed}".encode()).hexdigest()[:10]
    return Task(
        task_id=f"{family}-{digest}",
        family=family,
        memory_driver=fam.memory_driver,
        func_name=spec["func_name"],
        imports=spec["imports"],
        body=spec["body"],
        inputs=spec["inputs"],
        knobs=spec["knobs"],
        size_factors=factors,
        packages=fam.packages,
        preamble=spec.get("preamble", ""),
    )


# ── rendering ───────────────────────────────────────────────────────────


def render_task(task: Task) -> str:
    """The Flyte module an arm is asked to size. No resources declared:
    the request is exactly what is missing."""
    args = ", ".join(f"{s['name']}: File" for s in task.inputs)
    downloads = "\n".join(f"    {s['name']}_path = await {s['name']}.download()" for s in task.inputs)
    pkgs = ", ".join(f'"{p}"' for p in task.packages)
    head = f"{task.preamble}\n" if task.preamble else ""
    image = f"flyte.Image.from_debian_base().with_pip_packages({pkgs})" if pkgs else "flyte.Image.from_debian_base()"
    return (
        f"{head}import flyte\nfrom flyte.io import File\n{task.imports}\n\n"
        f'env = flyte.TaskEnvironment(name="{task.family.replace("_", "-")}", image={image})\n\n\n'
        f"@env.task\nasync def {task.func_name}({args}) -> dict:\n{downloads}\n{_ind(task.body)}\n"
    )


def render_harness(task: Task) -> str:
    """The same body as a plain function of local paths -- what executes."""
    args = ", ".join(f"{s['name']}_path" for s in task.inputs)
    head = f"{task.preamble}\n" if task.preamble else ""
    return f"{head}{task.imports}\n\n\ndef main({args}):\n{_ind(task.body)}\n"


def _fmt_bytes(n: float) -> str:
    for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{int(n)} B"


_KIND_NAMES = {"f64": "double", "f32": "float", "i64": "int64", "i32": "int32",
               "cat": "string (dictionary)", "str": "string", "text": "string"}


def describe_input(spec: dict, size_bytes: int | None) -> str:
    """One input as a scheduler sees it: format metadata only.

    - parquet: the footer gives rows and the schema
    - csv: only the file size and the header line
    - jsonl: only the file size and the keys of the first record
    - npy: the header gives shape and dtype
    """
    size = _fmt_bytes(size_bytes) if size_bytes else "unknown size"
    name, fmt = spec["name"], spec["fmt"]
    if fmt == "npy":
        return f"{name}: npy, {size}, shape={tuple(spec['shape'])}, dtype={spec['dtype']}"
    names = [c["name"] for c in spec["columns"]]
    if fmt == "parquet":
        schema = ", ".join(f"{c['name']}: {_KIND_NAMES[c['kind']]}" for c in spec["columns"])
        return f"{name}: parquet, {size}, {spec['rows']:,} rows, schema=[{schema}]"
    if fmt == "csv":
        return f"{name}: csv, {size}, header: {','.join(names)}"
    return f"{name}: jsonl, {size}, first record keys: {names}"


def describe_inputs(inputs: list[dict], sizes: dict[str, int] | None) -> str:
    sizes = sizes or {}
    return "\n".join(describe_input(s, sizes.get(s["name"])) for s in inputs)
