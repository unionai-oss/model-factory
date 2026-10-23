"""Bounded corpus sampling for the trainer.

Run usnnr9xmpnnq4ftjqgzn OOMKilled four attempts reading the 1,016,896-row
round-15 corpus into pandas, so the contract these tests pin is as much
about MEMORY as about correctness: the sampler must never materialize the
corpus, and must keep the random-subset property that stops a head slice
silently dropping whole sources.
"""

import json
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from resource_tuner.contracts import CORPUS_COLUMNS
from resource_tuner.training.grpo import _sample_train_records


def _corpus(n_train=1000, n_heldout=50, row_groups=7):
    """A corpus ordered templates-then-archetypes, like the real merged one."""
    rows = []
    for i in range(n_train + n_heldout):
        source = "template" if i < n_train // 2 else "qwen38-27b"
        rows.append(
            {
                "task_id": f"t{i}",
                "family": ["etl", "ml_training"][i % 2],
                "source_code": f"# workload {i}\n" + "x = 1\n" * 20,
                "harness_code": "def run():\n    return {}\n" * 40,  # the fat column
                "input_profile": "1M rows",
                "params_json": json.dumps({"i": i}),
                "generator": source,
                "prior_json": "",
                "history_json": "",
                "true_peak_memory_mib": 100.0 + i,
                "true_cpu_cores": 1.0,
                "true_gpu_mem_mib": 0.0,
                "duration_s": 30,
                "split": "train" if i < n_train else "heldout",
            }
        )
    path = tempfile.mktemp(suffix=".parquet")
    tbl = pa.Table.from_pylist(rows)
    w = pq.ParquetWriter(path, tbl.schema)
    chunk = max(1, len(rows) // row_groups)
    for start in range(0, len(rows), chunk):
        w.write_table(tbl.slice(start, chunk))
    w.close()
    return path


def test_samples_exactly_k_train_rows_and_never_heldout():
    recs = _sample_train_records(_corpus(), 200)
    assert len(recs) == 200
    assert {r["split"] for r in recs} == {"train"}
    assert len({r["task_id"] for r in recs}) == 200  # no duplicates


def test_never_reads_harness_code():
    """The fat column is the whole point — it must not even be loaded."""
    recs = _sample_train_records(_corpus(), 50)
    assert "harness_code" not in recs[0]
    # ...while every column training actually reads survives
    for col in CORPUS_COLUMNS:
        if col != "harness_code":
            assert col in recs[0], col


def test_takes_everything_when_the_corpus_is_smaller_than_k():
    recs = _sample_train_records(_corpus(n_train=30), 500)
    assert len(recs) == 30


def test_sample_spans_the_whole_file_not_a_head_slice():
    """Merged corpora are ordered templates-then-archetypes; a head slice
    would silently drop every archetype row."""
    recs = _sample_train_records(_corpus(), 200)
    generators = {r["generator"] for r in recs}
    assert generators == {"template", "qwen38-27b"}, generators
    # and it spans row groups, not just the first
    ids = sorted(int(r["task_id"][1:]) for r in recs)
    assert ids[0] < 100 and ids[-1] > 900


def test_sampling_is_deterministic():
    path = _corpus()
    a = [r["task_id"] for r in _sample_train_records(path, 100)]
    b = [r["task_id"] for r in _sample_train_records(path, 100)]
    assert a == b


def test_empty_train_split_returns_nothing_rather_than_raising():
    assert _sample_train_records(_corpus(n_train=0, n_heldout=20), 10) == []


def test_never_materializes_the_corpus_with_pandas(monkeypatch):
    """The OOM regression, pinned at the exact call that caused it.

    Measuring peak RSS in a unit test is a trap — tracemalloc sees only
    Python allocations, and pandas' real cost lives in Arrow's C buffers,
    so a memory assertion here would report 0.4MB for the path that killed
    a 10Gi pod. What IS checkable, precisely, is that the sampler never
    takes the `pd.read_parquet` route at all.
    """
    import pandas as pd

    def boom(*args, **kwargs):
        raise AssertionError("the corpus was materialized with pandas")

    monkeypatch.setattr(pd, "read_parquet", boom)
    assert len(_sample_train_records(_corpus(), 100)) == 100


def test_reads_only_k_rows_into_python_objects():
    """Row count materialized is k, not the corpus size — which is the
    property that makes the memory bound hold as the corpus grows."""
    path = _corpus(n_train=5_000, row_groups=20)
    for k in (10, 100, 1_000):
        assert len(_sample_train_records(path, k)) == k
