"""Run paper.paper_audit on a finished factory run's corpus; write <dst>/<run>-audit.json.gz (round 24)."""
import asyncio, base64, os, sys, flyte
from flyte.remote import Action
flyte.init_from_config(".flyte/config.yaml")
sys.path.insert(0, ".")
from resource_tuner.bench_all.paper import paper_audit

def sync(x):
    return asyncio.run(x) if asyncio.iscoroutine(x) else x

run, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
corpus = None
for a in Action.listall(for_run_name=run):
    if str(a.task_name).endswith("join_corpora") and "SUCCEEDED" in str(a.phase):
        o = sync(sync(a.details()).outputs())
        corpus = o[0] if isinstance(o, (list, tuple)) else o
r = flyte.with_runcontext(name=name).run(paper_audit, corpus=corpus)
print(r.url, flush=True)
r.wait(quiet=True)
o = sync(r.outputs())
v = o[0] if isinstance(o, (list, tuple)) else o
with open(os.path.join(dst, f"{run}-audit.json.gz"), "wb") as fh:
    fh.write(base64.b64decode(v))
print("done", len(v))
