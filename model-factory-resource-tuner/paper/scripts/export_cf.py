"""Fetch a finished factory run's counterfactual output inline (paper.paper_file);
write it to <dst>/<run>-counterfactual.json.gz."""
import asyncio, base64, os, sys, flyte
from flyte.remote import Action
flyte.init_from_config(".flyte/config.yaml")
sys.path.insert(0, ".")
from resource_tuner.bench_all.paper import paper_file

def sync(x):
    return asyncio.run(x) if asyncio.iscoroutine(x) else x

run, dst, tag = sys.argv[1], sys.argv[2], sys.argv[3]
f = None
for a in Action.listall(for_run_name=run):
    if str(a.task_name).endswith("staging_counterfactual") and "SUCCEEDED" in str(a.phase):
        o = sync(sync(a.details()).outputs())
        f = o[0] if isinstance(o, (list, tuple)) else o
r = flyte.with_runcontext(name=f"rt-paper-{tag}-cf").run(paper_file, f=f)
print(r.url, flush=True)
r.wait(quiet=True)
o = sync(r.outputs())
v = o[0] if isinstance(o, (list, tuple)) else o
with open(os.path.join(dst, f"{run}-counterfactual.json.gz"), "wb") as fh:
    fh.write(base64.b64decode(v))
print("done", len(v))
