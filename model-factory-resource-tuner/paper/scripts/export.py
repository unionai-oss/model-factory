"""Run paper_export on a finished factory run's outputs; write each part to <dst>/<run>-<part>.json.gz."""
import asyncio, base64, os, sys, flyte
from flyte.remote import Action
flyte.init_from_config(".flyte/config.yaml")
sys.path.insert(0, ".")
from resource_tuner.bench_all.paper import paper_export

def sync(x):
    return asyncio.run(x) if asyncio.iscoroutine(x) else x

run, dst, parts = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
corpus, dirs = None, []
for a in Action.listall(for_run_name=run):
    t = str(a.task_name)
    if "SUCCEEDED" not in str(a.phase):
        continue
    if t.endswith(("join_corpora", "fit_all_arm", "rl_all_policy", "pool_all_arms", "gate_all_arms")):
        o = sync(sync(a.details()).outputs())
        v = o[0] if isinstance(o, (list, tuple)) else o
        if t.endswith("join_corpora"):
            corpus = v
        else:
            dirs.append(v)
print("corpus", corpus is not None, "dirs", len(dirs))
runs = [(p, flyte.with_runcontext(name=f"rt-paper-{sys.argv[4]}-{p}").run(paper_export, corpus=corpus, arm_dirs=dirs, part=p))
        for p in parts]
for p, r in runs:
    print(p, r.url, flush=True)
for p, r in runs:
    r.wait(quiet=True)
    o = sync(r.outputs())
    v = o[0] if isinstance(o, (list, tuple)) else o
    with open(os.path.join(dst, f"{run}-{p}.json.gz"), "wb") as f:
        f.write(base64.b64decode(v))
    print(p, "done", len(v), flush=True)
