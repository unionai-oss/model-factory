"""Stage (and optionally push) a RightSize Bench release to the Hugging Face Hub.

    # dry run: stage every repo in the cluster, push nothing, print where each one is
    uv run python hub_release.py --release k8s-2 --name rt-hub-k8s-2-stage-1

    # push, private, after reviewing the staged repos (the staging tasks cache-hit)
    uv run python hub_release.py --release k8s-2 --push --name rt-hub-k8s-2-push-1

    # make the repos public once they look right (or flip them in the Hub UI)
    uv run python hub_release.py --release k8s-2 --push --public --name rt-hub-k8s-2-public-1

The release's trainings are factory runs (``RUNS``): seed 1's outputs go to each
model repo's ``main``, seeds 2 and 3 to branches. The laptop only resolves those
runs' outputs and launches ``hub.release``; everything is read and written in
the cluster. Runbook: docs/huggingface-release.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import flyte
from flyte.remote import Action

flyte.init_from_config(".flyte/config.yaml")
sys.path.insert(0, ".")
from resource_tuner.bench_all.hub import release  # noqa: E402

#: release -> {seed: factory run}
RUNS = {"k8s-2": {1: "rsk-k8s-2", 2: "rsk-k8s-2-s2", 3: "rsk-k8s-2-s3"}}
ARM_TASKS = ("fit_all_arm", "rl_all_policy", "pool_all_arms", "gate_all_arms")


def sync(x):
    return asyncio.run(x) if asyncio.iscoroutine(x) else x


def outputs_of(run: str):
    """The run's corpus and every arm-shaped output (a Dir with meta.json and predictions.parquet)."""
    corpus, dirs = None, []
    for a in Action.listall(for_run_name=run):
        t = str(a.task_name)
        if "SUCCEEDED" not in str(a.phase) or not t.endswith(("join_corpora",) + ARM_TASKS):
            continue
        o = sync(sync(a.details()).outputs())
        v = o[0] if isinstance(o, (list, tuple)) else o
        if t.endswith("join_corpora"):
            corpus = v
        else:
            dirs.append(v)
    return corpus, dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="k8s-2", choices=sorted(RUNS))
    ap.add_argument("--name", required=True, help="the Flyte run name (unique per launch)")
    ap.add_argument("--push", action="store_true", help="upload to the Hub (default: stage only)")
    ap.add_argument("--public", action="store_true", help="create the repos public (default: private)")
    ap.add_argument("--no-ablations", action="store_true", help="skip the Job-only and Flyte-only decision models")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()

    corpus, dirs, seeds = None, [], []
    for seed in map(int, args.seeds.split(",")):
        run = RUNS[args.release][seed]
        c, ds = outputs_of(run)
        corpus = corpus or c
        dirs += ds
        seeds += [seed] * len(ds)
        print(f"seed {seed}: {run}: {len(ds)} arm outputs", flush=True)
    if corpus is None:
        sys.exit("no corpus found among the release's runs")
    r = flyte.with_runcontext(name=args.name).run(
        release, corpus=corpus, arm_dirs=dirs, arm_seeds=seeds, release=args.release,
        push_to_hub=args.push, private=not args.public, ablations=not args.no_ablations)
    print(r.url, flush=True)
    r.wait(quiet=True)
    o = sync(r.outputs())
    print(json.dumps(json.loads(o[0] if isinstance(o, (list, tuple)) else o), indent=2))


if __name__ == "__main__":
    main()
