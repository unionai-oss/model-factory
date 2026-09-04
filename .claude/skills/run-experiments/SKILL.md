---
name: run-experiments
description: Use whenever running experiments against a model factory in this repo — launching cluster runs (flyte run), training/eval pipelines, corpus or synthetic-data generation, trigger-driven dark runs, or analyzing their results. Enforces the research-log upkeep that every experiment round requires. Trigger words: experiment, run the pipeline, train, eval, smoke run, dev run, kick off, launch on the cluster, results.
---

# Running experiments in this repo

Every factory subproject keeps an auditable research log. **Experimentation
is not done until the log is updated** — a run whose outcome isn't logged
is a run that effectively didn't happen for the next person (or the next
session).

## The rule

Any experimentation against a factory — cluster runs, training, evals,
synthetic-data generation, trigger/dark-mode exercises — entails updating
the research log in **the corresponding factory directory**:

- `resource-tuner-model-factory/research_log/`
- `basic-model-factory/research_log/`
- (new factories: create `<factory>/research_log/` with a README copied
  from an existing one before the first experiment lands)

## Procedure

1. **Before a round**: skim the factory's `research_log/README.md` index
   and the latest entry — prior findings (and prior failures) shape what
   to run next. Do not re-discover a logged failure.
2. **During**: capture run names as they launch. Every run gets its Union
   console URL, base:
   `https://demo.hosted.unionai.cloud/v2/domain/development/project/<project>/runs/<run>`
3. **After a round** (or when a round ends early), write/extend an entry
   `YYYY-MM-DD-<slug>.md` containing:
   - **Context** — what question the round was asking.
   - **Run ledger** — a table of every run: link, what it was, outcome.
     Failed and aborted runs stay in the ledger; a failure that changed
     the code is a result. Link the commit/PR that fixed it.
   - **Findings** — what was learned, stated as claims the runs support.
   - **Code changes** — what the findings forced, with PR links.
4. **Update the index**: add the entry to `research_log/README.md`'s
   entries table, and refresh the standing-results table when new eval
   reports landed (metrics come from the eval-report artifacts / the
   lineage dashboard's `/api/lineage`).
5. **Commit the log with the round** — log entries ride the same branch/PR
   as the experiment's code changes, so results and code stay auditable
   together.

## Conventions (mirror the existing entries)

- One entry per experiment round, not per run.
- Never rewrite a past entry's conclusions; append corrections with a
  dated note (the log is an audit trail, not documentation).
- Include links that make results verifiable: run URLs, artifact/dashboard
  URLs, PRs, upstream issues, external docs relied on.
- Keep the reading of results honest: say what failed, what's still open,
  and which gate/target has not been met.
