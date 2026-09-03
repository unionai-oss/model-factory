# model-factory

Model factory proofs-of-concept on [Flyte v2](https://www.union.ai/docs/v2/)
/ Union, one subdirectory per factory.

| project | what it is |
|---|---|
| [`basic-model-factory/`](basic-model-factory/) | A dark model factory: RL from verifiable rewards (GRPO + sandboxed unit tests) where every asset is a versioned artifact and artifact events drive the next station. |

Each project is self-contained (its own `pyproject.toml`, `uv.lock`,
`.flyte/config.yaml`, tests, and docs); run its commands from inside its
directory:

```bash
cd basic-model-factory
uv sync
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs each project's tests on every push/PR
and deploys it to its Flyte cluster on green `main`.
