"""The decision model: a multi-head classifier over the discretized grid.

A third arm, alongside the two that already exist:

| arm | how it decides | trained by |
|---|---|---|
| LLM policy (`grpo.py`) | emits `{"cpu","memory","gpu"}` as text, snapped onto the grid | GRPO on a shaped reward |
| GBT baseline (`ml_baseline.py`) | regresses a high quantile per resource, then buckets | quantile loss |
| **decision model (here)** | **chooses a grid cell directly** | **cost-weighted cross-entropy** |

The distinction from the GBT baseline is not the model family, it is the
target. The baseline predicts a *number* and rounds it; this predicts the
*action*. That matters because the thing being optimized is a choice among
finitely many requests, and the cost of being wrong is wildly asymmetric in a
way a regression loss cannot express:

- One bucket too small -> the task OOMs. Total loss of the run, plus a retry.
- One bucket too big -> you pay for memory you did not use. A few percent.

So the loss is cross-entropy weighted by a cost matrix over (true bucket,
predicted bucket): `under_penalty` multiplies every under-provisioning
mistake, and over-provisioning is charged in proportion to how many times
more resource the chosen bucket costs. `under_penalty` is the arm's main
hyperparameter — sweeping it traces out the OOM-rate / waste frontier, which
is the actual engineering decision this factory exists to inform.

Features are reused verbatim from `ml_baseline.extract_features` (sampled
params, numbers in the input profile, lexical code flags, declared prior, run
history) so the two arms are compared on identical information and any
difference is attributable to the target and the loss, not the features.

Torch is imported lazily inside the functions that need it: this module is
imported by the tune service and by unit tests that only exercise the
encoding, and the CPU task image does not carry torch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..policy.action_space import ActionSpace, get_action_space
from ..policy.actions import Proposal
from .ml_baseline import extract_features

#: Feature keys, frozen in a sorted order so a checkpoint's input layout is
#: reproducible. Stored in the manifest; a mismatch at serve time is fatal.
def feature_names(records: Sequence[dict], sample: int = 512) -> list[str]:
    """The feature layout, settled from a sample.

    `extract_features` returns a fixed key set for any record, so scanning
    every row of a million-row corpus to discover it is pure waste. The sample
    is capped but still a prefix scan rather than one row, so a generator that
    emits an extra key on some family is not missed.
    """
    keys: set[str] = set()
    for r in records[:sample] if sample else records:
        keys.update(extract_features(r).keys())
    return sorted(keys)


def featurize(records: Sequence[dict], names: Sequence[str]) -> list[list[float]]:
    """Records -> a dense matrix in ``names`` order, with log1p on big values.

    `log1p` on the magnitude-like features matters: raw byte counts span nine
    orders of magnitude, and an un-squashed input makes the first linear layer
    spend its whole dynamic range on the largest feature.
    """
    rows: list[list[float]] = []
    for r in records:
        f = extract_features(r)
        rows.append([_squash(float(f.get(k, 0.0))) for k in names])
    return rows


def to_tensor(records: Sequence[dict], names: Sequence[str]):
    """Records -> a float32 tensor, filled row by row.

    Deliberately not `torch.tensor(featurize(...))`: the intermediate
    list-of-lists stores every value as a boxed Python float (~24 bytes plus
    list overhead), which is an order of magnitude more memory than the tensor
    it becomes. At corpus scale that intermediate is what OOMs the pod.
    """
    import torch

    out = torch.empty((len(records), len(names)), dtype=torch.float32)
    for i, r in enumerate(records):
        f = extract_features(r)
        for j, k in enumerate(names):
            out[i, j] = _squash(float(f.get(k, 0.0)))
    return out


def _squash(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return math.copysign(math.log1p(abs(x)), x)


@dataclass(frozen=True)
class DecisionArm:
    """One training configuration of the decision model.

    This is the thing the factory fans out over: `arm` is a partition
    dimension, so each configuration is trained, scored and compared as a
    separate artifact instance.
    """

    name: str
    #: Which discretization to decide over (see `policy.action_space`).
    action_space: str = "default"
    #: How many times worse an under-provisioning mistake is than a
    #: same-distance over-provisioning one. The frontier knob.
    under_penalty: float = 12.0
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 128
    #: Row caps. The corpus runs to ~1e6 archetype rows, and a 256x128 MLP
    #: over 43 features saturates long before that — while materializing a
    #: million rows of `source_code` and a 1e6 x 43 feature matrix OOMs a
    #: 4Gi pod (run rt-decision-r1, exit 137 on all three arms). Capping is
    #: both a sizing fix and the honest hyperparameter: more rows is a knob,
    #: not a free good.
    max_train_rows: int = 60_000
    max_eval_rows: int = 10_000
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 17

    @property
    def space(self) -> ActionSpace:
        return get_action_space(self.action_space)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "action_space": self.action_space,
            "under_penalty": self.under_penalty,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_train_rows": self.max_train_rows,
            "max_eval_rows": self.max_eval_rows,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }


#: The suite the factory trains. Three points on the asymmetry frontier plus
#: one coarse-grid control, so a run answers both "how averse should it be?"
#: and "does grid resolution matter?" at once.
ARMS: dict[str, DecisionArm] = {
    a.name: a
    for a in (
        # Near-symmetric: what plain accuracy maximization gives you.
        DecisionArm(name="balanced", under_penalty=2.0),
        # The default: OOMs hurt roughly an order of magnitude more.
        DecisionArm(name="oom-averse", under_penalty=12.0),
        # Paranoid: should drive OOMs toward zero and waste up.
        DecisionArm(name="oom-paranoid", under_penalty=40.0),
        # Same asymmetry as the default, coarser grid.
        DecisionArm(name="coarse-grid", action_space="coarse", under_penalty=12.0),
    )
}

DEFAULT_ARMS: tuple[str, ...] = ("balanced", "oom-averse", "oom-paranoid")


def get_arm(name: str) -> DecisionArm:
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(f"unknown decision arm {name!r}; choose from {sorted(ARMS)}")


# ── the cost matrix ─────────────────────────────────────────────────────


def cost_matrix(step_ratios: Sequence[float], under_penalty: float) -> list[list[float]]:
    """``C[true][pred]``: what choosing ``pred`` costs when ``true`` is right.

    Over-provisioning costs the extra resource actually paid for, as a
    multiple of the true requirement. Under-provisioning costs
    ``under_penalty`` — a flat, large number, because the outcome (the task
    dies) does not get meaningfully worse the further under you are, and
    scaling it by distance would make a 2-bucket miss look twice as bad as the
    1-bucket miss that already lost the whole run.
    """
    n = len(step_ratios)
    matrix: list[list[float]] = []
    for t in range(n):
        row = []
        for p in range(n):
            if p == t:
                row.append(0.0)
            elif p < t:
                row.append(float(under_penalty))
            else:
                row.append(float(step_ratios[p] / step_ratios[t] - 1.0))
        matrix.append(row)
    return matrix


def gpu_cost_matrix(n: int, under_penalty: float) -> list[list[float]]:
    """Cost over GPU choices, ordered cheapest-first like `gpu_options`.

    Same shape as the memory/CPU matrices: picking a smaller card than needed
    (or CPU-only for a GPU task) is an under-provision; picking a bigger one
    wastes the price difference, approximated as one unit per step because
    pricing.py's per-card dollars are not available to the loss.
    """
    matrix: list[list[float]] = []
    for t in range(n):
        row = []
        for p in range(n):
            if p == t:
                row.append(0.0)
            elif p < t:
                row.append(float(under_penalty))
            else:
                row.append(float(p - t))
        matrix.append(row)
    return matrix


# ── the model ───────────────────────────────────────────────────────────


def build_module(n_features: int, arm: DecisionArm):
    """A shared trunk with three classification heads."""
    import torch
    from torch import nn

    space = arm.space
    n_mem, n_cpu, n_gpu = space.head_sizes

    class DecisionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = n_features
            for width in arm.hidden:
                layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(arm.dropout)]
                prev = width
            self.trunk = nn.Sequential(*layers)
            # One head per resource; the trunk is shared because the three
            # decisions are driven by the same workload features.
            self.memory_head = nn.Linear(prev, n_mem)
            self.cpu_head = nn.Linear(prev, n_cpu)
            self.gpu_head = nn.Linear(prev, n_gpu)

        def forward(self, x):
            h = self.trunk(x)
            return self.memory_head(h), self.cpu_head(h), self.gpu_head(h)

    torch.manual_seed(arm.seed)
    return DecisionNet()


def expected_cost_loss(logits, targets, costs):
    """Cost-weighted cross-entropy: sum_p P(p) * C[true][p].

    Plain cross-entropy would treat every wrong bucket alike. This takes the
    expectation of the cost matrix under the predicted distribution, so the
    gradient pushes probability mass toward the *cheap* side of the right
    answer — which is what makes `under_penalty` translate into behaviour
    rather than just a different number in a log.
    """
    import torch

    probs = torch.softmax(logits, dim=-1)
    # costs: (n_classes, n_classes); row per true class.
    row = costs[targets]  # (batch, n_classes)
    return (probs * row).sum(dim=-1).mean()


@dataclass
class TrainedDecisionModel:
    """A fitted decision model plus everything needed to decode its output."""

    arm: DecisionArm
    feature_names: list[str]
    state_dict: Any = None
    history: list[dict] = field(default_factory=list)
    module: Any = field(default=None, repr=False)

    @property
    def space(self) -> ActionSpace:
        return self.arm.space

    def predict(self, records: Sequence[dict]) -> list[Proposal]:
        """Proposals for a batch of corpus records."""
        import torch

        if not records:
            return []
        x = to_tensor(records, self.feature_names)
        self.module.eval()
        with torch.no_grad():
            mem_logits, cpu_logits, gpu_logits = self.module(x)
            mem = mem_logits.argmax(dim=-1).tolist()
            cpu = cpu_logits.argmax(dim=-1).tolist()
            gpu = gpu_logits.argmax(dim=-1).tolist()
        space = self.space
        return [space.decode(m, c, g) for m, c, g in zip(mem, cpu, gpu)]

    def manifest(self, extra: dict | None = None) -> dict:
        d = {
            "arm": self.arm.to_dict(),
            "action_space": self.space.to_dict(),
            "feature_names": list(self.feature_names),
            "final_metrics": self.history[-1] if self.history else {},
            "history": self.history,
        }
        if extra:
            d.update(extra)
        return d


def labels_from_records(records: Sequence[dict], space: ActionSpace) -> list[tuple[int, int, int]]:
    """The correct grid cell for each corpus record, from its ground truth."""
    out = []
    for r in records:
        out.append(
            space.encode_labels(
                float(r.get("true_peak_memory_mib") or 0.0),
                float(r.get("true_cpu_cores") or 0.0),
                float(r.get("true_gpu_mem_mib") or 0.0),
            )
        )
    return out


def fit(
    records: Sequence[dict],
    arm: DecisionArm,
    val_records: Sequence[dict] | None = None,
) -> TrainedDecisionModel:
    """Train the decision model on corpus records. CPU-only and fast."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not records:
        raise ValueError("no records to train on")

    space = arm.space
    names = feature_names(records)
    x = to_tensor(records, names)
    y = torch.tensor(labels_from_records(records, space), dtype=torch.long)

    mem_costs = torch.tensor(
        cost_matrix(space.memory_step_ratios(), arm.under_penalty), dtype=torch.float32
    )
    cpu_costs = torch.tensor(
        cost_matrix([float(c) for c in space.cpu_grid], arm.under_penalty), dtype=torch.float32
    )
    gpu_costs = torch.tensor(
        gpu_cost_matrix(len(space.gpu_options), arm.under_penalty), dtype=torch.float32
    )

    module = build_module(x.shape[1], arm)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=arm.learning_rate, weight_decay=arm.weight_decay
    )
    loader = DataLoader(
        TensorDataset(x, y), batch_size=arm.batch_size, shuffle=True, drop_last=False
    )

    val_x = val_y = None
    if val_records:
        val_x = to_tensor(val_records, names)
        val_y = torch.tensor(labels_from_records(val_records, space), dtype=torch.long)

    history: list[dict] = []
    for epoch in range(arm.epochs):
        module.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            mem_logits, cpu_logits, gpu_logits = module(xb)
            loss = (
                expected_cost_loss(mem_logits, yb[:, 0], mem_costs)
                + expected_cost_loss(cpu_logits, yb[:, 1], cpu_costs)
                + expected_cost_loss(gpu_logits, yb[:, 2], gpu_costs)
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        entry = {"epoch": epoch, "train_loss": total / max(len(x), 1)}
        if val_x is not None:
            entry.update(_val_metrics(module, val_x, val_y))
        history.append(entry)

    return TrainedDecisionModel(
        arm=arm, feature_names=names, state_dict=module.state_dict(), history=history, module=module
    )


def _val_metrics(module, val_x, val_y) -> dict:
    """Exact-cell accuracy and the under-provisioning rate per head."""
    import torch

    module.eval()
    with torch.no_grad():
        mem_logits, cpu_logits, gpu_logits = module(val_x)
    out = {}
    for i, (name, logits) in enumerate(
        (("memory", mem_logits), ("cpu", cpu_logits), ("gpu", gpu_logits))
    ):
        pred = logits.argmax(dim=-1)
        truth = val_y[:, i]
        out[f"val_{name}_exact"] = float((pred == truth).float().mean())
        out[f"val_{name}_under"] = float((pred < truth).float().mean())
    out["val_all_fit"] = float(
        (
            (mem_logits.argmax(-1) >= val_y[:, 0])
            & (cpu_logits.argmax(-1) >= val_y[:, 1])
            & (gpu_logits.argmax(-1) >= val_y[:, 2])
        )
        .float()
        .mean()
    )
    return out


def save(model: TrainedDecisionModel, out_dir: str, extra_manifest: dict | None = None) -> None:
    """Write weights + manifest. The manifest carries the grid and the feature
    layout, because a checkpoint is undecodable without both."""
    import os

    import torch

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict, os.path.join(out_dir, "decision_model.pt"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(model.manifest(extra_manifest), f, indent=2, default=str)


def load(model_dir: str) -> TrainedDecisionModel:
    """Rehydrate a saved decision model, grid and feature layout included."""
    import os

    import torch

    with open(os.path.join(model_dir, "manifest.json")) as f:
        manifest = json.load(f)
    arm = DecisionArm(
        name=manifest["arm"]["name"],
        action_space=manifest["arm"]["action_space"],
        under_penalty=float(manifest["arm"]["under_penalty"]),
        hidden=tuple(manifest["arm"]["hidden"]),
        dropout=float(manifest["arm"]["dropout"]),
    )
    names = list(manifest["feature_names"])
    state = torch.load(os.path.join(model_dir, "decision_model.pt"), map_location="cpu")
    module = build_module(len(names), arm)
    module.load_state_dict(state)
    module.eval()
    return TrainedDecisionModel(
        arm=arm,
        feature_names=names,
        state_dict=state,
        history=manifest.get("history", []),
        module=module,
    )
