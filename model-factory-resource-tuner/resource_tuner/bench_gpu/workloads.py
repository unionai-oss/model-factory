"""RightSizeGPU workload families: PyTorch code whose GPU memory and
utilization the code decides.

Same shape as the other benches (code + inputs + hidden knobs). What sets
GPU memory here is what an ML engineer reasons about: parameter count x
(weights + gradients + optimizer state -- Adam keeps two more copies),
activations (batch x sequence x width x depth, and logits over the
vocabulary), mixed precision, gradient checkpointing, a dataset copied to
the GPU whole versus streamed, outputs accumulated on the device, and the
block size of a chunked computation. Utilization is set by whether the GPU
is fed: tiny models, small batches, and per-sample Python augmentation on
a CPU dataloader leave it idle.

Every workload runs until ``TIME_BUDGET_S`` (120-180 s): 4-6 samples of
Union's 30 s DCGM interval. ``driver`` names what an arm had to read:

- ``params``       weights + grads + optimizer state dominate
- ``activations``  batch x shape x depth dominate (or logits over a vocabulary)
- ``data``         where the dataset lives (whole on GPU vs streamed) dominates
- ``blocks``       a chunked computation: the block size sets the peak
- ``input_bound``  the GPU waits on a CPU dataloader: low utilization

Code runs on CPU when no GPU is present, so the tests can execute it.
"""

from __future__ import annotations

import hashlib
from random import Random

from ..bench.workloads import Family, Task, _floats, _log_uniform, array, col, table

_BUDGETS = (120, 150, 180)
_SIZES = (0.5, 2.0)

_HEAD = [
    'device = "cuda" if torch.cuda.is_available() else "cpu"',
]


def _budget(rng: Random) -> str:
    return f'TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "{rng.choice(_BUDGETS)}"))'


def _amp_lines(amp: bool) -> list[str]:
    return [f"AMP = {amp} and device == \"cuda\"",
            'scaler = torch.amp.GradScaler("cuda", enabled=AMP)']


# ── 1. tabular MLP training ─────────────────────────────────────────────


def _mlp_tabular(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 2e5, 2e6))
    n_feat = rng.choice([32, 64, 128, 256])
    width = rng.choice([512, 2048, 4096, 8192])
    depth = rng.choice([2, 4, 6])
    batch = rng.choice([512, 4096, 32768])
    amp = rng.random() < 0.5
    on_gpu = rng.random() < 0.5
    opt = rng.choice(["adam", "sgd"])
    cols = _floats("f", n_feat, "f32") + [col("target", "f32")]
    layers = " ".join(f"torch.nn.Linear({width}, {width}), torch.nn.ReLU()," for _ in range(depth - 1))
    body = [_budget(rng), *_HEAD, *_amp_lines(amp),
            "df = pd.read_parquet(train_path)",
            'X = torch.from_numpy(df.drop(columns="target").to_numpy())',
            'y = torch.from_numpy(df["target"].to_numpy())[:, None]',
            *(["X, y = X.to(device), y.to(device)  # whole dataset on the GPU"] if on_gpu else []),
            f"model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], {width}), torch.nn.ReLU(), {layers} "
            f"torch.nn.Linear({width}, 1)).to(device)",
            ("opt = torch.optim.Adam(model.parameters(), lr=1e-3)" if opt == "adam"
             else "opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)"),
            "t0 = time.monotonic()", "steps = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    i = (steps * {batch}) % max(1, len(X) - {batch})",
            f"    xb, yb = X[i : i + {batch}].to(device, non_blocking=True), y[i : i + {batch}].to(device, non_blocking=True)",
            "    with torch.autocast(device_type=device, dtype=torch.float16, enabled=AMP):",
            "        loss = torch.nn.functional.mse_loss(model(xb).float(), yb)",
            "    opt.zero_grad(set_to_none=True)",
            "    scaler.scale(loss).backward()", "    scaler.step(opt)", "    scaler.update()", "    steps += 1",
            'return {"steps": steps}']
    return {"func_name": "train_regressor", "imports": "import os\nimport time\nimport pandas as pd\nimport torch",
            "body": "\n".join(body), "inputs": [table("train", "parquet", rows, cols)],
            "knobs": {"width": width, "depth": depth, "batch": batch, "amp": amp, "on_gpu": on_gpu, "opt": opt}}


# ── 2. CNN image training ───────────────────────────────────────────────


def _cnn_train(rng: Random) -> dict:
    side = rng.choice([64, 96, 128, 224])
    n = int(_log_uniform(rng, 2e3, 1.5e4) * (64 / side) ** 2 * 4)
    base = rng.choice([32, 64, 128])
    blocks = rng.choice([3, 4, 5])
    batch = rng.choice([32, 64, 128, 256])
    amp = rng.random() < 0.5
    channels_last = rng.random() < 0.4
    preload = rng.random() < 0.4
    convs = []
    c_in = 3
    for b in range(blocks):
        c_out = base * 2 ** min(b, 3)
        convs.append(f"torch.nn.Conv2d({c_in}, {c_out}, 3, padding=1), torch.nn.BatchNorm2d({c_out}), torch.nn.ReLU(),"
                     f" torch.nn.Conv2d({c_out}, {c_out}, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),")
        c_in = c_out
    net = "torch.nn.Sequential(" + " ".join(convs) + f" torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear({c_in}, 10))"
    body = [_budget(rng), *_HEAD, *_amp_lines(amp),
            "images = torch.from_numpy(np.load(images_path))  # (N, H, W, 3) uint8",
            *(["images = images.to(device)  # the whole uint8 dataset on the GPU"] if preload else []),
            f"model = {net}.to(device)",
            *(["model = model.to(memory_format=torch.channels_last)"] if channels_last else []),
            "opt = torch.optim.AdamW(model.parameters(), lr=1e-3)",
            "t0 = time.monotonic()", "steps = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    i = (steps * {batch}) % max(1, len(images) - {batch})",
            f"    xb = images[i : i + {batch}].to(device).permute(0, 3, 1, 2).float().div_(255)",
            *(["    xb = xb.contiguous(memory_format=torch.channels_last)"] if channels_last else []),
            "    yb = torch.randint(0, 10, (len(xb),), device=device)",
            "    with torch.autocast(device_type=device, dtype=torch.float16, enabled=AMP):",
            "        loss = torch.nn.functional.cross_entropy(model(xb).float(), yb)",
            "    opt.zero_grad(set_to_none=True)",
            "    scaler.scale(loss).backward()", "    scaler.step(opt)", "    scaler.update()", "    steps += 1",
            'return {"steps": steps}']
    return {"func_name": "train_classifier", "imports": "import os\nimport time\nimport numpy as np\nimport torch",
            "body": "\n".join(body), "inputs": [array("images", [n, side, side, 3], "uint8")],
            "knobs": {"side": side, "base": base, "blocks": blocks, "batch": batch, "amp": amp,
                      "channels_last": channels_last, "preload": preload}}


# ── 3. GPT-style language model training ────────────────────────────────


def _gpt_train(rng: Random) -> dict:
    seq = rng.choice([256, 512, 1024])
    n = int(_log_uniform(rng, 4e3, 4e4) * 512 / seq)
    d = rng.choice([256, 512, 768, 1024])
    layers = rng.choice([4, 6, 8, 12])
    vocab = rng.choice([8192, 32000, 50257])
    batch = rng.choice([4, 8, 16, 32])
    amp = rng.random() < 0.6
    ckpt = rng.random() < 0.3
    opt = rng.choice(["adamw", "adamw", "sgd"])
    block = [
        "class Block(torch.nn.Module):",
        "    def __init__(self, d):",
        "        super().__init__()",
        "        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)",
        "        self.attn = torch.nn.MultiheadAttention(d, d // 64, batch_first=True)",
        "        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))",
        "    def forward(self, x, mask):",
        "        h = self.ln1(x)",
        "        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]",
        "        return x + self.mlp(self.ln2(x))",
    ]
    fwd = ("x = torch.utils.checkpoint.checkpoint(blk, x, mask, use_reentrant=False)" if ckpt else "x = blk(x, mask)")
    body = [_budget(rng), *_HEAD, *_amp_lines(amp), *block,
            f"VOCAB, D, LAYERS = {vocab}, {d}, {layers}",
            "tokens = torch.from_numpy(np.load(tokens_path).astype(np.int64))  # (N, seq + 1)",
            "seq = tokens.shape[1] - 1",
            "emb = torch.nn.Embedding(VOCAB, D).to(device)",
            "pos = torch.nn.Parameter(torch.zeros(1, seq, D, device=device))",
            "blocks = torch.nn.ModuleList([Block(D) for _ in range(LAYERS)]).to(device)",
            "head = torch.nn.Linear(D, VOCAB, bias=False).to(device)",
            "params = [*emb.parameters(), pos, *blocks.parameters(), *head.parameters()]",
            ("opt = torch.optim.AdamW(params, lr=3e-4)" if opt == "adamw" else "opt = torch.optim.SGD(params, lr=1e-2)"),
            "mask = torch.triu(torch.full((seq, seq), float('-inf'), device=device), 1)",
            "t0 = time.monotonic()", "steps = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    i = (steps * {batch}) % max(1, len(tokens) - {batch})",
            f"    xb = tokens[i : i + {batch}].to(device)",
            "    with torch.autocast(device_type=device, dtype=torch.float16, enabled=AMP):",
            "        x = emb(xb[:, :-1]) + pos",
            "        for blk in blocks:",
            f"            {fwd}",
            "        logits = head(x)",
            "        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, VOCAB), xb[:, 1:].reshape(-1))",
            "    opt.zero_grad(set_to_none=True)",
            "    scaler.scale(loss).backward()", "    scaler.step(opt)", "    scaler.update()", "    steps += 1",
            'return {"steps": steps}']
    return {"func_name": "train_lm", "imports": "import os\nimport time\nimport numpy as np\nimport torch\nimport torch.utils.checkpoint",
            "body": "\n".join(body), "inputs": [{**array("tokens", [n, seq + 1], "int32"), "high": vocab}],
            "knobs": {"d": d, "layers": layers, "vocab": vocab, "seq": seq, "batch": batch, "amp": amp,
                      "ckpt": ckpt, "opt": opt}}


# ── 4. encoder inference (embedding extraction) ─────────────────────────


def _encoder_infer(rng: Random) -> dict:
    seq = rng.choice([128, 256, 512])
    n = int(_log_uniform(rng, 2e4, 2e5) * 128 / seq)
    d = rng.choice([384, 768, 1024])
    layers = rng.choice([6, 12, 24])
    batch = rng.choice([32, 128, 512])
    half = rng.random() < 0.5
    no_grad = rng.random() < 0.7
    keep_on_gpu = rng.random() < 0.4
    body = [_budget(rng), *_HEAD,
            "tokens = torch.from_numpy(np.load(tokens_path).astype(np.int64))  # (N, seq)",
            f"layer = torch.nn.TransformerEncoderLayer({d}, {d // 64}, 4 * {d}, batch_first=True)",
            f"encoder = torch.nn.TransformerEncoder(layer, {layers}, enable_nested_tensor=False)",
            f"model = torch.nn.Sequential(torch.nn.Embedding(30522, {d}), encoder).to(device).eval()",
            *(["model = model.half()"] if half else []),
            "outputs = []", "t0 = time.monotonic()", "batches = 0",
            "ctx = torch.inference_mode() if " + str(no_grad) + " else contextlib.nullcontext()",
            "with ctx:",
            "    while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"        i = (batches * {batch}) % max(1, len(tokens) - {batch})",
            f"        emb = model(tokens[i : i + {batch}].to(device)).mean(dim=1)",
            ("        outputs.append(emb)  # kept on the device" if keep_on_gpu else "        outputs.append(emb.float().cpu())"),
            "        batches += 1",
            'return {"batches": batches, "rows": sum(len(o) for o in outputs)}']
    return {"func_name": "embed_corpus", "imports": "import contextlib\nimport os\nimport time\nimport numpy as np\nimport torch",
            "body": "\n".join(body), "inputs": [{**array("tokens", [n, seq], "int32"), "high": 30522}],
            "knobs": {"d": d, "layers": layers, "seq": seq, "batch": batch, "half": half, "no_grad": no_grad,
                      "keep_on_gpu": keep_on_gpu}}


# ── 5. recommender embeddings (held out) ────────────────────────────────


def _recsys_train(rng: Random) -> dict:
    rows = int(_log_uniform(rng, 1e6, 6e6))
    users = rng.choice([100_000, 1_000_000, 4_000_000])
    items = rng.choice([50_000, 500_000, 2_000_000])
    dim = rng.choice([32, 64, 128, 256])
    sparse = rng.random() < 0.4
    batch = rng.choice([4096, 65536])
    cols = [col("user", "i64", distinct=users), col("item", "i64", distinct=items), col("clicked", "f32")]
    opt = ("opt = torch.optim.SparseAdam([*U.parameters(), *V.parameters()], lr=1e-2)" if sparse
           else "opt = torch.optim.Adam([*U.parameters(), *V.parameters()], lr=1e-2)")
    body = [_budget(rng), *_HEAD,
            "df = pd.read_parquet(clicks_path)",
            'u = torch.from_numpy(df["user"].to_numpy())',
            'v = torch.from_numpy(df["item"].to_numpy())',
            'y = torch.from_numpy((df["clicked"].to_numpy() > 0).astype("float32"))',
            f"NUM_USERS, NUM_ITEMS, DIM = {users}, {items}, {dim}",
            f"U = torch.nn.Embedding(NUM_USERS, DIM, sparse={sparse}).to(device)",
            f"V = torch.nn.Embedding(NUM_ITEMS, DIM, sparse={sparse}).to(device)",
            opt, "t0 = time.monotonic()", "steps = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    i = (steps * {batch}) % max(1, len(y) - {batch})",
            f"    ub, vb, yb = (t[i : i + {batch}].to(device) for t in (u, v, y))",
            "    logit = (U(ub) * V(vb)).sum(-1)",
            "    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)",
            "    opt.zero_grad()", "    loss.backward()", "    opt.step()", "    steps += 1",
            'return {"steps": steps}']
    return {"func_name": "train_two_tower", "imports": "import os\nimport time\nimport pandas as pd\nimport torch",
            "body": "\n".join(body), "inputs": [table("clicks", "parquet", rows, cols)],
            "knobs": {"users": users, "items": items, "dim": dim, "sparse": sparse, "batch": batch}}


# ── 6. input-bound training (a CPU dataloader feeds the GPU) ────────────


def _loader_bound(rng: Random) -> dict:
    n = int(_log_uniform(rng, 4e3, 3e4))
    side = rng.choice([64, 96, 128])
    workers = rng.choice([0, 1, 2])
    batch = rng.choice([32, 64, 128])
    aug = rng.choice(["flip", "crop_noise", "per_pixel_python"])
    width = rng.choice([16, 64, 256])
    aug_code = {
        "flip": ["        img = self.images[i][:, ::-1] if i % 2 else self.images[i]"],
        "crop_noise": ["        img = self.images[i][4:-4, 4:-4].astype(np.float32)",
                       "        img = img + np.random.normal(0, 3, img.shape).astype(np.float32)"],
        "per_pixel_python": ["        img = self.images[i].copy()",
                             "        for r in range(img.shape[0]):",
                             "            img[r, :, 0] = [min(255, int(p) + 3) for p in img[r, :, 0]]"],
    }[aug]
    ds = ["class Augmented(torch.utils.data.Dataset):",
          "    def __init__(self, path):", "        self.images = np.load(path, mmap_mode='r')",
          "    def __len__(self):", "        return len(self.images)",
          "    def __getitem__(self, i):", *aug_code,
          "        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255"]
    body = [_budget(rng), *_HEAD,
            f"loader = torch.utils.data.DataLoader(Augmented(images_path), batch_size={batch}, shuffle=True, "
            f"num_workers={workers}, pin_memory=device == 'cuda', drop_last=True)",
            f"model = torch.nn.Sequential(torch.nn.Conv2d(3, {width}, 3, padding=1), torch.nn.ReLU(), "
            f"torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear({width}, 10)).to(device)",
            "opt = torch.optim.SGD(model.parameters(), lr=1e-2)",
            "t0 = time.monotonic()", "steps = 0", "done = False",
            "while not done:",
            "    for xb in loader:",
            "        xb = xb.to(device, non_blocking=True)",
            "        loss = torch.nn.functional.cross_entropy(model(xb), torch.zeros(len(xb), dtype=torch.long, device=device))",
            "        opt.zero_grad()", "        loss.backward()", "        opt.step()", "        steps += 1",
            "        if time.monotonic() - t0 > TIME_BUDGET_S:", "            done = True", "            break",
            'return {"steps": steps}']
    return {"func_name": "train_augmented", "imports": "import os\nimport time\nimport numpy as np\nimport torch",
            "body": "\n".join(body), "inputs": [array("images", [n, side, side, 3], "uint8")],
            "knobs": {"workers": workers, "batch": batch, "aug": aug, "width": width}, "helpers": "\n".join(ds)}


# ── 7. blocked similarity matrix ────────────────────────────────────────


def _blocked_gemm(rng: Random) -> dict:
    n = int(_log_uniform(rng, 2e4, 1.5e5))
    d = rng.choice([128, 256, 512, 1024])
    block = rng.choice([1024, 4096, 8192, 16384])
    half = rng.random() < 0.5
    body = [_budget(rng), *_HEAD,
            "X = torch.from_numpy(np.load(vectors_path)).to(device)",
            *(["X = X.half()"] if half else []),
            "X = torch.nn.functional.normalize(X, dim=1)",
            "t0 = time.monotonic()", "passes = 0", "hits = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    for i in range(0, len(X), {block}):",
            f"        S = X[i : i + {block}] @ X.T  # ({block}, N) similarities",
            "        hits += int((S > 0.9).sum())",
            "        if time.monotonic() - t0 > TIME_BUDGET_S:", "            break",
            "    passes += 1",
            'return {"passes": passes, "hits": hits}']
    return {"func_name": "near_duplicates", "imports": "import os\nimport time\nimport numpy as np\nimport torch",
            "body": "\n".join(body), "inputs": [array("vectors", [n, d], "float32")],
            "knobs": {"d": d, "block": block, "half": half}}


# ── 8. k-NN search over embeddings (held out) ───────────────────────────


def _knn_search(rng: Random) -> dict:
    n = int(_log_uniform(rng, 1e5, 5e5))
    d = rng.choice([64, 128, 256, 384])
    q_chunk = rng.choice([256, 1024, 4096])
    k = rng.choice([10, 100])
    cols_dtype = rng.choice(["float32", "float16"])
    body = [_budget(rng), *_HEAD,
            f"base = torch.from_numpy(np.load(index_path).astype(np.{cols_dtype})).to(device)",
            "queries = base[torch.randperm(len(base), device=device)[:50_000]]",
            "t0 = time.monotonic()", "rounds = 0",
            "while time.monotonic() - t0 < TIME_BUDGET_S:",
            f"    for i in range(0, len(queries), {q_chunk}):",
            f"        dist = torch.cdist(queries[i : i + {q_chunk}].float(), base.float())",
            f"        top = dist.topk({k}, largest=False)",
            "        if time.monotonic() - t0 > TIME_BUDGET_S:", "            break",
            "    rounds += 1",
            'return {"rounds": rounds}']
    return {"func_name": "nearest_neighbors", "imports": "import os\nimport time\nimport numpy as np\nimport torch",
            "body": "\n".join(body), "inputs": [array("index", [n, d], "float32")],
            "knobs": {"d": d, "q_chunk": q_chunk, "k": k, "dtype": cols_dtype}}


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family("mlp_tabular", "params", _mlp_tabular, ("torch", "pandas", "pyarrow"), _SIZES),
        Family("cnn_train", "activations", _cnn_train, ("torch", "numpy"), _SIZES),
        Family("gpt_train", "activations", _gpt_train, ("torch", "numpy"), _SIZES),
        Family("encoder_infer", "data", _encoder_infer, ("torch", "numpy"), _SIZES),
        Family("recsys_train", "params", _recsys_train, ("torch", "pandas", "pyarrow"), _SIZES),
        Family("loader_bound", "input_bound", _loader_bound, ("torch", "numpy"), _SIZES),
        Family("blocked_gemm", "blocks", _blocked_gemm, ("torch", "numpy"), _SIZES),
        Family("knn_search", "blocks", _knn_search, ("torch", "numpy"), _SIZES),
    )
}

#: Never trained on: embedding tables + sparse optimizers, and a chunked
#: distance computation -- each a skill the training families teach in
#: another form (optimizer state in mlp/gpt, block sizes in blocked_gemm).
HELDOUT_FAMILIES: tuple[str, ...] = ("recsys_train", "knn_search")


def make_task(family: str, seed: int, n_runs: int = 2) -> Task:
    fam = FAMILIES[family]
    rng = Random(seed)
    spec = fam.make(rng)
    lo, hi = fam.size_range
    factors = sorted(_log_uniform(rng, lo, hi) for _ in range(n_runs))
    rng.shuffle(factors)
    digest = hashlib.sha1(f"gpu:{family}:{seed}".encode()).hexdigest()[:10]
    imports = spec["imports"] + ("\n\n\n" + spec["helpers"] if spec.get("helpers") else "")
    return Task(task_id=f"{family}-{digest}", family=family, memory_driver=fam.memory_driver,
                func_name=spec["func_name"], imports=imports, body=spec["body"], inputs=spec["inputs"],
                knobs=spec["knobs"], size_factors=factors, packages=fam.packages, preamble="")
