"""Linear / MLP heads for evaluation.

Some metrics require a linear readout of the features, and this module implements the heads and
their training recipes. Heads are trained on a balanced pool of pixels from the training events
and evaluated on a pool from the validation events. Those pools are drawn once, at extraction time,
by `wcfm.eval.pools`, so the same pixels serve every checkpoint and every metric by construction
rather than by every probe redrawing them with the same seed.

Training uses a fixed recipe and fixed hyperparameters, never tuned per checkpoint. The reported
scores are therefore not the best possible for each checkpoint but a fixed, comparable measure
of how much is linearly extractable.

That is a deliberate choice against the common convention of sweeping learning rates and
reporting the best macro-F1 as measured on the eval split. A tuned maximum estimates how much is
linearly extractable *at best*; a fixed recipe asks whether this checkpoint improved on that one
under one scoring procedure. The two answer different questions, so published linear-probe
numbers are not like-for-like with these.

    fit_svm    linear SVM, C fixed
    fit_mlp    Linear(D,128) -> ReLU -> Linear(128,k), Adam

Both are classifiers: every metric in the suite asks a yes/no or which-class question. A
regression head, if one is ever needed, must be the MLP at the same capacity as `fit_mlp` rather
than a linear ridge, or its numbers cannot be read beside a classification one.

Each returns predictions on the validation pool, never the fitted head: a head is trained fresh
for every (checkpoint, feature source, metric) and scores exactly one population, so there is
nothing to reuse it for.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_mlp", "fit_svm"]


def fit_svm(Xtr, ytr, Xva, seed: int):
    """Linear SVM head. Convex, so the fit is reproducible given the inputs.

    `C` is fixed and never tuned per checkpoint.

    Plain multiclass (liblinear one-vs-rest internally): measured faster than an explicit
    process-parallel `OneVsRestClassifier` at our pool size and core count, with bit-identical
    predictions.
    """
    from sklearn.svm import LinearSVC

    model = LinearSVC(C=1.0, max_iter=10000, random_state=seed).fit(Xtr, ytr)
    return model.predict(Xva)


def _predict_chunked(model, dev, X, chunk: int = 200_000):
    """Predict with a fitted MLP head, in chunks. Returns `(predictions, probabilities)`.

    Chunked so a large pool cannot blow up memory: features are float16 on disk and casting a
    whole side to float32 at once is the one step here big enough to matter. At current pool
    sizes this is a single chunk.
    """
    import torch

    preds, probs = [], []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            xb = np.ascontiguousarray(X[s : s + chunk], dtype=np.float32)
            logits = model(torch.from_numpy(xb).to(dev))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(probs)


def fit_mlp(
    Xtr,
    ytr,
    Xva,
    n_classes: int,
    seed: int,
    epochs: int = 30,
    lr: float = 5e-3,
    batch: int = 256,
    device: str = "cpu",
):
    """Small MLP head: `Linear(D,128) -> ReLU -> Linear(128,k)`, Adam, fixed budget.

    There is no early stopping: selecting the stopping point on the val pool would leak it into
    the reported score. Returns `(predictions, probabilities)`.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    dev = torch.device(device)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, n_classes)
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32)).to(dev)
    yt = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.int64)).to(dev)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            Fnn.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return _predict_chunked(model, dev, Xva)
