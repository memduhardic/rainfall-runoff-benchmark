"""
LSTM rainfall-runoff model (PyTorch, optional dependency).

- Input: sequences of the P/PET features over a look-back window.
- Target: log1p(Q), standardised with training-period statistics.
- Loss: MSE weighted towards high flows so the network does not settle on
  baseflow only.
- Early stopping on validation-period NSE; the test period is never seen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rrbench.features import build_features
from rrbench.metrics import evaluate

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class LSTMConfig:
    lookback: int = 120
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 40
    grad_clip: float = 1.0


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if torch is not None:

    class _Network(nn.Module):
        def __init__(self, n_inputs, cfg: LSTMConfig):
            super().__init__()
            self.lstm = nn.LSTM(
                n_inputs, cfg.hidden_size, cfg.num_layers, batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_size, 32), nn.ReLU(),
                nn.Dropout(cfg.dropout), nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    class _Windows(Dataset):
        """Builds each look-back window on the fly (no huge 3-D array)."""

        def __init__(self, x, y, w, idx, lookback):
            self.x, self.y, self.w = x, y, w
            self.idx, self.lookback = idx, lookback

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, k):
            i = self.idx[k]
            return self.x[i - self.lookback + 1 : i + 1], self.y[i], self.w[i]


def lstm(df: pd.DataFrame, split, seed=42, cfg: LSTMConfig | None = None, log=print):
    if torch is None:
        raise ImportError("The LSTM model needs PyTorch: pip install torch")
    cfg = cfg or LSTMConfig()
    if split.warmup < cfg.lookback - 1:
        raise ValueError("warm-up must be at least as long as the LSTM look-back window")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()

    feats = build_features(df).to_numpy(np.float32)
    q = df["Q"].to_numpy(float)
    n = len(q)

    train = np.arange(split.warmup, split.train_end)
    train = train[np.isfinite(q[train])]
    valid = np.arange(split.train_end, split.valid_end)
    valid = valid[np.isfinite(q[valid])]

    # scaling, fitted on the training rows only
    mu, sd = feats[train].mean(0), feats[train].std(0)
    sd[sd == 0] = 1.0
    x = torch.tensor((feats - mu) / sd, dtype=torch.float32)

    ylog = np.log1p(np.maximum(np.nan_to_num(q, nan=0.0), 0.0))
    y_mu, y_sd = ylog[train].mean(), ylog[train].std()
    y = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)

    q95 = np.nanpercentile(q[train], 95) or 1.0
    w = torch.tensor(1.0 + 2.0 * np.clip(np.nan_to_num(q) / q95, 0, 1),
                     dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(_Windows(x, y, w, train, cfg.lookback),
                        batch_size=cfg.batch_size, shuffle=True)
    net = _Network(x.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=cfg.patience // 3)

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(idx), 1024):
                batch = idx[start : start + 1024]
                xb = torch.stack([x[i - cfg.lookback + 1 : i + 1] for i in batch])
                out.append(net(xb.to(device)).cpu().numpy().ravel())
        pred = np.concatenate(out) * y_sd + y_mu
        return np.maximum(np.expm1(pred), 0.0)

    best_nse, best_state, stale = -np.inf, None, 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = torch.mean(wb * (net(xb) - yb) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

        valid_nse = evaluate(q[valid], predict(valid))["NSE"]
        sched.step(valid_nse if np.isfinite(valid_nse) else -1e9)
        if valid_nse > best_nse:
            best_nse, stale = valid_nse, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            log(f"    epoch {epoch:4d}  valid NSE {valid_nse:6.3f}  (best {best_nse:.3f})")
        if stale >= cfg.patience:
            log(f"    early stop at epoch {epoch}, best valid NSE {best_nse:.3f}")
            break

    net.load_state_dict(best_state)
    sim = np.full(n, np.nan)
    scored = np.arange(cfg.lookback - 1, n)
    sim[scored] = predict(scored)
    return sim
