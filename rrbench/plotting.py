"""Figures: per-model hydrographs, test scatter plots and the comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OBS_COLOR = "#12222B"
COLORS = {
    "GR4J": "#7F77DD",
    "HBV": "#D85A30",
    "RF": "#0E7C86",
    "ANN": "#C97B2E",
    "LSTM": "#378ADD",
}
PERIOD_SHADES = {"train": "#0E7C86", "valid": "#C97B2E", "test": "#7F77DD"}


def _precip_axis(ax, dates, precip, band=0.3):
    ax2 = ax.twinx()
    ax2.bar(dates, precip, width=1.0, color="#378ADD", alpha=0.5, label="Precipitation")
    pmax = np.nanmax(precip) if np.isfinite(precip).any() else 1.0
    ax2.set_ylim(pmax / band, 0)
    ax2.set_ylabel("Precipitation (mm/day)")
    return ax2


def _metrics_box(ax, m, title="Test period"):
    text = (f"{title}\nNSE  {m['NSE']:.3f}\nKGE  {m['KGE']:.3f}\n"
            f"PBIAS {m['PBIAS']:.1f}%\nRMSE {m['RMSE']:.3f}")
    ax.text(0.01, 0.97, text, transform=ax.transAxes, va="top", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#DCE4E7"))


def hydrograph(df, sim, name, split, test_metrics, path):
    dates, obs = df["Date"], df["Q"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for period, sl in split.periods().items():
        ax.axvspan(dates.iloc[sl.start], dates.iloc[sl.stop - 1],
                   color=PERIOD_SHADES[period], alpha=0.07, lw=0)
        ax.text(dates.iloc[sl.start], 1.0, f" {period}", transform=ax.get_xaxis_transform(),
                va="bottom", fontsize=8, color="#5B6C74")
    ax.plot(dates, obs, color=OBS_COLOR, lw=1.0, label="Observed")
    ax.plot(dates, sim, color=COLORS.get(name, "red"), lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_ylim(0, np.nanmax(obs) * 1.6)
    ax.grid(alpha=0.25)
    ax2 = _precip_axis(ax, dates, df["P"].to_numpy(float))
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    _metrics_box(ax, test_metrics)
    ax.set_title(f"{name} - simulated vs. observed discharge", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(obs, sim, name, test_metrics, path):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    top = max(o.max(), s.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(o, s, s=8, alpha=0.4, color=COLORS.get(name, "red"), edgecolor="none")
    ax.plot([0, top], [0, top], ls="--", color="#5B6C74", lw=1, label="1:1 line")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Observed (m³/s)")
    ax.set_ylabel(f"{name} (m³/s)")
    ax.set_title(f"{name} - test period   NSE {test_metrics['NSE']:.3f}   "
                 f"R² {test_metrics['R2']:.3f}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison(df, sims: dict, split, metrics, path, days=730):
    """All models over (the start of) the test period + test NSE/KGE bars."""
    # show the stretch of the test period with the most observed data
    sl = split.periods()["test"]
    days = min(days, sl.stop - sl.start)
    has_obs = np.isfinite(df["Q"].to_numpy(float)).astype(int)
    counts = np.convolve(has_obs[sl], np.ones(days, dtype=int), mode="valid")
    start = sl.start + int(np.argmax(counts))
    window = slice(start, start + days)
    dates = df["Date"].iloc[window]

    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.35)
    ax = fig.add_subplot(grid[0])
    ax.plot(dates, df["Q"].iloc[window], color=OBS_COLOR, lw=1.8, label="Observed", zorder=5)
    for name, sim in sims.items():
        ax.plot(dates, sim[window], color=COLORS.get(name), lw=1.0, alpha=0.9, label=name)
    ax.set_ylabel("Discharge (m³/s)")
    peak = max(np.nanmax(df["Q"].iloc[window]),
               *(np.nanmax(s[window]) for s in sims.values()))
    ax.set_ylim(0, peak * 1.5)
    ax.grid(alpha=0.25)
    _precip_axis(ax, dates, df["P"].iloc[window].to_numpy(float))
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(sims) + 1,
              frameon=False, fontsize=9)
    ax.set_title("Test period (unseen by every model)", loc="left", fontweight="bold")

    bx = fig.add_subplot(grid[1])
    test = metrics[metrics["Period"] == "test"].set_index("Model")
    names = list(sims)
    x = np.arange(len(names))
    for k, metric in enumerate(("NSE", "KGE")):
        vals = [test.loc[m, metric] for m in names]
        bars = bx.bar(x + (k - 0.5) * 0.36, vals, width=0.34,
                      color=[COLORS.get(m) for m in names],
                      alpha=1.0 if metric == "NSE" else 0.55, label=metric)
        for b, v in zip(bars, vals):
            bx.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8)
    bx.set_xticks(x, names)
    bx.set_ylim(min(0, np.nanmin(test[["NSE", "KGE"]].values) - 0.1), 1.05)
    bx.set_ylabel("Score")
    bx.grid(axis="y", alpha=0.25)
    bx.set_title("Test-period NSE (solid) and KGE (light)", loc="left", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
