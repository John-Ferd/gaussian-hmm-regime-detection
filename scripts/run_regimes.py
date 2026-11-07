"""Fit a Gaussian HMM to regime-switching returns and report what it recovers.

Default: synthetic data with known regimes, so recovery can be checked.
With --ticker (and --csv FILE, or `pip install -e ".[real-data]"` plus
internet to download), fits real daily log returns instead; there is no ground truth then, only the fitted model.
Writes artifacts/results.json (or results_<ticker>.json) and plots.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussian_hmm import GaussianHMM, load_returns, simulate_regimes

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--ticker", default=None)
ap.add_argument("--csv", default=None, help="daily price CSV (Date + Adj Close/Close); use with --ticker for the name")
ap.add_argument("--start", default="2005-01-01")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = Path(__file__).resolve().parents[1] / "artifacts"
out.mkdir(exist_ok=True)
tag = f"_{args.ticker}" if args.ticker else ""

if args.ticker:
    y, dates = load_returns(args.ticker, args.start, csv=args.csv)
    truth = None
else:
    y, truth = simulate_regimes(T=2500, seed=args.seed)
    dates = np.arange(len(y))


def bic(m, y):
    K = m.n_states
    n_params = (K - 1) + K * (K - 1) + 2 * K
    return -2 * m.score(y) + n_params * np.log(len(y))


models = {K: GaussianHMM(K, seed=args.seed).fit(y) for K in (1, 2, 3)}
bics = {K: round(float(bic(m, y)), 1) for K, m in models.items()}
m = models[2]
ll = np.array(m.loglik_history)

res = {
    "data": args.ticker or "synthetic (T=2500, seed=%d)" % args.seed,
    "n_obs": len(y),
    "bic_by_K": bics,
    "chosen_K_by_bic": int(min(bics, key=bics.get)),
    "K2_fit": {
        "mu": m.mu.round(5).tolist(),
        "sigma": m.sigma.round(5).tolist(),
        "A": m.A.round(4).tolist(),
        "expected_duration_days": m.expected_durations().round(1).tolist(),
        "em_iterations": m.n_iter_,
        "final_loglik": round(float(ll[-1]), 3),
        "loglik_monotone": bool(np.all(np.diff(ll) >= -1e-8)),
    },
}
if truth is not None:
    res["truth"] = {"mu": [-0.0015, 0.0008], "sigma": [0.020, 0.008], "A": [[0.97, 0.03], [0.01, 0.99]]}
    res["state_accuracy"] = {
        "viterbi": round(float((m.viterbi(y) == truth).mean()), 4),
        "smoothed_argmax": round(float((m.smoothed_proba(y).argmax(1) == truth).mean()), 4),
        "filtered_argmax_causal": round(float((m.filtered_proba(y).argmax(1) == truth).mean()), 4),
    }
(out / f"results{tag}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

# Plot: cumulative return shaded by decoded regime, plus P(high-vol regime).
path = m.viterbi(y)
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
cum = np.cumsum(y)
axes[0].plot(dates, cum, color="k", lw=0.8)
for t in np.flatnonzero(path == 0):
    axes[0].axvspan(dates[t], dates[min(t + 1, len(y) - 1)], color="tab:red", alpha=0.15, lw=0)
axes[0].set_ylabel("cumulative log return")
axes[0].set_title("Viterbi-decoded low-mean regime shaded red")
axes[1].plot(dates, m.smoothed_proba(y)[:, 0], lw=0.8, label="smoothed P(low-mean regime)")
if truth is not None:
    axes[1].plot(dates, (truth == 0).astype(float), lw=0.6, color="gray", alpha=0.6, label="true regime")
axes[1].set_ylabel("probability")
axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(out / f"regimes{tag}.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(ll, marker="o", ms=3)
ax.set_xlabel("EM iteration")
ax.set_ylabel("log-likelihood")
ax.set_title("Baum-Welch never decreases the likelihood")
fig.tight_layout()
fig.savefig(out / f"loglik{tag}.png", dpi=130)
plt.close(fig)
