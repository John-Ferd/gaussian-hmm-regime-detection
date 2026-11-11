"""Gaussian HMM regimes on real SPY daily returns, 2000 to October 2026.

Three periods, used strictly in order:
    train       2000-2012   fit the models
    validation  2013-2018   choose the number of regimes K and the trading rule
    test        2019-2026   report once, with nothing chosen on it

Part 1, density forecasting. For K = 1..4, fit on train and score the
one-step-ahead predictive log-likelihood of the validation days (the filter runs
causally through all earlier days). Pick K on validation, refit on
train+validation, and score the test days against two baselines: a single
Gaussian (K = 1) and RiskMetrics EWMA volatility (lambda = 0.94).

Part 2, trading. Walk forward one calendar year at a time: refit the HMM on all
data before the year, then trade the year using only the regime forecast made
at each previous close. Rules: "hard" (flat when the turbulent regime is more
likely than not) and "soft" (exposure = 1 - P(turbulent)). Baselines:
buy-and-hold and 20-day volatility targeting. 2 bp cost per unit of turnover.

Writes artifacts/results_SPY.json and plots.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaussian_hmm import GaussianHMM, load_returns
from gaussian_hmm.backtest import (
    backtest,
    block_bootstrap_sharpe_diff,
    strategy_returns,
    vol_target_positions,
)
from gaussian_hmm.env import environment


def _fit(task):
    y_part, K, restarts, seed = task
    return GaussianHMM(K, n_restarts=restarts, seed=seed).fit(y_part)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--csv", default=None, help="daily price CSV (Date + Adj Close/Close); default: data/<ticker>.csv")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    OUT = ROOT / "artifacts"
    OUT.mkdir(exist_ok=True)

    y, dates = load_returns(args.ticker, csv=args.csv, cache_dir=ROOT / "data")
    year = dates.year.to_numpy()
    TRAIN_END, VAL_END = 2012, 2018
    tr, va, te = year <= TRAIN_END, (year > TRAIN_END) & (year <= VAL_END), year > VAL_END
    KS = (1, 2, 3, 4)


    def fit_many(ex, specs):
        """Fit one HMM per (K, number of leading days) in parallel worker processes."""
        return list(ex.map(_fit, [(y[:n], K, args.restarts, args.seed) for K, n in specs]))


    def ewma_logpdf(y, lam=0.94):
        """RiskMetrics: zero-mean Gaussian with EWMA variance from data up to t-1."""
        var = np.empty(len(y))
        var[0] = y[:250].var()
        for t in range(1, len(y)):
            var[t] = lam * var[t - 1] + (1 - lam) * y[t - 1] ** 2
        return -0.5 * (np.log(2 * np.pi * var) + y ** 2 / var)


    def bic(m, n):
        K = m.n_states
        return -2 * m.score(y[:n]) + ((K - 1) + K * (K - 1) + 2 * K) * np.log(n)


    # ---------------------------------------------------------------- part 1: density forecasts
    n_tr, n_trva = int(tr.sum()), int((tr | va).sum())
    with ProcessPoolExecutor(args.workers) as ex:
        fits_tr = dict(zip(KS, fit_many(ex, [(K, n_tr) for K in KS])))
        fits_trva = dict(zip(KS, fit_many(ex, [(K, n_trva) for K in KS])))

    val_ll = {K: float(m.one_step_logpdf(y[:n_trva])[va[:n_trva]].sum()) for K, m in fits_tr.items()}
    bic_tr = {K: float(bic(m, n_tr)) for K, m in fits_tr.items()}
    K_best = max(val_ll, key=val_ll.get)
    test_ll = {f"hmm_K{K}": float(m.one_step_logpdf(y)[te].sum()) for K, m in fits_trva.items()}
    test_ll["ewma_riskmetrics"] = float(ewma_logpdf(y)[te].sum())
    n_te = int(te.sum())
    m = fits_trva[K_best]
    order = np.argsort(m.sigma)          # calm -> turbulent, for reporting
    regimes = [{"mean_daily": round(float(m.mu[k]), 5), "ann_vol": round(float(m.sigma[k] * np.sqrt(252)), 3),
                "expected_days": round(float(m.expected_durations()[k]), 1),
                "share_of_days_viterbi": round(float((m.viterbi(y[:n_trva]) == k).mean()), 3)} for k in order]

    # ---------------------------------------------------------------- part 2: walk-forward trading
    K_trade = max(K_best, 2)               # a 1-state model has no turbulent regime to step aside from
    years = list(range(TRAIN_END + 1, int(year.max()) + 1))
    first_idx = {Y: int(np.argmax(year == Y)) for Y in years}
    with ProcessPoolExecutor(args.workers) as ex:
        # model for year Y: fit on every day up to the last close before Y starts
        yearly = dict(zip(years, fit_many(ex, [(K_trade, first_idx[Y]) for Y in years])))

    # p_turb[t] = P(turbulent regime on day t+1 | data up to day t). The year-Y model makes
    # every forecast from the last close before Y through the second-to-last day of Y.
    p_turb = np.full(len(y), np.nan)
    for Y, mod in yearly.items():
        days = np.flatnonzero(year == Y)
        lo, hi_day = days[0] - 1, days[-1]
        turb = int(np.argmax(mod.sigma))
        p_turb[lo:hi_day + 1] = mod.predict_next_proba(y[:hi_day + 1])[lo:hi_day + 1, turb]

    target = float(y[tr].std() * np.sqrt(252))
    pos = {"buy_and_hold": np.ones(len(y)),
           "vol_target_20d": vol_target_positions(y, target),
           "hmm_hard": (p_turb < 0.5).astype(float),
           "hmm_soft": 1 - p_turb}


    def period_slice(mask):
        """Positions from the close before the first day of the period to the last day."""
        idx = np.flatnonzero(mask)
        return slice(idx[0] - 1, idx[-1] + 1)


    def evaluate(mask):
        s = period_slice(mask)
        return {k: {kk: round(v, 4) for kk, v in backtest(y[s], p[s], args.cost_bps).items()} for k, p in pos.items()}


    bt_val, bt_test = evaluate(va), evaluate(te)
    rule = max(("hmm_hard", "hmm_soft"), key=lambda k: bt_val[k]["sharpe"])
    s = period_slice(te)
    r_rule = strategy_returns(y[s], pos[rule][s], args.cost_bps)
    boot = {base: block_bootstrap_sharpe_diff(r_rule, strategy_returns(y[s], pos[base][s], args.cost_bps),
                                              seed=args.seed)
            for base in ("buy_and_hold", "vol_target_20d")}

    ll_best = test_ll[f"hmm_K{K_best}"]
    res = {
        "data": f"{args.ticker} daily log returns of adjusted closes, {dates[0].date()} to {dates[-1].date()}",
        "n_days": {"train": n_tr, "validation": int(va.sum()), "test": n_te},
        "periods": {"train": f"2000-{TRAIN_END}", "validation": f"{TRAIN_END + 1}-{VAL_END}",
                    "test": f"{VAL_END + 1}-"},
        "density": {
            "validation_loglik_by_K": {K: round(v, 1) for K, v in val_ll.items()},
            "train_bic_by_K": {K: round(v, 1) for K, v in bic_tr.items()},
            "chosen_K_on_validation": K_best,
            "test_loglik": {k: round(v, 1) for k, v in test_ll.items()},
            "test_loglik_per_day_gain_vs_single_gaussian": round((ll_best - test_ll["hmm_K1"]) / n_te, 4),
            "test_loglik_per_day_gain_vs_ewma": round((ll_best - test_ll["ewma_riskmetrics"]) / n_te, 4),
        },
        "regimes_fit_on_train_plus_validation_calm_to_turbulent": regimes,
        "em_iterations_final_fit": m.n_iter_,
        "trading": {"cost_bps_per_unit_turnover": args.cost_bps, "vol_target_ann": round(target, 4),
                    "rule_chosen_on_validation": rule, "validation": bt_val, "test": bt_test,
                    "test_sharpe_diff_block_bootstrap": {k: {"diff": round(v["diff"], 3),
                                                             "ci95": [round(c, 3) for c in v["ci95"]]}
                                                         for k, v in boot.items()}},
    }
    res["environment"] = environment()
    (OUT / f"results_{args.ticker}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ---------------------------------------------------------------- plots
    px = np.exp(np.cumsum(y))
    mfull = fits_trva[K_best]
    turb = int(np.argmax(mfull.sigma))
    filt = mfull.filtered_proba(y)[:, turb]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].semilogy(dates, px, color="k", lw=0.8)
    axes[0].fill_between(dates, px.min(), px.max(), where=filt > 0.5, color="tab:red", alpha=0.18, lw=0,
                         label="P(turbulent) > 0.5 (filtered)")
    axes[0].set_ylabel(f"{args.ticker} growth of $1 (log scale)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"K = {K_best} Gaussian HMM fit on 2000-{VAL_END}; filtered shading, "
                      f"in-sample before {VAL_END + 1}")
    axes[1].plot(dates, filt, lw=0.6, color="tab:red")
    for b in (TRAIN_END, VAL_END):
        for ax in axes:
            ax.axvline(pd.Timestamp(f"{b + 1}-01-01"), color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("P(turbulent regime)")
    fig.tight_layout()
    fig.savefig(OUT / f"regimes_{args.ticker}.png", dpi=130)
    plt.close(fig)

    s = period_slice(te)
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in pos:
        ax.plot(dates[s][1:], np.exp(np.cumsum(strategy_returns(y[s], pos[k][s], args.cost_bps))), lw=1,
                label=f"{k} (Sharpe {bt_test[k]['sharpe']:.2f})")
    ax.set_ylabel("growth of $1")
    ax.set_title(f"Test period {VAL_END + 1}-{dates[-1].year}, walk-forward yearly refits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"backtest_{args.ticker}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(mfull.loglik_history, marker="o", ms=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("Baum-Welch never decreases the likelihood")
    fig.tight_layout()
    fig.savefig(OUT / f"loglik_{args.ticker}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
