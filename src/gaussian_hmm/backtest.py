"""
Causal backtesting of daily exposure rules, a volatility-targeting baseline,
and a block bootstrap for Sharpe-ratio differences.

Convention: ``positions[t]`` is decided at the close of day t from data up to
day t and is held during day t+1. Nothing here looks ahead.
"""
from __future__ import annotations

import numpy as np

PERIODS = 252


def performance(returns, periods: int = PERIODS) -> dict:
    """Annualized Sharpe (zero risk-free rate), annualized return and volatility,
    total log return, and maximum drawdown of daily log returns."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    wealth = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return {"sharpe": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0,
            "ann_return": float(np.expm1(r.mean() * periods)),
            "ann_vol": float(sd * np.sqrt(periods)),
            "total_log_return": float(r.sum()),
            "max_drawdown": float((wealth / peak - 1).min())}


def strategy_returns(y, positions, cost_bps: float = 2.0) -> np.ndarray:
    """Daily log returns of holding ``positions[t]`` during day t+1, minus
    ``cost_bps`` basis points per unit of turnover. Length len(y) - 1."""
    y = np.asarray(y, dtype=float)
    held = np.asarray(positions, dtype=float)[:-1]
    turnover = np.abs(np.diff(np.concatenate([[held[0]], held])))
    return held * y[1:] - cost_bps * 1e-4 * turnover


def backtest(y, positions, cost_bps: float = 2.0) -> dict:
    held = np.asarray(positions, dtype=float)[:-1]
    res = performance(strategy_returns(y, positions, cost_bps))
    res["avg_exposure"] = float(held.mean())
    res["turnover_per_year"] = float(np.abs(np.diff(held)).sum() / len(held) * PERIODS)
    return res


def vol_target_positions(y, target_ann_vol: float, window: int = 20, cap: float = 1.0) -> np.ndarray:
    """Classic volatility targeting: exposure = target / trailing realized vol,
    capped at ``cap``. positions[t] uses returns up to and including day t.
    The standard simple benchmark for any regime rule that de-risks in turbulence."""
    y = np.asarray(y, dtype=float)
    pos = np.full(len(y), cap)
    sq = np.concatenate([[0.0], np.cumsum(y ** 2)])
    for t in range(window - 1, len(y)):
        rv = np.sqrt((sq[t + 1] - sq[t + 1 - window]) / window * PERIODS)
        pos[t] = min(cap, target_ann_vol / rv) if rv > 0 else cap
    return pos


def block_bootstrap_sharpe_diff(a, b, n_boot: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Stationary-block bootstrap of Sharpe(a) - Sharpe(b) for two aligned daily
    return series. Resampling whole blocks keeps volatility clustering, which an
    i.i.d. bootstrap would destroy (and so understate the uncertainty)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)

    def sharpe(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(PERIODS) if sd > 0 else 0.0

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        t = 0
        while t < n:
            start, length = rng.integers(n), rng.geometric(1 / block)
            seg = (start + np.arange(length)) % n
            take = min(length, n - t)
            idx[t:t + take] = seg[:take]
            t += take
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(sharpe(a) - sharpe(b)), "ci95": [float(lo), float(hi)]}
