import numpy as np

from gaussian_hmm.backtest import backtest, block_bootstrap_sharpe_diff, performance, vol_target_positions


def test_backtest_holds_yesterdays_position():
    y = np.array([0.0, 0.01, -0.02, 0.03])
    assert np.isclose(backtest(y, np.ones(4), cost_bps=0)["total_log_return"], y[1:].sum())
    # in the market only on the day after the signal: position [0, 1, 0, 0] earns y[2]
    assert np.isclose(backtest(y, np.array([0.0, 1.0, 0.0, 0.0]), cost_bps=0)["total_log_return"], y[2])


def test_costs_are_charged_per_unit_of_turnover():
    y = np.zeros(5)
    res = backtest(y, np.array([1.0, 0.0, 1.0, 1.0, 1.0]), cost_bps=10)
    assert np.isclose(res["total_log_return"], -2 * 10e-4)        # two switches of size 1


def test_max_drawdown():
    r = np.log([1.1, 0.5, 1.2])          # wealth 1.1 -> 0.55 -> 0.66
    assert np.isclose(performance(r)["max_drawdown"], 0.55 / 1.1 - 1)


def test_vol_target_uses_only_past_returns_and_is_capped():
    y = np.random.default_rng(0).normal(0, 0.01, 300)
    full, short = vol_target_positions(y, 0.10), vol_target_positions(y[:200], 0.10)
    np.testing.assert_allclose(full[:200], short)
    assert full.max() <= 1.0


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.0005, 0.01, 1000), rng.normal(0.0, 0.01, 1000)
    res = block_bootstrap_sharpe_diff(a, b, n_boot=300)
    assert res["ci95"][0] <= res["diff"] <= res["ci95"][1]
