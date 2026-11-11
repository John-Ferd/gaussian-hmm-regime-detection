import itertools

import numpy as np
import pytest

from gaussian_hmm import GaussianHMM, simulate_regimes
from gaussian_hmm.hmm import log_gaussian


def _toy():
    m = GaussianHMM(2)
    m.pi = np.array([0.6, 0.4])
    m.A = np.array([[0.8, 0.2], [0.3, 0.7]])
    m.mu = np.array([-1.0, 1.0])
    m.sigma = np.array([1.0, 0.5])
    y = np.array([0.3, -1.2, 0.9, 1.1, -0.4])
    return m, y


def _brute_force(m, y):
    """Enumerate all K^T paths: joint log p(y, s) for each path."""
    logB = log_gaussian(y, m.mu, m.sigma)
    out = {}
    for path in itertools.product(range(2), repeat=len(y)):
        lp = np.log(m.pi[path[0]]) + logB[0, path[0]]
        for t in range(1, len(y)):
            lp += np.log(m.A[path[t - 1], path[t]]) + logB[t, path[t]]
        out[path] = lp
    return out


def test_forward_likelihood_matches_brute_force():
    m, y = _toy()
    joint = _brute_force(m, y)
    assert np.isclose(m.score(y), np.logaddexp.reduce(list(joint.values())))


def test_viterbi_matches_brute_force_argmax():
    m, y = _toy()
    joint = _brute_force(m, y)
    assert tuple(m.viterbi(y)) == max(joint, key=joint.get)


def test_smoothed_posteriors_match_brute_force():
    m, y = _toy()
    joint = _brute_force(m, y)
    lz = np.logaddexp.reduce(list(joint.values()))
    expected = np.zeros((len(y), 2))
    for path, lp in joint.items():
        expected[np.arange(len(y)), path] += np.exp(lp - lz)
    np.testing.assert_allclose(m.smoothed_proba(y), expected, atol=1e-12)


def test_likelihood_matches_hmmlearn_on_long_sequence():
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    y, _ = simulate_regimes(T=3000, seed=1)
    m = GaussianHMM(2, n_restarts=2).fit(y)
    ref = hmmlearn.GaussianHMM(2, covariance_type="diag")
    ref.startprob_, ref.transmat_ = m.pi, m.A
    ref.means_, ref.covars_ = m.mu[:, None], (m.sigma**2)[:, None]
    assert np.isclose(m.score(y), ref.score(y[:, None]), rtol=1e-10)


def test_em_never_decreases_likelihood():
    y, _ = simulate_regimes(T=1500, seed=2)
    m = GaussianHMM(2, n_restarts=3, seed=2).fit(y)
    assert np.all(np.diff(m.loglik_history) >= -1e-8)


def test_fit_recovers_regimes_and_parameters():
    y, s = simulate_regimes(T=2500, seed=0)
    m = GaussianHMM(2, n_restarts=3).fit(y)
    np.testing.assert_allclose(m.sigma, [0.020, 0.008], rtol=0.1)
    np.testing.assert_allclose(np.diag(m.A), [0.97, 0.99], atol=0.02)
    assert (m.viterbi(y) == s).mean() > 0.95


def test_filtered_probabilities_are_causal():
    m, y = _toy()
    full = m.filtered_proba(np.append(y, [5.0, -5.0]))
    np.testing.assert_allclose(full[: len(y)], m.filtered_proba(y))
    np.testing.assert_allclose(full.sum(axis=1), 1.0)


def test_states_are_ordered_by_mean():
    y, _ = simulate_regimes(T=1500, seed=3)
    m = GaussianHMM(2, n_restarts=2, seed=5).fit(y)
    assert m.mu[0] < m.mu[1]
    np.testing.assert_allclose(m.A.sum(axis=1), 1.0)


def test_one_step_predictive_logpdf_sums_to_total_loglik():
    y, _ = simulate_regimes(T=400, seed=6)
    m = GaussianHMM(2, n_restarts=2, seed=0).fit(y)
    np.testing.assert_allclose(m.one_step_logpdf(y).sum(), m.score(y), rtol=1e-10)


def test_regime_forecast_is_causal_and_a_distribution():
    y, _ = simulate_regimes(T=300, seed=7)
    m = GaussianHMM(2, n_restarts=2, seed=0).fit(y)
    p_long, p_short = m.predict_next_proba(y), m.predict_next_proba(y[:200])
    np.testing.assert_allclose(p_long[:200], p_short)
    np.testing.assert_allclose(p_long.sum(axis=1), 1.0)
