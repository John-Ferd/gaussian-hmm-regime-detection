"""
Gaussian HMM with K hidden states and univariate Gaussian emissions.

    s_1 ~ pi,   s_t | s_{t-1} ~ A[s_{t-1}, :],   y_t | s_t = k ~ N(mu_k, sigma_k^2)

Everything runs in log space. Probabilities of long sequences underflow
double precision quickly (a 5,000-step sequence has likelihood around
e^{-5000}), so instead of rescaling alpha/beta at every step we keep log
quantities and combine them with log-sum-exp. Both approaches are standard;
log space is easier to verify because every quantity has a direct meaning.

* forward:   log alpha_t(k) = log b_k(y_t) + logsumexp_j( log alpha_{t-1}(j) + log A[j,k] )
* backward:  log beta_t(j)  = logsumexp_k( log A[j,k] + log b_k(y_{t+1}) + log beta_{t+1}(k) )
* posteriors gamma_t(k) and xi_t(j,k) from alpha, beta (E-step)
* Baum-Welch M-step: closed-form weighted updates of pi, A, mu, sigma
* Viterbi: max-product version of the forward pass, plus backtracking

EM only finds a local optimum, so ``fit`` runs several random restarts and
keeps the best log-likelihood. States are relabelled by increasing mean so the
output is identifiable (state 0 = lowest mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

_LOG_2PI = np.log(2 * np.pi)


def log_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(T, K) matrix of log N(y_t | mu_k, sigma_k^2)."""
    z = (y[:, None] - mu[None, :]) / sigma[None, :]
    return -0.5 * (_LOG_2PI + z**2) - np.log(sigma)[None, :]


@dataclass
class GaussianHMM:
    n_states: int = 2
    n_restarts: int = 5
    max_iter: int = 500
    tol: float = 1e-6
    min_sigma: float = 1e-4
    seed: int = 0
    # fitted
    pi: np.ndarray | None = None
    A: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    loglik_history: list = field(default_factory=list)
    n_iter_: int = 0

    # ------------------------------------------------------------------ E-step
    # logsumexp_j(a_j + log A[j, k]) is computed as c + log(exp(a - c) @ A) with
    # c = max_j a_j: the same quantity, but a matrix product instead of a
    # (K, K) log-sum-exp per time step, which is much faster in Python.
    def _forward(self, logB):
        T, K = logB.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            prev = log_alpha[t - 1]
            c = prev.max()
            log_alpha[t] = logB[t] + c + np.log(np.exp(prev - c) @ self.A)
        return log_alpha

    def _backward(self, logB):
        T, K = logB.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            nxt = logB[t + 1] + log_beta[t + 1]
            c = nxt.max()
            log_beta[t] = c + np.log(self.A @ np.exp(nxt - c))
        return log_beta

    def _e_step(self, y):
        logB = log_gaussian(y, self.mu, self.sigma)
        la, lb = self._forward(logB), self._backward(logB)
        loglik = float(logsumexp(la[-1]))
        log_gamma = la + lb - loglik
        logA = np.log(self.A)
        # log xi_t(j,k) for t = 0..T-2
        log_xi = (la[:-1, :, None] + logA[None, :, :] + (logB[1:] + lb[1:])[:, None, :]) - loglik
        return np.exp(log_gamma), np.exp(log_xi), loglik

    # ------------------------------------------------------------------ M-step
    def _m_step(self, y, gamma, xi):
        self.pi = np.clip(gamma[0], 1e-12, None)
        self.pi /= self.pi.sum()
        A = xi.sum(axis=0)
        self.A = np.clip(A / A.sum(axis=1, keepdims=True), 1e-12, None)
        self.A /= self.A.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        self.mu = (gamma * y[:, None]).sum(axis=0) / w
        var = (gamma * (y[:, None] - self.mu[None, :]) ** 2).sum(axis=0) / w
        self.sigma = np.maximum(np.sqrt(var), self.min_sigma)

    # ------------------------------------------------------------------ fit
    def _init(self, y, rng):
        K = self.n_states
        qs = np.sort(rng.choice(y, size=K, replace=False))
        self.mu = qs.astype(float)
        self.sigma = np.full(K, y.std())
        self.A = np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1))
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.full(K, 1.0 / K)

    def _em(self, y):
        hist = []
        for it in range(self.max_iter):
            gamma, xi, ll = self._e_step(y)
            hist.append(ll)
            if it > 0 and abs(hist[-1] - hist[-2]) < self.tol * abs(hist[-2]):
                break
            self._m_step(y, gamma, xi)
        return hist

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_restarts):
            self._init(y, rng)
            hist = self._em(y)
            if best is None or hist[-1] > best[0][-1]:
                best = (hist, self.pi.copy(), self.A.copy(), self.mu.copy(), self.sigma.copy())
        hist, self.pi, self.A, self.mu, self.sigma = best
        self.loglik_history, self.n_iter_ = hist, len(hist)
        self._relabel()
        return self

    def _relabel(self):
        order = np.argsort(self.mu)
        self.mu, self.sigma, self.pi = self.mu[order], self.sigma[order], self.pi[order]
        self.A = self.A[np.ix_(order, order)]

    # ------------------------------------------------------------------ inference
    def score(self, y) -> float:
        """log p(y_1..T) under the fitted parameters."""
        y = np.asarray(y, dtype=float)
        return float(logsumexp(self._forward(log_gaussian(y, self.mu, self.sigma))[-1]))

    def smoothed_proba(self, y) -> np.ndarray:
        """P(s_t = k | y_1..T): uses the whole sequence, so NOT usable for trading."""
        return self._e_step(np.asarray(y, dtype=float))[0]

    def filtered_proba(self, y) -> np.ndarray:
        """P(s_t = k | y_1..t): uses only data up to t (causal)."""
        la = self._forward(log_gaussian(np.asarray(y, dtype=float), self.mu, self.sigma))
        return np.exp(la - logsumexp(la, axis=1, keepdims=True))

    def one_step_logpdf(self, y) -> np.ndarray:
        """log p(y_t | y_1..t-1) for every t: the density forecast the model made
        the day before (t = 0 uses the initial distribution). Causal.

        log p(y_1..t) = logsumexp_k log alpha_t(k), so each day's predictive
        log density is the difference of consecutive cumulative values. The sum
        over a held-out period is its out-of-sample log-likelihood.
        """
        la = self._forward(log_gaussian(np.asarray(y, dtype=float), self.mu, self.sigma))
        cum = logsumexp(la, axis=1)
        return np.diff(np.concatenate([[0.0], cum]))

    def predict_next_proba(self, y) -> np.ndarray:
        """P(s_{t+1} = k | y_1..t): row t is the regime forecast for day t+1,
        made at the close of day t. Causal, so usable for trading."""
        return self.filtered_proba(y) @ self.A

    def viterbi(self, y) -> np.ndarray:
        """Most likely state path argmax_s p(s_1..T | y_1..T)."""
        logB = log_gaussian(np.asarray(y, dtype=float), self.mu, self.sigma)
        T, K = logB.shape
        logA = np.log(self.A)
        delta = np.log(self.pi) + logB[0]
        back = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            cand = delta[:, None] + logA
            back[t] = np.argmax(cand, axis=0)
            delta = cand[back[t], np.arange(K)] + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def expected_durations(self) -> np.ndarray:
        """Mean regime length 1 / (1 - A_kk) in time steps."""
        return 1.0 / (1.0 - np.diag(self.A))
