"""Gaussian hidden Markov model for market regime detection: forward-backward,
Viterbi and Baum-Welch (EM), all in log space, in NumPy."""

from .data import load_returns, simulate_regimes
from .hmm import GaussianHMM

__all__ = ["GaussianHMM", "load_returns", "simulate_regimes"]
__version__ = "0.1.0"
