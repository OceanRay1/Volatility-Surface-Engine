import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from scipy.stats import norm
from numba import jit
from src.pipeline import AmericanPricingEngine

# European Black-Scholes Helper for Analytics
def analytical_black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Test Cases
@pytest.mark.parametrize("S, K, T, r, sigma", [
    (100.0, 100.0, 1.0, 0.04, 0.20),
    (100.0, 90.0,  0.5, 0.04, 0.25),
    (100.0, 110.0, 0.25, 0.04, 0.15),
])

def test_binomial_convergence_to_black_scholes(S, K, T, r, sigma):
    bs_price = analytical_black_scholes_call(S, K, T, r, sigma)
    tree_price = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=300)
    assert abs(tree_price - bs_price) < 0.05, f"Tree price {tree_price:.4f} diverged from BS price {bs_price:.4f}"

def test_binomial_edge_cases():
    assert AmericanPricingEngine.price_call_binomial(105.0, 100.0, 0.0, 0.04, 0.20) == 5.0
    assert AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.0, 0.0) == 0.0