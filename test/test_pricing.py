import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from scipy.stats import norm
from src.pipeline import AmericanPricingEngine

def analytical_black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical European Black-Scholes baseline calculator."""
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

@pytest.mark.parametrize("S, K, T, r, sigma", [
    (100.0, 100.0, 1.0, 0.04, 0.20),
    (100.0, 90.0,  0.5, 0.04, 0.25),
    (100.0, 110.0, 0.25, 0.04, 0.15),
])

def test_binomial_convergence_to_black_scholes(S, K, T, r, sigma):
    """
    Verify binomial tree convergence against analytical Black-Scholes model.
    NOTE: For non-dividend paying assets, early exercise of an American Call 
    is sub-optimal, allowing direct convergence verification against European options.
    """
    bs_price = analytical_black_scholes_call(S, K, T, r, sigma)
    tree_price = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=300)
    
    # Assert convergence using standard absolute tolerance bounds
    assert tree_price == pytest.approx(bs_price, abs=1e-2), \
        f"Tree execution diverged from analytical baseline. Tree: {tree_price:.4f}, BS: {bs_price:.4f}"

def test_binomial_edge_cases():
    """Verify boundaries, immediate expirations, and zero-volatility conditions."""
    # Immediate expiry (T=0) - pass N=300 explicitly
    assert AmericanPricingEngine.price_call_binomial(105.0, 100.0, 0.0, 0.04, 0.20, N=300) == pytest.approx(5.0)

    # Zero volatility baseline - pass N=300 explicitly
    assert AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.0, 0.0, N=300) == pytest.approx(0.0)

def test_engine_compilation_status():
    """Ensure that the engine method runs as a compiled Numba dispatcher."""
    # Trigger JIT compilation first so dispatcher metadata is populated
    _ = AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.04, 0.20, N=100)
    
    # Retrieve the inner static method function target 
    target_func = getattr(AmericanPricingEngine.price_call_binomial, "__func__", AmericanPricingEngine.price_call_binomial)
    
    # Assert the method contains Numba compilation dispatcher signatures or py_func
    assert hasattr(target_func, "signatures") or hasattr(target_func, "py_func"), \
        "Performance optimization failure: Target engine function is missing JIT compiler properties."

if __name__ == "__main__":
    print("Running American Pricing Engine test suite...")
    # Programmatically invokes pytest on the current script file
    sys.exit(pytest.main([__file__, "-v"]))