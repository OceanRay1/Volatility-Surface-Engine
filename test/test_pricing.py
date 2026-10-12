import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import numpy as np
import pytest
from src.pipeline import AmericanPricingEngine, black_scholes

@pytest.mark.parametrize("S, K, T, r, sigma", [
    (100.0, 100.0, 1.0, 0.04, 0.20),
    (100.0, 90.0,  0.5, 0.04, 0.25),
    (100.0, 110.0, 0.25, 0.04, 0.15),
])

def test_binomial_convergence_to_black_scholes(S, K, T, r, sigma):
    """With q = 0, early exercise of a call is never optimal -> equals European BS."""
    bs = float(black_scholes(S, K, T, r, sigma))
    tree = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=300)

    # Assert convergence using standard absolute tolerance bounds
    assert tree == pytest.approx(bs, abs=1e-2)
 
def test_binomial_edge_cases():
    """Verify boundaries, immediate expirations, and zero-volatility conditions."""
    # Expired: intrinsic
    assert AmericanPricingEngine.price_call_binomial(105.0, 100.0, 0.0, 0.04, 0.20, N=300) == pytest.approx(5.0)
    # Zero vol, r = 0: no value
    assert AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.0, 0.0, N=300) == pytest.approx(0.0)
    # Zero vol, r > 0: discounted-forward bound, NOT bare intrinsic
    expected = 100.0 - 100.0 * np.exp(-0.04 * 1.0)
    assert AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.04, 0.0, N=300) == pytest.approx(expected)
 
def test_american_put_has_early_exercise_premium():
    """Exercises the early-exercise branch, which a q=0 call never does."""
    am = AmericanPricingEngine.price_put_binomial(100.0, 110.0, 1.0, 0.05, 0.25, N=400)
    eu = float(black_scholes(100.0, 110.0, 1.0, 0.05, 0.25, is_call=False))
    assert am > eu + 0.05
    assert am >= 110.0 - 100.0  # at least intrinsic
 
def test_dividend_call_has_early_exercise_premium():
    am = AmericanPricingEngine.price_call_binomial(100.0, 80.0, 1.0, 0.02, 0.2, N=400, q=0.08)
    eu = float(black_scholes(100.0, 80.0, 1.0, 0.02, 0.2, q=0.08))
    assert am > eu + 0.05
 
def test_put_call_parity_european_limit():
    """q = 0 call lattice vs BS put via parity."""
    S, K, T, r, s = 100.0, 100.0, 1.0, 0.04, 0.2
    call = AmericanPricingEngine.price_call_binomial(S, K, T, r, s, N=400)
    put_bs = float(black_scholes(S, K, T, r, s, is_call=False))
    assert call - put_bs == pytest.approx(S - K * np.exp(-r * T), abs=1e-2)
 
def test_engine_compilation_status():
    _ = AmericanPricingEngine.price_call_binomial(100.0, 100.0, 1.0, 0.04, 0.20, N=100)
    target = AmericanPricingEngine.price_call_binomial
    assert hasattr(target, "signatures") and len(target.signatures) > 0, \
        "price_call_binomial is not a compiled Numba dispatcher"
 
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
 