import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from src.pipeline import (AmericanPricingEngine, SVIParameters, SVIVolatilitySurface,
                          SurfaceIntegrityMonitor, black_scholes)

TRUE = SVIParameters(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.10)

@pytest.mark.parametrize("K, sigma, is_call", [
    (100.0, 0.20, True), (90.0, 0.35, True), (110.0, 0.15, True),
    (100.0, 0.25, False), (115.0, 0.30, False),
])
def test_iv_round_trip(K, sigma, is_call):
    """price -> IV -> must recover sigma (same lattice depth both ways)."""
    price_fn = AmericanPricingEngine.price_call_binomial if is_call \
        else AmericanPricingEngine.price_put_binomial
    px = price_fn(100.0, K, 1.0, 0.04, sigma, 150, 0.01)
    iv = AmericanPricingEngine.calculate_implied_volatility(
        px, 100.0, K, 1.0, 0.04, q=0.01, N=150, is_call=is_call)
    assert iv == pytest.approx(sigma, abs=1e-5)

def test_iv_rejects_arbitrage_prices():
    # Price below intrinsic must not produce a vol
    assert np.isnan(AmericanPricingEngine.calculate_implied_volatility(5.0, 120.0, 100.0, 1.0, 0.04))

def test_svi_recovers_synthetic_slice():
    k = np.linspace(-0.3, 0.3, 15)
    rng = np.random.default_rng(0)
    w = SVIVolatilitySurface.total_variance(k, TRUE) + rng.normal(0, 3e-4, k.shape)
    fit = SVIVolatilitySurface.fit_slice(k, w, grid_k=np.linspace(-0.4, 0.4, 9), T=0.5)
    rmse = np.sqrt(np.mean((SVIVolatilitySurface.total_variance(k, fit) - w) ** 2))
    assert rmse < 1e-3

def test_svi_needs_enough_points():
    with pytest.raises(ValueError):
        SVIVolatilitySurface.fit_slice(np.array([0., .1, .2, .3]), np.array([.04] * 4),
                                       grid_k=np.linspace(-.2, .2, 5))

def test_calendar_penalty_prevents_crossing():
    grid = np.linspace(-0.2, 0.2, 40)
    k = np.linspace(-0.2, 0.2, 12)
    w1_grid = SVIVolatilitySurface.total_variance(grid, TRUE)
    # Noisy "later" slice whose market data dips BELOW the earlier slice
    w2_market = SVIVolatilitySurface.total_variance(k, TRUE) * 0.97
    free = SVIVolatilitySurface.fit_slice(k, w2_market, grid)
    constrained = SVIVolatilitySurface.fit_slice(k, w2_market, grid, previous_fitted_w=w1_grid)
    gap_free = np.max(np.maximum(w1_grid - SVIVolatilitySurface.total_variance(grid, free), 0))
    gap_con = np.max(np.maximum(w1_grid - SVIVolatilitySurface.total_variance(grid, constrained), 0))
    assert gap_con < gap_free
    assert gap_con < 5e-4

def test_monitor_clean_surface_has_no_violations():
    S, r = 100.0, 0.04
    grid_k = np.linspace(-0.2, 0.2, 40)
    Ts = np.array([0.1, 0.25, 0.5, 1.0])
    T_mesh, k_mesh = np.meshgrid(Ts, grid_k)
    w = np.column_stack([T * SVIVolatilitySurface.total_variance(grid_k, TRUE) for T in Ts])
    rep = SurfaceIntegrityMonitor(S, r).evaluate(T_mesh, k_mesh, np.sqrt(w / T_mesh))
    assert rep.butterfly_violations == 0 and rep.calendar_violations == 0
    assert rep.alignment_score == 100.0

def test_monitor_detects_calendar_arbitrage_and_ignores_nan_columns():
    S, r = 100.0, 0.04
    grid_k = np.linspace(-0.2, 0.2, 40)
    Ts = np.array([0.25, 0.5, 1.0])
    T_mesh, k_mesh = np.meshgrid(Ts, grid_k)
    w = np.column_stack([T * SVIVolatilitySurface.total_variance(grid_k, TRUE) for T in Ts])
    iv = np.sqrt(w / T_mesh)
    iv[:, 1] = np.nan                      # skipped slice must be ignored, not scored
    clean = SurfaceIntegrityMonitor(S, r).evaluate(T_mesh, k_mesh, iv)
    assert clean.calendar_violations == 0
    iv[:, 2] *= 0.3                        # later maturity now has LOWER variance
    bad = SurfaceIntegrityMonitor(S, r).evaluate(T_mesh, k_mesh, iv)
    assert bad.calendar_violations > 0
    assert 0.0 <= bad.alignment_score <= 100.0

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))