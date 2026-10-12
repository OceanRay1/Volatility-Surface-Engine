"""
Volatility Surface Engine
-------------------------
CRR binomial pricing (Numba JIT), implied-vol inversion, per-expiry SVI
calibration with soft calendar / wing penalties, and a closed-form
arbitrage monitor.

Conventions
-----------
* T is in YEARS everywhere (the old `days_to_expiry` field stored years).
* log-moneyness k = ln(K / F) with forward F = S * exp((r - q) * T).
* SVI total variance w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize, root_scalar
from scipy.stats import norm

# 1. Domain & model config
@dataclass(frozen=True)
class MarketNode:
    """Validated options market datum."""
    years_to_expiry: float
    strike: float
    log_moneyness: float      # ln(K / F)
    market_price: float
    implied_vol: float
    total_variance: float

    def __post_init__(self):
        if self.years_to_expiry <= 0:
            raise ValueError("years_to_expiry must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.implied_vol <= 0 or not np.isfinite(self.implied_vol):
            raise ValueError("implied_vol must be positive and finite")

@dataclass(frozen=True)
class SVIParameters:
    """Gatheral SVI raw parameters."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self):
        if self.b < 0:
            raise ValueError("SVI b must be >= 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("SVI rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("SVI sigma must be > 0")

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

@dataclass(frozen=True)
class IntegrityReport:
    """Immutable validation summary."""
    butterfly_violations: int
    calendar_violations: int
    butterfly_checks: int
    calendar_checks: int
    alignment_score: float    # % of checks passed, always in [0, 100]
    total_nodes: int

# 2. Pricing layer
@jit(nopython=True, cache=True, fastmath={"contract"})
def _crr_american(S, K, T, r, q, sigma, N, is_call):
    """
    Cox-Ross-Rubinstein American option on an underlying with continuous
    dividend yield q.  Uses a single 1-D value buffer plus a precomputed
    table of u^k so the O(N^2) inner loop contains no pow() calls.
    """
    sign = 1.0 if is_call else -1.0
    intrinsic = max(0.0, sign * (S - K))
    if T <= 0.0:
        return intrinsic
    if sigma <= 0.0:
        # Deterministic limit: best of exercising now vs. holding to expiry
        # (discounted forward payoff), never the bare intrinsic value.
        fwd = max(0.0, sign * (S * np.exp(-q * T) - K * np.exp(-r * T)))
        return max(intrinsic, fwd)

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if p >= 1.0 or p <= 0.0:
        return np.nan  # lattice not risk-neutral-consistent: refuse to guess
    df = np.exp(-r * dt)

    # pw[N + k] = u**k for k in [-N, N]
    pw = np.empty(2 * N + 1)
    pw[N] = 1.0
    for k in range(1, N + 1):
        pw[N + k] = pw[N + k - 1] * u
        pw[N - k] = pw[N - k + 1] * d

    V = np.empty(N + 1)
    for j in range(N + 1):
        V[j] = max(0.0, sign * (S * pw[2 * N - 2 * j] - K))  # u^(N-2j)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = df * (p * V[j] + (1.0 - p) * V[j + 1])
            exercise = sign * (S * pw[N + i - 2 * j] - K)      # u^(i-2j)
            V[j] = max(continuation, exercise)

    return V[0]

def black_scholes(S, K, T, r, sigma, q: float = 0.0, is_call: bool = True):
    """Vectorised European Black-Scholes with continuous dividend yield."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    sign = 1.0 if is_call else -1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
        d2 = d1 - sq
        px = sign * (S * np.exp(-q * T) * norm.cdf(sign * d1)
                     - K * np.exp(-r * T) * norm.cdf(sign * d2))
    # Degenerate cases: discounted-forward bound, not bare intrinsic
    degenerate = (T <= 0) | (sigma <= 0)
    fwd = np.maximum(0.0, sign * (S * np.exp(-q * T) - K * np.exp(-r * T)))
    return np.where(degenerate, fwd, px)

class AmericanPricingEngine:
    """Binomial pricing and implied-vol inversion."""

    @staticmethod
    @jit(nopython=True, cache=True, fastmath={"contract"})
    def price_call_binomial(S: float, K: float, T: float, r: float,
                            sigma: float, N: int = 50, q: float = 0.0) -> float:
        """American call (CRR lattice). With q == 0 this equals the European call."""
        return _crr_american(S, K, T, r, q, sigma, N, True)

    @staticmethod
    @jit(nopython=True, cache=True, fastmath={"contract"})
    def price_put_binomial(S: float, K: float, T: float, r: float,
                           sigma: float, N: int = 50, q: float = 0.0) -> float:
        """American put (CRR lattice). Early exercise is genuinely active here."""
        return _crr_american(S, K, T, r, q, sigma, N, False)

    @classmethod
    def calculate_implied_volatility(cls, C_market: float, S: float, K: float,
                                     T: float, r: float, q: float = 0.0,
                                     N: int = 200, is_call: bool = True) -> float:
        """Brent inversion of the binomial model (same N used for pricing)."""
        if T <= 0.0 or not np.isfinite(C_market):
            return np.nan
        sign = 1.0 if is_call else -1.0
        lower_bound = max(0.0, sign * (S - K),
                          sign * (S * np.exp(-q * T) - K * np.exp(-r * T)))
        if C_market <= lower_bound + 1e-10:
            return np.nan

        price = cls.price_call_binomial if is_call else cls.price_put_binomial

        def objective(sig: float) -> float:
            return price(S, K, T, r, sig, N, q) - C_market

        lo, hi = 0.01, 3.0
        f_lo, f_hi = objective(lo), objective(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return np.nan
        try:
            res = root_scalar(objective, bracket=[lo, hi], method="brentq", xtol=1e-8)
            return float(res.root) if res.converged else np.nan
        except ValueError:
            return np.nan

# 3. SVI calibration
class SVIVolatilitySurface:
    """SVI slice calibration."""

    MIN_POINTS = 6  # 5 parameters: need strictly more points than parameters

    @staticmethod
    def total_variance(k, params: SVIParameters) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return params.a + params.b * (
            params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma ** 2)
        )

    @classmethod
    def fit_slice(cls, k_market: np.ndarray, w_market: np.ndarray, grid_k: np.ndarray,
                  previous_fitted_w: Optional[np.ndarray] = None,
                  T: Optional[float] = None) -> SVIParameters:
        """
        Calibrate one SVI slice.

        Hard constraints (box bounds) already guarantee b > 0, |rho| < 1,
        sigma > 0 and therefore w >= a > 0, so no penalties for those.
        Soft penalties:
          * calendar: new slice must not fall below `previous_fitted_w` on
            `grid_k`.  Both are raw SVI values on the same grid, so the two
            objects being compared are consistent.
          * Lee wing bound: b * (1 + |rho|) <= 2 (right/left wing slope of
            total variance <= 2), which rules out wing butterfly arbitrage.
        Objective is normalised by mean(w)^2 so the penalty weights do not
        depend on the absolute variance level.
        """
        k_market = np.asarray(k_market, dtype=float)
        w_market = np.asarray(w_market, dtype=float)
        if len(k_market) < cls.MIN_POINTS:
            raise ValueError(f"Need at least {cls.MIN_POINTS} points to fit 5 SVI parameters")

        scale = max(float(np.mean(w_market)) ** 2, 1e-12)

        def objective(p_arr: np.ndarray) -> float:
            params = SVIParameters(*p_arr)
            w_pred = cls.total_variance(k_market, params)
            loss = np.mean((w_market - w_pred) ** 2) / scale

            slope = params.b * (1.0 + abs(params.rho))
            if slope > 2.0:
                loss += 100.0 * (slope - 2.0) ** 2

            if previous_fitted_w is not None:
                w_grid = cls.total_variance(grid_k, params)
                shortfall = np.maximum(previous_fitted_w - w_grid, 0.0)
                loss += 1.0e3 * np.sum(shortfall ** 2) / scale
            return float(loss)

        bounds = [(1e-5, 2.0), (1e-5, 2.0), (-0.95, 0.95), (-1.0, 1.0), (1e-4, 1.0)]
        initial_guess = [max(0.5 * float(np.min(w_market)), 1e-4), 0.1, -0.3, 0.0, 0.1]
        res = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000})
        if not res.success:
            warnings.warn(f"SVI slice fit did not fully converge: {res.message}")
        return SVIParameters(*res.x)

# 4. Ingestion
class OptionsDataPipeline:
    """Raw vendor ingestion and data-hygiene filters."""

    def __init__(self, ticker_symbol: str, risk_free_rate: float,
                 dividend_yield: float = 0.0, max_expiries: int = 7,
                 iv_tree_depth: int = 200):
        import yfinance as yf  # keep network deps out of pricing/tests

        self.ticker_symbol = ticker_symbol
        self.r = risk_free_rate
        self.q = dividend_yield
        self.max_expiries = max_expiries
        self.iv_tree_depth = iv_tree_depth
        self._ticker = yf.Ticker(ticker_symbol)

    def fetch_market_state(self) -> Tuple[float, List[MarketNode]]:
        history = self._ticker.history(period="1d")
        if history.empty:
            raise ValueError(f"Failed to fetch market spot for symbol: {self.ticker_symbol}")

        S_spot = float(history["Close"].iloc[-1])
        today = datetime.now()
        nodes: List[MarketNode] = []

        for exp_str in self._ticker.options[: self.max_expiries]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T_years = (exp_date - today).total_seconds() / (365.25 * 86400.0)
            if T_years < 6 / 365.25:
                continue

            try:
                calls = self._ticker.option_chain(exp_str).calls
            except Exception:
                continue

            calls = calls[(calls["strike"] > S_spot * 0.80) & (calls["strike"] < S_spot * 1.20)]
            F = S_spot * np.exp((self.r - self.q) * T_years)

            for _, c in calls.iterrows():
                bid, ask, K = float(c["bid"]), float(c["ask"]), float(c["strike"])
                if bid <= 0.02 or ask <= 0.02 or (ask - bid) / ask > 0.40:
                    continue

                C_mid = 0.5 * (bid + ask)
                iv = AmericanPricingEngine.calculate_implied_volatility(
                    C_mid, S_spot, K, T_years, self.r, self.q, N=self.iv_tree_depth)

                if np.isfinite(iv) and 0.05 < iv < 2.0:
                    nodes.append(MarketNode(
                        years_to_expiry=T_years,
                        strike=K,
                        log_moneyness=float(np.log(K / F)),
                        market_price=C_mid,
                        implied_vol=iv,
                        total_variance=float(iv ** 2 * T_years),
                    ))
        return S_spot, nodes

# 5. Arbitrage monitor (closed form, no lattice noise)
class SurfaceIntegrityMonitor:
    """
    Static-arbitrage checks on a fitted surface.

    Prices are European Black-Scholes (closed form): a lattice with N ~ 50
    has discretisation error far larger than the second-difference
    thresholds, so it would report lattice noise as "arbitrage".
    """

    BUTTERFLY_TOL = 1e-8
    CALENDAR_TOL = 1e-8

    def __init__(self, S_spot: float, r: float, q: float = 0.0):
        self.S_spot = S_spot
        self.r = r
        self.q = q

    def evaluate(self, T_mesh: np.ndarray, k_mesh: np.ndarray,
                 iv_surface: np.ndarray) -> IntegrityReport:
        """Meshes have shape (n_k, n_T); columns with NaN IV (unfitted) are dropped."""
        valid = np.all(np.isfinite(iv_surface), axis=0)
        T_mesh, k_mesh, iv = T_mesh[:, valid], k_mesh[:, valid], iv_surface[:, valid]
        n_k, n_T = iv.shape
        if n_T == 0 or n_k < 3:
            return IntegrityReport(0, 0, 0, 0, 100.0, 0)

        F = self.S_spot * np.exp((self.r - self.q) * T_mesh)
        K = F * np.exp(k_mesh)
        C = black_scholes(self.S_spot, K, T_mesh, self.r, iv, self.q)

        # Butterfly: call price convex in strike (non-uniform spacing)
        h1 = K[1:-1] - K[:-2]
        h2 = K[2:] - K[1:-1]
        d2C = 2.0 * ((C[2:] - C[1:-1]) / h2 - (C[1:-1] - C[:-2]) / h1) / (h1 + h2)
        butterfly = int(np.sum(d2C < -self.BUTTERFLY_TOL))
        n_butterfly = d2C.size

        # Calendar: total variance non-decreasing in T at fixed forward moneyness
        w = iv ** 2 * T_mesh
        dw = np.diff(w, axis=1)
        calendar = int(np.sum(dw < -self.CALENDAR_TOL))
        n_calendar = dw.size

        checks = n_butterfly + n_calendar
        score = 100.0 * (1.0 - (butterfly + calendar) / checks) if checks else 100.0
        return IntegrityReport(butterfly, calendar, n_butterfly, n_calendar,
                               score, int(iv.size))

# 6. Orchestration
class VolatilitySurfaceOrchestrator:
    """Fetch -> calibrate -> verify."""

    def __init__(self, ticker_symbol: str, risk_free_rate: float = 0.042,
                 dividend_yield: float = 0.0):
        self.pipeline = OptionsDataPipeline(ticker_symbol, risk_free_rate, dividend_yield)
        self.r = risk_free_rate
        self.q = dividend_yield

    def execute_lifecycle(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        S_spot, market_nodes = self.pipeline.fetch_market_state()
        if not market_nodes:
            raise ValueError(f"[FATAL] Zero valid nodes for {self.pipeline.ticker_symbol}.")

        df = pd.DataFrame([n.__dict__ for n in market_nodes])
        print(f"[INFO] Ingest finalized | {len(df)} clean market instruments cached.")

        grid_k = np.linspace(-0.20, 0.20, 40)
        grid_T = np.sort(df["years_to_expiry"].unique())
        T_mesh, k_mesh = np.meshgrid(grid_T, grid_k)
        iv_surface = np.full_like(T_mesh, np.nan)  # NaN = slice not fitted

        last_w: Optional[np.ndarray] = None
        for idx, t_expiry in enumerate(grid_T):
            slice_df = (df[df["years_to_expiry"] == t_expiry]
                        .sort_values(by="log_moneyness"))
            if len(slice_df) < SVIVolatilitySurface.MIN_POINTS:
                print(f"[WARN] Skipping expiry {t_expiry:.4f}y: "
                      f"{len(slice_df)} < {SVIVolatilitySurface.MIN_POINTS} liquid options.")
                continue

            params = SVIVolatilitySurface.fit_slice(
                slice_df["log_moneyness"].values, slice_df["total_variance"].values,
                grid_k=grid_k, previous_fitted_w=last_w, T=t_expiry)

            # Native SVI wings (asymptotically linear) on the grid, so the
            # object stored for the next slice's calendar penalty is exactly
            # the object the penalty compares against.
            fitted_w = SVIVolatilitySurface.total_variance(grid_k, params)
            last_w = fitted_w.copy()
            iv_surface[:, idx] = np.sqrt(fitted_w / t_expiry)

        report = SurfaceIntegrityMonitor(S_spot, self.r, self.q).evaluate(
            T_mesh, k_mesh, iv_surface)
        self._print_report(report)
        return T_mesh, k_mesh, iv_surface, df

    @staticmethod
    def _print_report(r: IntegrityReport) -> None:
        print("\n" + "=" * 60)
        print("Volatility Surface Integrity Report")
        print("=" * 60)
        print(f"Butterfly violations : {r.butterfly_violations} / {r.butterfly_checks} checks")
        print(f"Calendar violations  : {r.calendar_violations} / {r.calendar_checks} checks")
        print(f"Alignment score      : {r.alignment_score:.2f}%")
        print(f"Surface nodes        : {r.total_nodes}")
        print("=" * 60 + "\n")

# Offline self-check (no network)
if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    tree = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, 300)
    bs = float(black_scholes(S, K, T, r, sigma))
    print(f"Call  tree={tree:.4f}  BS={bs:.4f}")
    assert abs(tree - bs) < 1e-2

    am_put = AmericanPricingEngine.price_put_binomial(S, K, T, r, sigma, 300)
    eu_put = float(black_scholes(S, K, T, r, sigma, is_call=False))
    print(f"Put   American={am_put:.4f}  European={eu_put:.4f}  (premium={am_put - eu_put:.4f})")
    assert am_put > eu_put

    iv = AmericanPricingEngine.calculate_implied_volatility(tree, S, K, T, r, N=300)
    print(f"IV round-trip: {iv:.4f} (true {sigma})")
    assert abs(iv - sigma) < 1e-4

    true = SVIParameters(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    k = np.linspace(-0.3, 0.3, 15)
    rng = np.random.default_rng(42)
    w = SVIVolatilitySurface.total_variance(k, true) + rng.normal(0, 5e-4, k.shape)
    fit = SVIVolatilitySurface.fit_slice(k, w, grid_k=np.linspace(-0.4, 0.4, 9))
    rmse = float(np.sqrt(np.mean((SVIVolatilitySurface.total_variance(k, fit) - w) ** 2)))
    print(f"SVI fit: {fit}\nRMSE on market points: {rmse:.2e}")
    assert rmse < 2e-3
    print("All offline self-checks passed.")