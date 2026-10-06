from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from numba import jit
from typing import Optional
from scipy.optimize import minimize


# 1. Domain & Model Config 
@dataclass(frozen=True)
class MarketNode:
    """Represents validated options market datum"""
    days_to_expiry: float
    strike: float
    log_moneyness: float
    market_price: float
    implied_vol: float
    total_variance: float

@dataclass(frozen=True)
class SVIParameters:
    """Gatheral SVI model parameter payload"""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

@dataclass(frozen=True)
class IntegrityReport:
    """Immutable validation summary for compliance & monitoring logging"""
    butterfly_violations: int
    calendar_violations: int
    alignment_score: float
    total_nodes: int


# 2. High-performance pricing layer
class AmericanPricingEngine:
    """High-performance option calculation engine"""

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)

    #Discrete-time binomial lattice (Cox-Ross-Rubinstein Tree) used to model the underlying asset's price path
    def price_call_binomial(S: float, K: float, T: float, r: float, sigma: float, N: int = 50) -> float:
        """Prices an American Call option using a memory-optimized binomial tree"""
        if T <= 0.0 or sigma <= 0.0:
            return float(max(0.0, S - K))

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        df = np.exp(-r * dt)

        if p >= 1.0 or p <= 0.0:
            return float(max(0.0, S - K))

        S_tree = np.zeros(N + 1)
        for j in range(N + 1):
            S_tree[j] = S * (u ** (N - j)) * (d ** j)

        C_tree = np.zeros(N + 1)
        for j in range(N + 1):
            C_tree[j] = max(0.0, S_tree[j] - K)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                S_node = S * (u ** (i - j)) * (d ** j)
                continuation = df * (p * C_tree[j] + (1.0 - p) * C_tree[j + 1])
                early_exercise = S_node - K
                C_tree[j] = max(continuation, early_exercise)

        return float(C_tree[0])

    #Uses Brent's method to calculate IV 
    @classmethod
    def calculate_implied_volatility(cls, C_market: float, S: float, K: float, T: float, r: float) -> float:
        """Inverts the American Binomial model to compute implied volatility."""
        intrinsic_value = max(0.0, S - K)
        if C_market <= intrinsic_value:
            return np.nan

        def objective(sigma: float) -> float:
            return cls.price_call_binomial(S, K, T, r, sigma, N = 50) - C_market

        try:
            result = root_scalar(objective, bracket=[0.01, 3.0], method='brentq')
            return result.root if result.converged else np.nan
        except ValueError:
            return np.nan


# 3. Volatility Monitoring & Extra-polation layer
class SVIVolatilitySurface:
    """Handles SVI surface calibrations, parameterization, and wing smoothing."""

    @staticmethod
    def total_variance(k: np.ndarray, params: SVIParameters) -> np.ndarray:
        """Vectorized Gatheral SVI implementation."""
        return params.a + params.b * (params.rho * (k - params.m) + np.sqrt((k - params.m)**2 + params.sigma**2))

    @classmethod
    def fit_slice(cls, k_market: np.ndarray, w_market: np.ndarray, grid_k: np.ndarray,
                  previous_fitted_w: Optional[np.ndarray] = None) -> SVIParameters:
        """Calibrates an SVI parametric slice under sequential regularization constraints."""

        #need to adjust penalties so not fixed rand numbers
        def objective(p_arr: np.ndarray) -> float:
            params = SVIParameters(*p_arr)
            w_pred_market = cls.total_variance(k_market, params)
            rmse = np.mean((w_market - w_pred_market) ** 2)

            # Structural Soft Penalties
            penalty = 0.0
            if params.b < 0: penalty += 50.0 * abs(params.b)
            if abs(params.rho) >= 1.0: penalty += 50.0 * (abs(params.rho) - 0.99)
            if params.sigma <= 0: penalty += 50.0 * abs(params.sigma)
            if any(w_pred_market < 0): penalty += 100.0 * np.sum(w_pred_market[w_pred_market < 0] ** -2)

            # Inter-slice Temporal Regularization (stopping calendar arbitrage)
            if previous_fitted_w is not None:
                w_pred_grid = cls.total_variance(grid_k, params)
                calendar_drift = previous_fitted_w - w_pred_grid
                violations = calendar_drift[calendar_drift > 0]
                if len(violations) > 0:
                    penalty += 150000.0 * np.sum(violations ** 2)

            return rmse + penalty

        initial_guess = [0.04, 0.1, -0.3, 0.0, 0.1]
        bounds = [(1e-5, 2.0), (1e-5, 2.0), (-0.95, 0.95), (-1.0, 1.0), (1e-4, 1.0)]
        res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        return SVIParameters(*res.x)


# 4. Ingestion & Pipeline 
class OptionsDataPipeline:
    """Manages raw vendor ingestion, data hygiene rules, and validation structures."""

    def __init__(self, ticker_symbol: str, risk_free_rate: float):
        self.ticker_symbol = ticker_symbol
        self.r = risk_free_rate
        self._ticker = yf.Ticker(ticker_symbol)

    def fetch_market_state(self) -> Tuple[float, List[MarketNode]]:
        """Ingests raw market data frames, processes types, and screens liquid matrices."""
        live_history = self._ticker.history(period="1d")
        if live_history.empty:
            raise ValueError(f"Failed to fetch market spot for symbol: {self.ticker_symbol}")

        S_spot = float(live_history['Close'].iloc[-1])
        expirations = self._ticker.options
        today = datetime.now()
        nodes: List[MarketNode] = []

        for exp_str in expirations[:7]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T_years = (exp_date - today).days / 365.25
            if T_years < (6 / 365.25):
                continue

            try:
                opt_chain = self._ticker.option_chain(exp_str)
                calls = opt_chain.calls
            except Exception:
                continue

            # Screen boundaries
            calls = calls[(calls['strike'] > S_spot * 0.80) & (calls['strike'] < S_spot * 1.20)]

            for _, contract in calls.iterrows():
                bid, ask, K = float(contract['bid']), float(contract['ask']), float(contract['strike'])
                if bid <= 0.02 or ask <= 0.02 or (ask - bid) / ask > 0.40:
                    continue

                C_mid = (bid + ask) / 2.0
                iv = AmericanPricingEngine.calculate_implied_volatility(C_mid, S_spot, K, T_years, self.r)

                if not np.isnan(iv) and 0.05 < iv < 2.0:
                    nodes.append(MarketNode(
                        days_to_expiry=T_years,
                        strike=K,
                        log_moneyness=float(np.log(K / S_spot)),
                        market_price=C_mid,
                        implied_vol=iv,
                        total_variance=float((iv ** 2) * T_years)
                    ))

        return S_spot, nodes


# 5. Validation & Analytics Monitor
class SurfaceIntegrityMonitor:
    """Calculates partial finite differences to assert matrix surface viability."""

    def __init__(self, S_spot: float, r: float):
        self.S_spot = S_spot
        self.r = r

    def evaluate(self, T_mesh: np.ndarray, k_mesh: np.ndarray, iv_surface: np.ndarray) -> IntegrityReport:
        """Performs localized edge testing for static butterfly and calendar structures."""
        K_mesh = self.S_spot * np.exp(k_mesh)
        call_prices = np.zeros_like(iv_surface)

        for i in range(iv_surface.shape[0]):
            for j in range(iv_surface.shape[1]):
                call_prices[i, j] = AmericanPricingEngine.price_call_binomial(
                    self.S_spot, K_mesh[i, j], T_mesh[i, j], self.r, iv_surface[i, j]
                )

        dK = np.diff(K_mesh, axis=0)
        d2C_dK2 = np.zeros_like(call_prices[:-2, :])
        for j in range(call_prices.shape[1]):
            d2C_dK2[:, j] = np.diff(np.diff(call_prices[:, j]) / dK[:, j]) / dK[:-1, j]

        dT = np.diff(T_mesh, axis=1)
        dC_dT = np.diff(call_prices, axis=1) / dT

        butterfly_violations = int(np.sum(d2C_dK2 < -1e-4))
        calendar_violations = int(np.sum(dC_dT < -1e-4))
        total_nodes = iv_surface.size
        alignment_score = ((total_nodes - (butterfly_violations + calendar_violations)) / total_nodes) * 100

        return IntegrityReport(
            butterfly_violations=butterfly_violations,
            calendar_violations=calendar_violations,
            alignment_score=alignment_score,
            total_nodes=total_nodes
        )

try:
    print("Running strict validation test...")

    # We pass extreme inputs to break the model:
    # S=100, K=100, T=1 year, r=200% (2.0), sigma=1% (0.01), N=1 step
    AmericanPricingEngine.price_call_binomial(S=100, K=100, T=1.0, r=2.0, sigma=0.01, N=1)

except ValueError as e:
    print(f"\n[SUCCESS] The validation worked perfectly!")
    print(f"Error Message: {e}")


# Overall Pricing tests
if __name__ == "__main__":
    S = 100.0  # Spot price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity (1 year)
    r = 0.05   # Risk-free rate
    sigma = 0.2 # Volatility (20%)

    # 1. Test American Call Pricing
    call_price = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=50)
    print(f"Calculated American Call Price: {call_price:.4f}")
    
    # 2. Test IV Inversion
    iv = AmericanPricingEngine.calculate_implied_volatility(call_price, S, K, T, r)
    print(f"Recovered Implied Volatility: {iv:.4f}")

    # 3. Test SVI Total Variance Calculation
    print("\n--- Testing SVI Total Variance ---")
    test_params = SVIParameters(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    log_moneyness_grid = np.linspace(-0.4, 0.4, 9)
    
    w_grid = SVIVolatilitySurface.total_variance(log_moneyness_grid, test_params)
    print("Log-Moneyness Grid:", np.round(log_moneyness_grid, 3))
    print("Calculated Total Variance:", np.round(w_grid, 4))

    # 4. Test SVI Slice Calibration (fit_slice)
    print("\n--- Testing SVI Slice Calibration ---")
    np.random.seed(42)
    k_market = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
    w_market = SVIVolatilitySurface.total_variance(k_market, test_params) + np.random.normal(0, 0.0005, size=k_market.shape)
    
    # Fit the SVI slice
    fitted_params = SVIVolatilitySurface.fit_slice(
        k_market=k_market, 
        w_market=w_market, 
        grid_k=log_moneyness_grid
    )
    
    print("Fitted SVI Parameters:")
    print(f"  a     = {fitted_params.a:.4f} (True: {test_params.a})")
    print(f"  b     = {fitted_params.b:.4f} (True: {test_params.b})")
    print(f"  rho   = {fitted_params.rho:.4f} (True: {test_params.rho})")
    print(f"  m     = {fitted_params.m:.4f} (True: {test_params.m})")
    print(f"  sigma = {fitted_params.sigma:.4f} (True: {test_params.sigma})")
