from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from numba import jit


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

# Pricing test
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