import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.pipeline import AmericanPricingEngine

# Pure Python baseline
def price_call_binomial_pure_python(
    S: float, K: float, T: float, r: float, sigma: float, N: int = 300
) -> float:
  if T <= 0.0 or sigma <= 0.0:
    return float(max(0.0, S - K))

  dt = T / N
  u = np.exp(sigma * np.sqrt(dt))
  d = 1.0 / u
  p = (np.exp(r * dt) - d) / (u - d)
  df = np.exp(-r * dt)

  if p >= 1.0 or p <= 0.0:
    return float(max(0.0, S - K))

  S_tree = [S * (u ** (N - j)) * (d ** j) for j in range(N + 1)]
  C_tree = [max(0.0, S_node - K) for S_node in S_tree]

  for i in range(N - 1, -1, -1):
    for j in range(i + 1):
      S_node = S * (u ** (i - j)) * (d ** j)
      continuation = df * (p * C_tree[j] + (1.0 - p) * C_tree[j + 1])
      early_exercise = S_node - K
      C_tree[j] = max(continuation, early_exercise)

  return float(C_tree[0])

def test_binomial_benchmark():
  S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.04, 0.20, 300
  iterations = 200

  # Warmup run for Numba JIT
  _ = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=N)

  # Time Pure Python
  t0 = time.perf_counter()
  for _ in range(iterations):
    _ = price_call_binomial_pure_python(S, K, T, r, sigma, N=N)
  py_time = time.perf_counter() - t0

  # Time Numba JIT
  t0 = time.perf_counter()
  for _ in range(iterations):
    _ = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=N)
  jit_time = time.perf_counter() - t0

  print(f"\nPure Python Time : {py_time:.4f}s")
  print(f"Numba JIT Time   : {jit_time:.4f}s")
  print(f"Speedup          : {py_time / jit_time:.1f}x faster")