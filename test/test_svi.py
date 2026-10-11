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

def run_benchmark():
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.04, 0.20, 300
    iterations = 200

    print("⚡ Warmup run for Numba JIT (triggering compilation)...")
    _ = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N=N)

    print(f"Running benchmark across {iterations} evaluations (N={N} tree depth)...")

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

    # Compute high-performance computing metrics
    avg_py_latency_ms = (py_time / iterations) * 1000   
    avg_jit_latency_ms = (jit_time / iterations) * 1000 
    
    py_throughput = iterations / py_time                
    jit_throughput = iterations / jit_time              
    
    time_saved_pct = ((py_time - jit_time) / py_time) * 100
    speedup_ratio = py_time / jit_time

    print("\n" + "=" * 60)
    print("Computing Benchmark Results")
    print("=" * 60)
    print(f"Baseline Pure Python Throughput : {py_throughput:.2f} ops/sec")
    print(f"Numba JIT Accelerated Throughput: {jit_throughput:.2f} ops/sec")
    print("-" * 60)
    print(f"Average Baseline Latency        : {avg_py_latency_ms:.4f} ms per option")
    print(f"Average JIT Latency             : {avg_jit_latency_ms:.4f} ms per option")
    print("-" * 60)
    print(f"Compute Time Saved              : {time_saved_pct:.2f}%")
    print(f"Performance Speedup Scaling     : {speedup_ratio:.2f}x execution scaling")
    print("=" * 60 + "\n")
    
if __name__ == "__main__":
    run_benchmark()
