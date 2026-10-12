import math
import os
import platform
import statistics
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
from src.pipeline import AmericanPricingEngine
 
# Pure Python baseline
def price_call_pure_python(S, K, T, r, sigma, N=300):
    """Plain-float baseline using `math` (no NumPy scalars, so it is not artificially slow)."""
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    V = [max(0.0, S * u ** (N - 2 * j) - K) for j in range(N + 1)]
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = df * (p * V[j] + (1.0 - p) * V[j + 1])
            V[j] = max(cont, S * u ** (i - 2 * j) - K)
    return V[0]
 
def run_benchmark(iterations=200):
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.04, 0.20, 300
    jit_px = AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N)  # warm-up / compile
    py_px = price_call_pure_python(S, K, T, r, sigma, N)
    assert abs(jit_px - py_px) < 1e-8, f"implementations disagree: {jit_px} vs {py_px}"

    # Time Pure Python
    t0 = time.perf_counter()
    for _ in range(iterations):
        price_call_pure_python(S, K, T, r, sigma, N)
    py_t = time.perf_counter() - t0

    # Time Numba Jit
    t0 = time.perf_counter()
    for _ in range(iterations):
        AmericanPricingEngine.price_call_binomial(S, K, T, r, sigma, N)
    jit_t = time.perf_counter() - t0
 
    print("=" * 60)
    print(f"Price check      : {py_px:.8f} (python) == {jit_px:.8f} (numba)")
    print(f"Pure Python      : {py_t / iterations * 1e3:.4f} ms/option ({iterations / py_t:.1f} ops/s)")
    print(f"Numba JIT        : {jit_t / iterations * 1e3:.4f} ms/option ({iterations / jit_t:.1f} ops/s)")
    print(f"Speedup          : {py_t / jit_t:.2f}x   (N={N}, {iterations} runs)")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run_benchmark()