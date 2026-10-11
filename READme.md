# Volatility Surface Engine
A high-performance computational pipeline designed to calibrate Gatheral's **Stochastic Volatility Inspired (SVI)** parametric surface using accelerated numerical lattices. The calculation features memory-optimized binomial pricing lattices accelerated via **Numba JIT compilation** to improve execution efficiency for low-latency derivatives modeling.

## Reference Acceleration Profiles & Benchmarks
The mathematical lattice was benchmarked across **200 evaluations** at a tree depth of **N=300** to measure the throughput scaling achieved by migrating away from the pure Python overhead to native JIT-compiled instructions. 

| **Metric** | **Baseline Pure Python** | **Numba JIT Accelerated** | **Optimization Gain** |
| :--- | :--- | :--- | :--- |
| **Computational Throughput** | 26.46 ops/sec | **398.29 ops/sec** | **+1,405.25% scaling** |
| **Average Evaluation Latency** | 37.7894 ms | **2.5107 ms** | **93.36% time reduction** |
| **Execution Acceleration** | 1.00x | **15.05x faster** | **Hardware-native scaling** |

> *Note: The benchmarks above were recorded using a standard GitHub Codespaces cloud environment (Dual-Core Intel Xeon @ 2.80GHz, 8GB RAM). Actual execution throughput scaling and latency reductions will scale relative to local CPU architecture, cache size, and thread availability.*

### Core Pipeline Components
*   **AmericanPricingEngine:** Features a memory-efficient implementation of the Cox-Ross-Rubinstein (CRR) Binomial Tree. By substituting dynamic nested lists with pre-allocated `numpy.zeros` array buffers and applying `@jit(nopython=True, fastmath=True)`, looping execution bypasses the CPython interpreter entirely.
*   **SVIVolatilitySurface:** Calibrates Gatheral's 5-parameter SVI model (`a, b, rho, m, sigma`). Incorporates temporal regularization to actively penalize and eliminate static calendar arbitrage across option maturities during non-linear optimization (`L-BFGS-B`).
*   **OptionsDataPipeline:** Handles asynchronous financial matrix serialization, enforcing strong data hygiene rules via immutable Python `@dataclass` structures (`MarketNode`, `SVIParameters`).

## Verification & Test Suite
Numerical accuracy, mathematical boundary conditions, and execution optimization are verified continuously via a strict `pytest` regression suite.

```bash
$ pytest test/test_pricing.py -v
```

### Test Suite Execution Output
```text
test_pricing.py::test_binomial_convergence_to_black_scholes[100.0-100.0-1.0-0.04-0.2] PASSED  [ 20%]
test_pricing.py::test_binomial_convergence_to_black_scholes[100.0-90.0-0.5-0.04-0.25] PASSED  [ 40%]
test_pricing.py::test_binomial_convergence_to_black_scholes[100.0-110.0-0.25-0.04-0.15] PASSED [ 60%]
test_pricing.py::test_binomial_edge_cases PASSED                                              [ 80%]
test_pricing.py::test_engine_compilation_status PASSED                                        [100%]

================================= 5 passed in 3.83s =================================
```

```test
Running strict validation test...
Warmup run for Numba JIT (triggering compilation)...
Running benchmark across 200 evaluations (N=300 tree depth)...
============================================================
Computing Benchmark Results
------------------------------------------------------------
Baseline Pure Python Throughput : 26.46 ops/sec
Numba JIT Accelerated Throughput: 398.29 ops/sec
------------------------------------------------------------
Average Baseline Latency        : 37.7894 ms per option
Average JIT Latency             : 2.5107 ms per option
------------------------------------------------------------
Compute Time Saved              : 93.36%
Performance Speedup Scaling     : 15.05x execution scaling
============================================================
```
*   **Convergence Verification:** Asserts that the discrete American Pricing Engine correctly converges to analytical European Black-Scholes values within an absolute tolerance of ≤ 10⁻² when assets pay no dividends.
*   **Compiler Sanity Checks:** The `test_engine_compilation_status` routine inspects the internal method target properties to ensure the Numba JIT dispatcher has fully bound machine-code signatures, ensuring zero silent interpreter fallbacks.

## How to use:

### 1. Installation
Installing the required numerical dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Benchmark Performance Dashboards
Run the optimization script to verify the throughput engine scaling inside your execution environment:
```bash
python test/test_pricing.py
python test/test_svi.py
```