# Volatility Surface Engine
Python pipeline that inverts implied volatilities from listed call options with a Numba-compiled binomial pricer, fits per-expiry **SVI** (Stochastic Volatility Inspired) slices, and checks the fitted surface for static arbitrage.

## Reference Acceleration Profiles & Benchmarks
The binomial lattice was benchmarked over **200 evaluations** at tree depth **N=300**, comparing a plain-float Python implementation against the Numba-compiled one.

| Metric | Pure Python | Numba JIT |
| :--- | :--- | :--- |
| Latency (ms / option) | 10.18 | 0.0332 |
| Throughput (ops / sec) | 98.2 | 30,159 |
| Speedup | 1.00x | ~307x (99.67% time saved) |

Setup: N = 300, 200 evaluations, GitHub Codespaces. Baseline is a plain-float Python lattice with the same algorithm. Numba compile time is excluded by a warm-up call.

### Core Pipeline Components
*   **AmericanPricingEngine:** Cox-Ross-Rubinstein binomial lattice for American calls and puts with a dividend yield. A precomputed power table and a single 1-D value buffer remove `pow()` calls from the O(N²) loop, and `@jit(nopython=True)` compiles it to machine code.
*   **SVIVolatilitySurface:** Fits Gatheral's 5-parameter SVI model (`a, b, rho, m, sigma`) per expiry with `L-BFGS-B`. Soft penalties discourage calendar arbitrage between slices and wing-slope violations (`b(1+|rho|) ≤ 2`).
*   **OptionsDataPipeline:** yfinance ingestion with liquidity and spread filters, producing validated, immutable dataclasses (`MarketNode`, `SVIParameters`).

## Verification & Test Suite
Numerical accuracy, mathematical boundary conditions, and execution optimization are verified with a `pytest` suite.

```bash
$ pytest test -v
```

### Test Suite Execution Output
```text
$ pytest test -v
...
============================== 19 passed in 2.33s ==============================
```

### Benchmark Output
```text
$ python test/benchmark.py
============================================================
Price check      : 9.91842458 (python) == 9.91842458 (numba)
Pure Python      : 10.1799 ms/option (98.2 ops/s)
Numba JIT        : 0.0332 ms/option (30159.2 ops/s)
Speedup          : 307.02x   (N=300, 200 runs)
============================================================
```
## Live Run

SPY, Oct 12 2026 (after-hours quotes), r = 4.2%, q = 1.2%. Two near-term weekly expiries passed the liquidity filters, giving 114 clean instruments and a 40 x 2 surface grid.

```bash
python -c "from src.pipeline import VolatilitySurfaceOrchestrator as V; V('SPY', 0.042, 0.012).execute_lifecycle()"
```

```text
[INFO] Ingest finalized | 114 clean market instruments cached.

============================================================
Volatility Surface Integrity Report
============================================================
Butterfly violations : 7 / 76 checks
Calendar violations  : 0 / 40 checks
Alignment score      : 93.97%
Surface nodes        : 80
============================================================
```

The monitor reports butterfly arbitrage but the fit does not enforce it (see Known limitations). With only two expiries, the calendar result rests on a single maturity pair.

*   **Convergence Verification:** Asserts that the discrete American Pricing Engine correctly converges to analytical European Black-Scholes values within an absolute tolerance of ≤ 10⁻² when assets pay no dividends.
*   **Compiler Sanity Checks:** The `test_engine_compilation_status` routine inspects the internal method target properties to ensure the Numba JIT dispatcher has fully bound machine-code signatures, ensuring zero silent interpreter fallbacks.
*   **Calibration Tests:** IV round-trip for calls and puts, SVI recovery of a noisy synthetic slice, calendar-penalty behavior, American put and dividend-call early-exercise premium, and arbitrage-monitor detection.

## How to use:

### 1. Installation
Installing the required numerical dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run tests and benchmark
```bash
pytest test -v
python test/benchmark.py
```

## Known limitations

- **Calls only**, priced at the bid/ask mid, with constant `r` and `q`; no discrete dividends or term structure of rates.
- Calendar and wing constraints are **soft penalties**, so no-arbitrage is discouraged, not guaranteed. Butterfly arbitrage in the body of the smile is reported by the monitor, not enforced in the fit.
- The monitor prices with European Black-Scholes while IVs come from an American lattice. For dividend-paying underlyings with deep ITM strikes these differ slightly.
- This model is meant for more accurate and timely data; Yahoo quotes are delayed and noisy, and results depend on the filter settings.
- IV inversion at N = 200 trades accuracy for speed; lattice error is roughly 1e-2 in price.