### Benchmark Results for 1 Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `0.10326` | 1.0x | `3.72e-15` |
| SciPy `quad_vec` | Auto | `0.21602` | **0.5x** | `3.72e-15` |
| TSP Vectorized | 121 | `0.00071` | **145.7x** | `1.07e-14` |
| TSP Batched | 121 | `0.00041` | **250.3x** | `2.93e-14` |
| TSP Vectorized | 301 | `0.00163` | **61.3x** | `2.09e-14` |
| TSP Batched | 301 | `0.00091` | **110.1x** | `9.19e-14` |
| TSP Vectorized | 501 | `0.00275` | **38.4x** | `2.22e-14` |
| TSP Batched | 501 | `0.00149` | **71.0x** | `2.27e-13` |
| TSP Vectorized | 1001 | `0.00527` | **19.6x** | `2.28e-13` |
| TSP Batched | 1001 | `0.00280` | **36.8x** | `1.49e-13` |
| TSP Vectorized | 2001 | `0.01088` | **10.8x** | `4.58e-13` |
| TSP Batched | 2001 | `0.00605` | **19.4x** | `2.69e-13` |
| TSP Vectorized | 5001 | `0.03577` | **2.7x** | `7.99e-14` |
| TSP Batched | 5001 | `0.01474` | **6.5x** | `1.05e-12` |

### Benchmark Results for 2 Fourier Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `1.81748` | 1.0x | `7.92e-10` |
| SciPy `quad_vec` | Auto | `8.48350` | **0.2x** | `7.92e-10` |
| TSP Vectorized | 121 | `0.00129` | **1410.2x** | `2.65e-01` |
| TSP Batched | 121 | `0.00081` | **2240.9x** | `2.65e-01` |
| TSP Vectorized | 301 | `0.00326` | **558.2x** | `6.58e-03` |
| TSP Batched | 301 | `0.00174` | **1044.6x** | `6.58e-03` |
| TSP Vectorized | 501 | `0.00554` | **345.5x** | `4.58e-05` |
| TSP Batched | 501 | `0.00337` | **567.8x** | `4.58e-05` |
| TSP Vectorized | 1001 | `0.01090` | **171.2x** | `9.88e-12` |
| TSP Batched | 1001 | `0.00655` | **284.8x** | `9.88e-12` |
| TSP Vectorized | 2001 | `0.02289` | **90.3x** | `3.06e-13` |
| TSP Batched | 2001 | `0.01501` | **137.7x** | `1.81e-13` |
| TSP Vectorized | 5001 | `0.07378` | **27.9x** | `5.00e-14` |
| TSP Batched | 5001 | `0.04323` | **47.6x** | `6.94e-13` |

### Benchmark Results for 3 Damped Cosine

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `1.35999` | 1.0x | `2.41e-09` |
| SciPy `quad_vec` | Auto | `4.25951` | **0.3x** | `2.41e-09` |
| TSP Vectorized | 121 | `0.00067` | **2025.7x** | `2.07e-01` |
| TSP Batched | 121 | `0.00039` | **3507.9x** | `2.07e-01` |
| TSP Vectorized | 301 | `0.00162` | **808.1x** | `5.83e-02` |
| TSP Batched | 301 | `0.00054` | **2435.3x** | `5.83e-02` |
| TSP Vectorized | 501 | `0.00279` | **483.8x** | `2.20e-02` |
| TSP Batched | 501 | `0.00124` | **1083.9x** | `2.20e-02` |
| TSP Vectorized | 1001 | `0.00541` | **257.6x** | `2.67e-03` |
| TSP Batched | 1001 | `0.00281` | **496.9x** | `2.67e-03` |
| TSP Vectorized | 2001 | `0.01063` | **134.0x** | `5.72e-05` |
| TSP Batched | 2001 | `0.00586` | **243.1x** | `5.72e-05` |
| TSP Vectorized | 5001 | `0.02857` | **48.4x** | `2.90e-09` |
| TSP Batched | 5001 | `0.01381` | **100.1x** | `2.90e-09` |
