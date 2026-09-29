### Benchmark Results for 1 Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `0.10331` | 1.0x | `3.72e-15` |
| SciPy `quad_vec` | Auto | `0.18847` | **0.5x** | `3.72e-15` |
| TSP Vectorized | 121 | `0.00067` | **154.6x** | `1.07e-14` |
| TSP Batched | 121 | `0.00038` | **269.9x** | `2.93e-14` |
| TSP Vectorized | 301 | `0.00146` | **71.7x** | `2.09e-14` |
| TSP Batched | 301 | `0.00081` | **128.9x** | `9.19e-14` |
| TSP Vectorized | 501 | `0.00235` | **39.0x** | `2.22e-14` |
| TSP Batched | 501 | `0.00132` | **69.5x** | `2.27e-13` |
| TSP Vectorized | 1001 | `0.00487` | **18.4x** | `2.28e-13` |
| TSP Batched | 1001 | `0.00266` | **33.7x** | `1.49e-13` |
| TSP Vectorized | 2001 | `0.00988` | **9.8x** | `4.58e-13` |
| TSP Batched | 2001 | `0.00551` | **17.7x** | `2.69e-13` |
| TSP Vectorized | 5001 | `0.03448` | **3.5x** | `7.99e-14` |
| TSP Batched | 5001 | `0.01397` | **8.7x** | `1.05e-12` |

### Benchmark Results for 2 Fourier Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `1.88812` | 1.0x | `7.92e-10` |
| SciPy `quad_vec` | Auto | `8.61286` | **0.2x** | `7.92e-10` |
| TSP Vectorized | 121 | `0.00120` | **1578.1x** | `2.65e-01` |
| TSP Batched | 121 | `0.00077` | **2438.7x** | `2.65e-01` |
| TSP Vectorized | 301 | `0.00313` | **580.3x** | `6.58e-03` |
| TSP Batched | 301 | `0.00163` | **1111.4x** | `6.58e-03` |
| TSP Vectorized | 501 | `0.00500` | **355.9x** | `4.58e-05` |
| TSP Batched | 501 | `0.00285` | **624.1x** | `4.58e-05` |
| TSP Vectorized | 1001 | `0.01016` | **172.7x** | `9.88e-12` |
| TSP Batched | 1001 | `0.00621` | **283.0x** | `9.88e-12` |
| TSP Vectorized | 2001 | `0.02071` | **89.8x** | `3.06e-13` |
| TSP Batched | 2001 | `0.01225` | **151.9x** | `1.81e-13` |
| TSP Vectorized | 5001 | `0.07997` | **24.7x** | `5.00e-14` |
| TSP Batched | 5001 | `0.03766` | **52.5x** | `6.94e-13` |

### Benchmark Results for 3 Damped Cosine

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `1.35558` | 1.0x | `2.41e-09` |
| SciPy `quad_vec` | Auto | `4.12918` | **0.3x** | `2.41e-09` |
| TSP Vectorized | 121 | `0.00089` | **1528.0x** | `2.07e-01` |
| TSP Batched | 121 | `0.00039` | **3465.6x** | `2.07e-01` |
| TSP Vectorized | 301 | `0.00159` | **828.2x** | `5.83e-02` |
| TSP Batched | 301 | `0.00070` | **1868.1x** | `5.83e-02` |
| TSP Vectorized | 501 | `0.00287` | **456.5x** | `2.20e-02` |
| TSP Batched | 501 | `0.00119` | **1104.2x** | `2.20e-02` |
| TSP Vectorized | 1001 | `0.00556` | **254.9x** | `2.67e-03` |
| TSP Batched | 1001 | `0.00281` | **504.5x** | `2.67e-03` |
| TSP Vectorized | 2001 | `0.01105` | **154.9x** | `5.72e-05` |
| TSP Batched | 2001 | `0.00581` | **294.6x** | `5.72e-05` |
| TSP Vectorized | 5001 | `0.03123` | **43.6x** | `2.90e-09` |
| TSP Batched | 5001 | `0.01553` | **87.6x** | `2.90e-09` |
