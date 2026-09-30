### Benchmark Results for 1 Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `0.13431` | 1.0x | `3.72e-15` |
| SciPy `quad_vec` | Auto | `0.21122` | **0.6x** | `3.72e-15` |
| TSP Vectorized | 121 | `0.00081` | **165.4x** | `1.07e-14` |
| TSP Batched | 121 | `0.00049` | **275.7x** | `2.93e-14` |
| TSP Vectorized | 301 | `0.00196` | **76.6x** | `2.09e-14` |
| TSP Batched | 301 | `0.00096` | **156.2x** | `9.19e-14` |
| TSP Vectorized | 501 | `0.00307` | **39.2x** | `2.22e-14` |
| TSP Batched | 501 | `0.00161` | **74.6x** | `2.27e-13` |
| TSP Vectorized | 1001 | `0.00603` | **17.4x** | `2.28e-13` |
| TSP Batched | 1001 | `0.00311` | **33.6x** | `1.49e-13` |
| TSP Vectorized | 2001 | `0.01178` | **8.7x** | `4.58e-13` |
| TSP Batched | 2001 | `0.00584` | **17.6x** | `2.69e-13` |
| TSP Vectorized | 5001 | `0.04047` | **2.8x** | `7.99e-14` |
| TSP Batched | 5001 | `0.01719` | **6.7x** | `1.05e-12` |

### Benchmark Results for 2 Fourier Gaussian

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `2.23663` | 1.0x | `7.92e-10` |
| SciPy `quad_vec` | Auto | `10.40842` | **0.2x** | `7.92e-10` |
| TSP Vectorized | 121 | `0.00148` | **1506.8x** | `2.65e-01` |
| TSP Batched | 121 | `0.00099` | **2258.8x** | `2.65e-01` |
| TSP Vectorized | 301 | `0.00376` | **522.1x** | `6.58e-03` |
| TSP Batched | 301 | `0.00240` | **815.9x** | `6.58e-03` |
| TSP Vectorized | 501 | `0.00570` | **323.2x** | `4.58e-05` |
| TSP Batched | 501 | `0.00358` | **513.7x** | `4.58e-05` |
| TSP Vectorized | 1001 | `0.01135` | **164.5x** | `9.88e-12` |
| TSP Batched | 1001 | `0.00700` | **266.6x** | `9.88e-12` |
| TSP Vectorized | 2001 | `0.02072` | **89.0x** | `3.06e-13` |
| TSP Batched | 2001 | `0.01309` | **141.0x** | `1.81e-13` |
| TSP Vectorized | 5001 | `0.07477` | **26.5x** | `5.00e-14` |
| TSP Batched | 5001 | `0.04132` | **47.9x** | `6.94e-13` |

### Benchmark Results for 3 Damped Cosine

| Method | N | ms/pt | Speedup | Error |
| :--- | :--- | :--- | :--- | :--- |
| SciPy `quad` | Auto | `1.29816` | 1.0x | `2.41e-09` |
| SciPy `quad_vec` | Auto | `3.85162` | **0.3x** | `2.41e-09` |
| TSP Vectorized | 121 | `0.00056` | **2320.0x** | `2.07e-01` |
| TSP Batched | 121 | `0.00030` | **4264.6x** | `2.07e-01` |
| TSP Vectorized | 301 | `0.00151` | **845.0x** | `5.83e-02` |
| TSP Batched | 301 | `0.00047` | **2731.2x** | `5.83e-02` |
| TSP Vectorized | 501 | `0.00256` | **487.7x** | `2.20e-02` |
| TSP Batched | 501 | `0.00094` | **1325.1x** | `2.20e-02` |
| TSP Vectorized | 1001 | `0.00526` | **248.7x** | `2.67e-03` |
| TSP Batched | 1001 | `0.00264` | **495.6x** | `2.67e-03` |
| TSP Vectorized | 2001 | `0.00970` | **138.0x** | `5.72e-05` |
| TSP Batched | 2001 | `0.00545` | **245.7x** | `5.72e-05` |
| TSP Vectorized | 5001 | `0.02904` | **42.9x** | `2.90e-09` |
| TSP Batched | 5001 | `0.01382` | **90.2x** | `2.90e-09` |
