| model | method | runs | rel err (med [IQR]) | shots (med) | circuits (med) | ops (med) | near-opt | ambiguous |
|---|---|---|---|---|---|---|---|---|
| random_ising(n=6,obc) | doubling_grouped | 20 | 4.08e-03 [1.78e-03, 8.67e-03] | 2,286,720 | 268 | 8 | 0.96 | 0.54 |
| random_ising(n=6,obc) | exact | 20 | 4.47e-03 [1.43e-03, 1.00e-02] | 0 | 0 | 8 | 1.00 | 0.00 |
| random_ising(n=6,obc) | random | 20 | 6.18e-02 [4.62e-02, 8.46e-02] | 0 | 0 | 8 | 0.15 | 0.00 |
| random_ising(n=6,obc) | variance_grouped | 20 | 5.28e-03 [1.80e-03, 1.02e-02] | 3,534,118 | 240 | 8 | 0.96 | 0.17 |
| tfim(n=6,J=1.0,h=1.0,obc) | doubling_grouped | 20 | 1.53e-02 [2.95e-03, 1.56e-02] | 2,841,984 | 274 | 8 | 0.95 | 0.83 |
| tfim(n=6,J=1.0,h=1.0,obc) | exact | 20 | 2.71e-03 [2.71e-03, 2.71e-03] | 0 | 0 | 8 | 1.00 | 0.00 |
| tfim(n=6,J=1.0,h=1.0,obc) | random | 20 | 6.67e-02 [4.90e-02, 7.89e-02] | 0 | 0 | 8 | 0.37 | 0.00 |
| tfim(n=6,J=1.0,h=1.0,obc) | variance_grouped | 20 | 1.32e-02 [2.81e-03, 1.56e-02] | 4,415,612 | 266 | 8 | 0.93 | 0.41 |
| xxz(n=6,J=1.0,delta=1.0,obc) | doubling_grouped | 20 | 9.77e-02 [9.15e-02, 9.77e-02] | 14,483,840 | 908 | 8 | 0.51 | 0.94 |
| xxz(n=6,J=1.0,delta=1.0,obc) | exact | 20 | 9.77e-02 [9.77e-02, 9.77e-02] | 0 | 0 | 3 | 1.00 | 0.00 |
| xxz(n=6,J=1.0,delta=1.0,obc) | random | 20 | 3.75e-01 [3.21e-01, 4.99e-01] | 0 | 0 | 8 | 0.11 | 0.00 |
| xxz(n=6,J=1.0,delta=1.0,obc) | variance_grouped | 20 | 9.77e-02 [9.77e-02, 9.77e-02] | 5,891,534 | 946 | 8 | 0.59 | 0.75 |

wrote benchmarks/reference_results/spin_n6_summary.csv
