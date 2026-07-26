| model | method | runs | rel err (med [IQR]) | shots (med) | circuits (med) | ops (med) | near-opt | ambiguous |
|---|---|---|---|---|---|---|---|---|
| random_ising(n=6,obc) | doubling_grouped | 20 | 4.47e-03 [1.46e-03, 9.49e-03] | 589,568 | 300 | 8 | 0.94 | 0.66 |
| random_ising(n=6,obc) | exact | 20 | 4.47e-03 [1.43e-03, 1.00e-02] | 0 | 0 | 8 | 1.00 | 0.00 |
| random_ising(n=6,obc) | random | 20 | 6.18e-02 [4.62e-02, 8.46e-02] | 0 | 0 | 8 | 0.15 | 0.00 |
| random_ising(n=6,obc) | variance_grouped | 20 | 5.98e-03 [1.76e-03, 1.12e-02] | 1,304,384 | 264 | 8 | 0.96 | 0.21 |
| tfim(n=6,J=1.0,h=1.0,obc) | doubling_grouped | 20 | 1.55e-02 [3.39e-03, 1.56e-02] | 697,856 | 314 | 8 | 0.94 | 0.84 |
| tfim(n=6,J=1.0,h=1.0,obc) | exact | 20 | 2.71e-03 [2.71e-03, 2.71e-03] | 0 | 0 | 8 | 1.00 | 0.00 |
| tfim(n=6,J=1.0,h=1.0,obc) | random | 20 | 6.67e-02 [4.90e-02, 7.89e-02] | 0 | 0 | 8 | 0.37 | 0.00 |
| tfim(n=6,J=1.0,h=1.0,obc) | variance_grouped | 20 | 3.28e-03 [2.67e-03, 1.29e-02] | 1,539,666 | 302 | 8 | 0.93 | 0.47 |
| xxz(n=6,J=1.0,delta=1.0,obc) | doubling_grouped | 20 | 9.77e-02 [8.53e-02, 9.77e-02] | 1,937,408 | 884 | 8 | 0.54 | 0.99 |
| xxz(n=6,J=1.0,delta=1.0,obc) | exact | 20 | 9.77e-02 [9.77e-02, 9.77e-02] | 0 | 0 | 3 | 1.00 | 0.00 |
| xxz(n=6,J=1.0,delta=1.0,obc) | random | 20 | 3.75e-01 [3.21e-01, 4.99e-01] | 0 | 0 | 8 | 0.11 | 0.00 |
| xxz(n=6,J=1.0,delta=1.0,obc) | variance_grouped | 20 | 9.77e-02 [8.53e-02, 9.77e-02] | 1,506,345 | 961 | 8 | 0.51 | 0.85 |
