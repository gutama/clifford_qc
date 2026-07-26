| model | method | runs | rel err (med [IQR]) | shots (med) | circuits (med) | ops (med) | near-opt | ambiguous |
|---|---|---|---|---|---|---|---|---|
| random_ising(n=4,obc) | doubling | 30 | 3.00e-05 [8.43e-16, 2.23e-04] | 1,739,776 | 914 | 8 | 0.78 | 0.66 |
| random_ising(n=4,obc) | doubling_grouped | 30 | 1.06e-05 [1.43e-15, 1.99e-04] | 1,638,656 | 278 | 8 | 0.72 | 0.66 |
| random_ising(n=4,obc) | exact | 30 | 1.31e-15 [6.66e-16, 1.16e-05] | 0 | 0 | 6 | 1.00 | 0.00 |
| random_ising(n=4,obc) | layered_exact | 30 | 1.17e-15 [6.79e-16, 5.35e-15] | 0 | 0 | 8 | 1.00 | 0.00 |
| random_ising(n=4,obc) | random | 30 | 2.76e-02 [1.17e-02, 6.70e-02] | 0 | 0 | 8 | 0.15 | 0.00 |
| random_ising(n=4,obc) | subpool_exact | 30 | 2.21e-05 [1.22e-15, 2.23e-04] | 0 | 0 | 5 | 1.00 | 0.00 |
| random_ising(n=4,obc) | variance_grouped | 30 | 8.74e-06 [8.43e-16, 1.46e-04] | 3,483,843 | 248 | 8 | 0.82 | 0.40 |
| tfim(n=4,J=1.0,h=1.0,obc) | doubling | 30 | 5.15e-05 [5.15e-05, 3.11e-04] | 1,882,368 | 926 | 8 | 0.71 | 0.76 |
| tfim(n=4,J=1.0,h=1.0,obc) | doubling_grouped | 30 | 5.15e-05 [5.15e-05, 3.11e-04] | 1,847,680 | 284 | 8 | 0.72 | 0.78 |
| tfim(n=4,J=1.0,h=1.0,obc) | exact | 30 | 1.49e-15 [1.49e-15, 1.49e-15] | 0 | 0 | 6 | 1.00 | 0.00 |
| tfim(n=4,J=1.0,h=1.0,obc) | layered_exact | 30 | 1.12e-15 [1.12e-15, 1.12e-15] | 0 | 0 | 12 | 1.00 | 0.00 |
| tfim(n=4,J=1.0,h=1.0,obc) | random | 30 | 2.22e-02 [1.99e-02, 5.76e-02] | 0 | 0 | 8 | 0.25 | 0.00 |
| tfim(n=4,J=1.0,h=1.0,obc) | subpool_exact | 30 | 3.11e-04 [3.11e-04, 3.11e-04] | 0 | 0 | 5 | 1.00 | 0.00 |
| tfim(n=4,J=1.0,h=1.0,obc) | variance_grouped | 30 | 5.15e-05 [1.12e-15, 5.15e-05] | 3,824,366 | 272 | 8 | 0.77 | 0.45 |
