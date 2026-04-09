# Experiment Set 1
- 2D
- $h = 0.01$
- $b$ from spinup simulation
- Preconditioners of the form $P = \left(\begin{array}{c c} \tilde A^{-1} & 0\\0 & \varepsilon^{2} \tilde M_p^{-1} \end{array}\right)$
    - $\tilde M_p^{-1} = $ 4 `cg` iterations preconditioned by `diag(M_p)` unless otherwise stated
    - Using GMRES with `atol = 1e-6`, `rtol = 1.5e-8`, `k=20` unless otherwise stated

### Classic Stokes ($\varepsilon^2 = 1$, $\gamma = 1$, $f = 0$)

__NOTE__ : $A$ is SPD

See `*1.png` for images of solution.

- $\tilde A^{-1} =$ `lu(A)`: converges in __15__ iterations 
- $\tilde A^{-1} =$ `ilu(A, τ=1e-5)`: converges in __17__ iterations
- $\tilde A^{-1} =$ `ilu(A, τ=1e-4)`: converges in __26__ iterations
- $\tilde A^{-1} =$ `ilu(A, τ=1e-3)`: converges in __72__ iterations
- no blocks, just `1/h^2` normalization: __53,581__ iterations (steady slope)

### Aspect Ratio Stokes ($\varepsilon^2 = 1$, $\gamma = 1/4$, $f = 0$)

__NOTE__ : $A$ is SPD

See `*2.png` for images of solution.

- $\tilde A^{-1} =$ `lu(A)`: converges in __46__ iterations 
- $\tilde A^{-1} =$ `ilu(A, τ=1e-3)`: converges in __218__ iterations

### Less Diff Stokes ($\varepsilon^2 = 10^{-4}$, $\gamma = 1$, $f = 0$)

__NOTE__ : $A$ is _neither_ symmetric _nor_ positive definite!

See `*3.png` for images of solution.

- $\tilde A^{-1} =$ `lu(A)`: converges in __15__ iterations 
- $\tilde A^{-1} =$ `ilu(A, τ=1e-10)`: converges in __26__ iterations
- $\tilde A^{-1} =$ `ilu(A, τ=1e-9)`: converges in __30__ iterations
- $\tilde A^{-1} =$ `ilu(A, τ=1e-8)`: converges in __55__ iterations
- $\tilde A^{-1} =$ `ilu(A, τ=1e-7)`: converges in __185__ iterations

### PG Thick BL ($\varepsilon^2 = 1$, $\gamma = 1$, $f = 1$)

__NOTE__ : $A$ is _neither_ symmetric _nor_ positive definite!

See `*4.png` for images of solution.

- $\tilde A^{-1} =$ `lu(A)`: converges in __15__ iterations 
- $\tilde A^{-1} =$ `ilu(A, τ=1e-3)`: converges in __72__ iterations

### PG Thin BL ($\varepsilon^2 = 10^{-4}$, $\gamma = 1$, $f = 1$)

__NOTE__ : $A$ is _neither_ symmetric _nor_ positive definite!

See `*5.png` for images of solution.

- $\tilde A^{-1} =$ `lu(A)`: "converges" in __1,568__ iterations 
    - rapid convergence in first 100 steps followed by very slow progress
    - not actually converged to true solution (see `*5a.png`)
- no blocks, just `1/h^2` normalization: __308,990__ iterations (steady slope)

---

Now try $P = \left(\begin{array}{c c} \tilde A & B^T\\0 & -\tilde S \end{array}\right)$ with $\tilde S = \tilde M_p/\varepsilon^{2}$ and the same scheme for $\tilde M_p^{-1}$ as above.

- Classic Stokes: __7__ iterations 
- Aspect Ratio Stokes: __11__ iterations 
- Less Diff Stokes: __10__ iterations 
-  PG Thick BL: __7__ iterations 
- PG Thin BL: __1,281__ iterations (same failure)

---

Same tridiagonal $P$ as above but with $\tilde S^{-1} = L^{-1} B T A T B^T L^{-1}$ where $T = ($ diag $M_u)^{-1}$ and $L = B T B^T$ (Least-Squares Commutator method).

- PG Thin BL: 
    - using `lu(A)` and `lu(L)`: about __2,000__ iterations, no failure! (see `*6.png`)
    - using `ilu(A, τ=1e-3)`, `lu(L)`: __5,966__ iterations

---

Tridiagonal $P$ with $\tilde S^{-1} = K_p^{-1}$ where $K_p$ is just the ($1/f$-weighted) pressure stiffness matrix.

- PG Thin BL:
    - using `lu(A)` and `lu(K_p)`: __1,819__ iterations, no failure!

Different forms of $\tilde A^{-1}$:
- $\tilde A^{-1} = \left(\begin{array}{c c c} 0 & -M^{-1} & 0 \\ M^{-1} & 0 & 0 \\ 0 & 0 & K^{-1} \end{array}\right)$

# Experiment Set 2 (January 2026)

- νPGCM v0.5.0
- 3D bowl
- $b$ from spin-up ($\mu\varrho = \varepsilon = 1$, $t = 0.1$)
- $h = 0.2\alpha$
- DoFs: 
    - $\alpha = 1$: 4201
    - $\alpha = 1/2$: 15946
    - $\alpha = 1/4$: 64597
    - $\alpha = 1/8$: 261736
- $f = 1 + y/2$

## $\alpha = 1$

### $\varepsilon = 1$ ($\delta/h = \varepsilon\sqrt{2}/0.2 \approx 7.07$)

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 8402       | 8.492e+00 (solved=false) |
| `BlockDiagonal(lu(A))`           | 41         | 1.131e-01 |
| `BlockDiagonal(lu(A_no_f))`      | 41         | 7.955e-02 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 41         | 5.196e+00 |

### $\varepsilon = 1/2$ ($\delta/h \approx 3.54$)

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 8189       | 4.005e+00 |
| `BlockDiagonal(lu(A))`           | 61         | 2.103e-01 |
| `BlockDiagonal(lu(A_no_f))`      | 61         | 1.668e-01 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 61         | 7.887e+00 |

### $\varepsilon = 1/4$ ($\delta/h \approx 1.77$)
 
| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 2043       | 1.024e+00 |
| `BlockDiagonal(lu(A))`           | 101        | 3.160e-01 |
| `BlockDiagonal(lu(A_no_f))`      | 521        | 1.034e+00 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 341        | 4.367e+01 |

## $\alpha = 1/2$

### $\varepsilon = 1$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 31892      | 1.607e+01 (solved=false) |
| `BlockDiagonal(lu(A))`           | 41         | 1.071e+01 |
| `BlockDiagonal(lu(A_no_f))`      | 41         | 8.790e+00 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 41         | 1.938e+01 |

### $\varepsilon = 1/2$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 11973      | 5.969e+00 |
| `BloackDiagonal(I/h^3)`          | 421        | 7.191e+00 |
| `BloackDiagonal(I/h^3)`, `itmax=20, 4` | 1561        | 4.201e+00 |
| `BloackDiagonal(I/h^3)`, `itmax=30, 4` | 1061        | 3.664e+00 |
| `BloackDiagonal(I/h^3)`, `itmax=40, 4` | 1181        | 4.921e+00 |
| `BlockDiagonal(lu(A))`           | 121        | 3.124e+01 |
| `BlockDiagonal(lu(A_no_f))`      | 281        | 5.648e+01 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 381        | 1.870e+02 |
| `BlockDiagonal(kp_ilu0(A_no_f))`, `itmax=10,4` | 561        | 1.795e+02 |
| `BlockDiagonal(kp_ilu0(Au, Av, Aw))` | 401        | 2.945+02 |
| `BlockDiagonal(kp_ilu0(Au, Av, Aw))`, `itmax=10,4` | 401        | 2.162e+02 |

### $\varepsilon = 1/4$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 3356       | 1.681e+00 |
| `BlockDiagonal(lu(A))`           | 1581       | 4.069e+02 |

## $\alpha = 1/4$

### $\varepsilon = 1$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 36265      | 2.139e+01 |
| `BlockDiagonal(lu(A))`           | 61         | 4.407e+01 |
| `BlockDiagonal(lu(A_no_f))`      | 61         | 4.097e+01 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 61         | 1.213e+02 |

### $\varepsilon = 1/2$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 17303      | 1.039e+01 |
| `BlockDiagonal(lu(A))`           | 61         | 4.398e+01 |
| `BlockDiagonal(lu(A_no_f))`      | 81         | 5.453e+01 |
| `BlockDiagonal(kp_ilu0(A_no_f))` | 81         | 1.491e+02 |

### $\varepsilon = 1/4$

| Preconditioner                   | iterations | time (s)  |
| -                                | -          | -         |
| `I/h^3`                          | 6794       | 3.985e+00 |
| `BlockDiagonal(lu(A))`           | 554        | 3.988e+02 |