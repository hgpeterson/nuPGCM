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
# Experiment Set 3 — rotating-Stokes preconditioners (`ferrite`, 2026-07-31)

Channel-basin box, eddy viscosity on, `f(x) = x[2]` (equator inside the domain).
Reduced/condensed system `[A -Bᵀ; B 0]`, `dof_order = :blocked`, GMRES(20),
`atol = rtol = 1e-6`.

`S̃` legend — see `src/pressure_operators.jl` for the derivation:

| tag | Schur approximation `S̃` | correct when |
|-----|--------------------------|--------------|
| `mass` | `Mν = ∫pq/(2η)` | `η\|k\|² ≫ f` (viscous) |
| `stiffness` | `Kf = ∫(1/\|f\|)∇p·∇q` | — (the Exp. Set 1 workaround) |
| `cahouet-chabard` | `Mν⁻¹ + Kf⁻¹` | interpolates the two limits |
| `geo-δ` | `M⁻¹K(Kz/η + δKh/η)⁻¹` | `η\|k\|² ≪ f` (rotating) |
| `geo-full` | same with the untruncated symbol | both limits, no free parameter |
| `lsc` | `(BTBᵀ)⁻¹(BTATBᵀ)(BTBᵀ)⁻¹` | commutator-free |
| `al` | `γM⁻¹ + S₀⁻¹` | any (1,1) block, given `γ` large |
| `EXACT` | dense `BA⁻¹Bᵀ` | always (reference only) |


## `A-h1e-1-a0.5-eps0.25`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 2.439e-02 (×0.25), η = α²ε²ν = 1.487e-04

δ_ekman/h = **0.15**, viscous/rotating crossover at |k| = 82.0 (0.7 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | — | — | 0.000 | 14.199 | 4.733 | — | 5.18e-02 | 6.35e-03 | 3.22e-01 |
| `bdiag-mass` | 3000 | **no** | — | — | 1.542 | 128.267 | 42.756 | — | 1.62e-01 | 4.10e-02 | 3.23e-01 |
| `btri-mass` | 3000 | **no** | — | — | 1.006 | 128.814 | 42.938 | — | 8.07e-01 | 2.80e-02 | 2.37e-01 |
| `btri-stiffness` | 3000 | **no** | — | — | 0.851 | 130.356 | 43.452 | — | 9.96e-01 | 9.96e-01 | 9.91e-01 |
| `btri-cahouet-chabard` | 3000 | **no** | — | — | 1.363 | 131.791 | 43.930 | — | 9.92e-01 | 9.92e-01 | 9.83e-01 |
| `btri-geo-0.9` | 3000 | **no** | — | — | 1.417 | 132.596 | 44.199 | — | 8.63e-01 | 5.55e-02 | 3.94e-01 |
| `btri-geo-0.5` | 3000 | **no** | — | — | 1.113 | 130.970 | 43.657 | — | 9.02e-01 | 8.90e-01 | 7.52e-01 |
| `btri-geo-1e-1` | 3000 | **no** | — | — | 1.350 | 131.256 | 43.752 | — | 9.83e-01 | 9.83e-01 | 9.66e-01 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 1.313 | 130.875 | 43.625 | — | 9.99e-01 | 9.99e-01 | 9.98e-01 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 1.287 | 130.906 | 43.635 | — | 1.00e+00 | 1.00e+00 | 9.99e-01 |
| `btri-geo-full` | 3000 | **no** | — | — | 1.668 | 134.869 | 44.956 | — | 1.00e+00 | 1.00e+00 | 9.99e-01 |
| `btri-lsc` | 3000 | **no** | — | — | 1.477 | 132.640 | 44.213 | — | 1.00e+00 | 1.00e+00 | 1.00e+00 |
| `btri-EXACT-schur` | 2 | yes | 3 | 3 | 24.808 | 0.421 | 210.412 | — | 1.00e+00 | 6.12e-12 | 1.02e-12 |

## `A-h1e-1-a0.5-eps1`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 9.757e-02 (×1), η = α²ε²ν = 2.380e-03

δ_ekman/h = **0.61**, viscous/rotating crossover at |k| = 20.5 (2.7 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | 2152 | — | 0.000 | 17.243 | 5.748 | — | 2.33e-02 | 5.01e-04 | 7.33e-02 |
| `bdiag-mass` | 3000 | **no** | — | — | 1.192 | 152.529 | 50.843 | — | 1.20e-01 | 4.79e-03 | 8.88e-02 |
| `btri-mass` | 3000 | **no** | 1560 | — | 1.280 | 198.791 | 66.264 | — | 2.08e-01 | 2.62e-05 | 4.73e-04 |
| `btri-stiffness` | 3000 | **no** | — | — | 2.149 | 169.934 | 56.645 | — | 9.17e-01 | 8.46e-01 | 7.24e-01 |
| `btri-cahouet-chabard` | 213 | yes | 100 | — | 1.456 | 11.676 | 54.817 | — | 1.78e-01 | 3.25e-06 | 4.86e-06 |
| `btri-geo-0.9` | 3000 | **no** | 2513 | — | 1.053 | 159.226 | 53.075 | — | 1.26e-01 | 3.88e-04 | 5.62e-03 |
| `btri-geo-0.5` | 3000 | **no** | 2714 | — | 1.328 | 142.384 | 47.461 | — | 3.35e-01 | 5.85e-04 | 1.71e-03 |
| `btri-geo-1e-1` | 3000 | **no** | — | — | 1.673 | 193.463 | 64.488 | — | 8.55e-01 | 8.54e-01 | 8.54e-01 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 0.970 | 173.042 | 57.681 | — | 9.87e-01 | 9.87e-01 | 9.88e-01 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 2.653 | 171.064 | 57.021 | — | 9.98e-01 | 9.96e-01 | 9.96e-01 |
| `btri-geo-full` | 3000 | **no** | — | — | 2.576 | 156.488 | 52.163 | — | 9.97e-01 | 9.77e-01 | 9.73e-01 |
| `btri-lsc` | 683 | yes | 376 | — | 1.778 | 31.790 | 46.545 | — | 4.28e-01 | 3.37e-06 | 1.81e-05 |
| `btri-EXACT-schur` | 2 | yes | 3 | 3 | 33.111 | 0.361 | 180.288 | — | 1.00e+00 | 3.85e-13 | 2.64e-13 |

## `A-h1e-1-a0.5-eps4`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 3.903e-01 (×4), η = α²ε²ν = 3.808e-02

δ_ekman/h = **2.44**, viscous/rotating crossover at |k| = 5.1 (10.8 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | 2343 | — | 0.000 | 15.686 | 5.229 | — | 2.10e-02 | 5.11e-04 | 7.19e-02 |
| `bdiag-mass` | 1460 | yes | 437 | — | 1.804 | 67.625 | 46.318 | — | 3.65e-02 | 3.38e-06 | 1.09e-04 |
| `btri-mass` | 80 | yes | 39 | — | 1.823 | 2.598 | 32.478 | — | 1.37e-03 | 2.71e-06 | 2.68e-05 |
| `btri-stiffness` | 855 | yes | 223 | — | 1.600 | 40.760 | 47.673 | — | 1.62e-01 | 3.38e-06 | 2.51e-04 |
| `btri-cahouet-chabard` | 43 | yes | 21 | — | 1.640 | 2.371 | 55.138 | — | 2.57e-03 | 3.03e-06 | 3.31e-05 |
| `btri-geo-0.9` | 85 | yes | 45 | — | 1.642 | 4.363 | 51.328 | — | 2.96e-02 | 2.79e-06 | 3.00e-05 |
| `btri-geo-0.5` | 174 | yes | 54 | — | 1.235 | 7.746 | 44.516 | — | 7.52e-02 | 3.39e-06 | 2.39e-05 |
| `btri-geo-1e-1` | 798 | yes | 481 | — | 1.586 | 37.893 | 47.485 | — | 2.12e-01 | 3.28e-06 | 9.03e-06 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 1.590 | 141.242 | 47.081 | — | 6.41e-01 | 1.20e-03 | 2.40e-03 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 1.607 | 142.426 | 47.475 | — | 8.99e-01 | 6.18e-02 | 1.18e-01 |
| `btri-geo-full` | 3000 | **no** | — | — | 1.215 | 141.359 | 47.120 | — | 1.64e-01 | 4.18e-03 | 8.18e-03 |
| `btri-lsc` | 35 | yes | 22 | — | 1.132 | 1.134 | 32.406 | — | 4.70e-05 | 3.06e-06 | 6.53e-06 |
| `btri-EXACT-schur` | 2 | yes | 3 | 3 | 27.000 | 0.437 | 218.634 | — | 9.98e-01 | 2.69e-14 | 2.19e-14 |

## `AL-h1e-1-g1e2`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 10008272

h = 1.133e-01, α = 0.500, ε = 9.757e-02 (×1), η = α²ε²ν = 2.380e-03

δ_ekman/h = **0.61**, viscous/rotating crossover at |k| = 20.5 (2.7 cells), gauge = `pin`, **γ_augment = 100**

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `al-γ100.0-mass` | 3 | yes | 3 | 4 | 12.773 | 0.641 | 213.540 | — | 7.97e-05 | 9.88e-08 | 2.80e-05 |
| `al-γ100.0-geo` | 3 | yes | 3 | 4 | 11.039 | 0.962 | 320.773 | — | 7.97e-05 | 9.74e-08 | 2.77e-05 |
| `al-γ100.0-ilut-mass` | 3000 | **no** | — | — | 2391.224 | 1416.171 | 472.057 | — | 8.88e-01 | 3.57e-01 | 9.61e-01 |

## `AL-h1e-1-g1e4`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 10008272

h = 1.133e-01, α = 0.500, ε = 9.757e-02 (×1), η = α²ε²ν = 2.380e-03

δ_ekman/h = **0.61**, viscous/rotating crossover at |k| = 20.5 (2.7 cells), gauge = `pin`, **γ_augment = 1e+04**

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `al-γ10000.0-mass` | 2 | yes | 2 | 3 | 8.303 | 0.530 | 265.007 | — | 3.59e-05 | 8.37e-09 | 1.60e-04 |
| `al-γ10000.0-geo` | 2 | yes | 2 | 3 | 7.894 | 0.759 | 379.469 | — | 3.59e-05 | 8.36e-09 | 1.60e-04 |
| `al-γ10000.0-ilut-mass` | 3000 | **no** | — | — | 2386.825 | 1255.708 | 418.569 | — | 7.71e-01 | 3.06e-01 | 9.24e-01 |

## `B-h4e-2-a0.25-cpu`

CPU, n = 147924 (nu = 139744, np = 8180, 5.5% pressure), nnz = 12820218

h = 4.865e-02, α = 0.250, ε = 9.757e-02 (×1), η = α²ε²ν = 5.950e-04

δ_ekman/h = **0.71**, viscous/rotating crossover at |k| = 41.0 (3.2 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 2000 | **no** | — | — | 0.000 | 180.494 | 90.247 | — | 1.60e-01 | 9.12e-03 | 6.40e-01 |
| `btri-ilut-mass` | 2000 | **no** | — | — | 0.279 | 190.885 | 95.443 | — | 9.67e-01 | 9.67e-01 | 1.03e+00 |
| `btri-ilut-cc` | 659 | yes | 293 | — | 0.519 | 65.058 | 98.722 | — | 8.25e-01 | 8.85e-06 | 4.47e-04 |
| `btri-ilut-geo-0.5` | 2000 | **no** | — | — | 0.267 | 200.455 | 100.227 | — | 9.76e-01 | 9.76e-01 | 1.05e+00 |
| `btri-ilut-geo-1e-1` | 2000 | **no** | — | — | 0.366 | 219.927 | 109.963 | — | 9.84e-01 | 9.83e-01 | 1.03e+00 |
| `btri-ilut-geo-1e-2` | 2000 | **no** | — | — | 0.420 | 206.522 | 103.261 | — | 9.87e-01 | 9.87e-01 | 1.02e+00 |
| `btri-ilut-lsc` | 2000 | **no** | — | — | 1.403 | 223.463 | 111.731 | — | 9.97e-01 | 9.97e-01 | 9.99e-01 |
| `btri-krylov5-geo-1e-3` | 2000 | **no** | — | — | 0.325 | 1110.218 | 555.109 | — | 1.00e+00 | 9.90e-01 | 9.94e-01 |
| `btri-krylov20-geo-1e-3` | 2000 | **no** | — | — | 0.343 | 3247.376 | 1623.688 | — | 1.00e+00 | 9.54e-01 | 9.56e-01 |
| `btri-lu-geo-1e-1` | 2000 | **no** | — | — | 29.029 | 798.943 | 399.472 | — | 1.00e+00 | 9.51e-01 | 9.46e-01 |

## `B-h4e-2-a0.25-gpu`

GPU, n = 147924 (nu = 139744, np = 8180, 5.5% pressure), nnz = 12820218

h = 4.865e-02, α = 0.250, ε = 9.757e-02 (×1), η = α²ε²ν = 5.950e-04

δ_ekman/h = **0.71**, viscous/rotating crossover at |k| = 41.0 (3.2 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 2000 | **no** | — | — | 0.000 | 2.762 | 1.381 | — | 2.30e-02 | 9.12e-03 | 6.40e-01 |
| `btri-ilu0-mass` | 2000 | **no** | — | — | 0.933 | 63.575 | 31.788 | — | 8.09e-01 | 3.90e-02 | 4.81e-01 |
| `btri-ilu0-cc` | 268 | yes | 128 | — | 0.836 | 9.359 | 34.922 | — | 2.77e-01 | 8.82e-06 | 1.07e-04 |
| `btri-ilu0-geo-0.5` | 2000 | **no** | — | — | 1.095 | 67.838 | 33.919 | — | 9.71e-01 | 9.71e-01 | 9.69e-01 |
| `btri-ilu0-geo-1e-1` | 2000 | **no** | — | — | 0.642 | 67.603 | 33.801 | — | 9.88e-01 | 9.88e-01 | 9.86e-01 |
| `btri-ilu0-geo-1e-2` | 2000 | **no** | — | — | 0.638 | 67.716 | 33.858 | — | 9.99e-01 | 9.99e-01 | 9.99e-01 |
| `btri-ilu0-lsc` | 2000 | **no** | — | — | 1.793 | 85.103 | 42.551 | — | 9.98e-01 | 9.98e-01 | 9.93e-01 |
| `btri-krylov5-geo-1e-3` | 2000 | **no** | — | — | 0.526 | 378.854 | 189.427 | — | 1.00e+00 | 9.59e-01 | 9.57e-01 |
| `btri-krylov20-geo-1e-3` | 2000 | **no** | — | — | 1.064 | 1330.255 | 665.127 | — | 1.00e+00 | 9.51e-01 | 9.46e-01 |
| `btri-lu-geo-1e-1` | 2000 | **no** | — | — | 50.546 | 814.815 | 407.407 | — | 1.00e+00 | 9.51e-01 | 9.46e-01 |

## `C-h1e-1-a0.25-gpu`

GPU, n = 9279 (nu = 8499, np = 780, 8.4% pressure), nnz = 626113

h = 1.000e-01, α = 0.250, ε = 9.757e-02 (×1), η = α²ε²ν = 5.950e-04

δ_ekman/h = **0.34**, viscous/rotating crossover at |k| = 41.0 (1.5 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 2000 | **no** | — | — | 0.000 | 1.891 | 0.946 | — | 9.39e-03 | 2.90e-03 | 1.35e-01 |
| `btri-ilu0-mass` | 2000 | **no** | — | — | 0.637 | 20.923 | 10.461 | — | 2.52e-01 | 1.01e-03 | 1.26e-02 |
| `btri-ilu0-cc` | 311 | yes | 123 | — | 0.140 | 3.532 | 11.356 | — | 4.66e-03 | 2.79e-06 | 2.34e-06 |
| `btri-ilu0-geo-0.5` | 2000 | **no** | — | — | 0.205 | 21.142 | 10.571 | — | 9.30e-02 | 8.05e-03 | 2.94e-02 |
| `btri-ilu0-geo-1e-1` | 2000 | **no** | — | — | 0.188 | 20.949 | 10.475 | — | 8.35e-01 | 1.17e-02 | 6.25e-03 |
| `btri-ilu0-geo-1e-2` | 2000 | **no** | — | — | 0.029 | 21.010 | 10.505 | — | 9.94e-01 | 9.94e-01 | 9.90e-01 |
| `btri-ilu0-lsc` | 1255 | yes | 618 | — | 0.374 | 13.871 | 11.052 | — | 3.37e-01 | 2.83e-06 | 4.18e-06 |
| `btri-krylov5-geo-1e-3` | 2000 | **no** | — | — | 0.421 | 120.765 | 60.382 | — | 8.36e-01 | 5.26e-02 | 3.44e-02 |
| `btri-krylov20-geo-1e-3` | 2000 | **no** | — | — | 0.057 | 429.686 | 214.843 | — | 1.00e+00 | 4.40e-02 | 3.09e-02 |
| `btri-lu-geo-1e-1` | 2000 | **no** | — | — | 1.060 | 9.286 | 4.643 | — | 5.03e-02 | 4.40e-02 | 3.09e-02 |

## `C-h2e-2-a0.25-gpu`

GPU, n = 1135750 (nu = 1082483, np = 53267, 4.7% pressure), nnz = 105782394

h = 2.613e-02, α = 0.250, ε = 9.757e-02 (×1), η = α²ε²ν = 5.950e-04

δ_ekman/h = **1.32**, viscous/rotating crossover at |k| = 41.0 (5.9 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 2000 | **no** | — | — | 0.000 | 10.249 | 5.125 | — | 9.89e-02 | 6.51e-03 | — |
| `btri-ilu0-mass` | 2000 | **no** | — | — | 2.708 | 172.198 | 86.099 | — | 9.94e-01 | 9.17e-01 | — |
| `btri-ilu0-cc` | 235 | yes | 113 | — | 3.853 | 47.499 | 202.124 | — | 9.99e-01 | 1.84e-05 | — |
| `btri-ilu0-geo-0.5` | 2000 | **no** | — | — | 3.986 | 415.650 | 207.825 | — | 9.99e-01 | 9.86e-01 | — |
| `btri-ilu0-geo-1e-1` | 2000 | **no** | — | — | 4.867 | 401.135 | 200.567 | — | 1.00e+00 | 9.92e-01 | — |
| `btri-ilu0-geo-1e-2` | 2000 | **no** | — | — | 4.122 | 399.710 | 199.855 | — | 1.00e+00 | 9.99e-01 | — |
| `btri-ilu0-lsc` | 2000 | **no** | — | — | 17.837 | 943.095 | 471.547 | — | 1.00e+00 | 9.99e-01 | — |
| `btri-krylov5-geo-1e-3` | 2000 | **no** | — | — | 4.880 | 1151.936 | 575.968 | — | 1.00e+00 | 9.80e-01 | — |
| `btri-krylov20-geo-1e-3` | 2000 | **no** | — | — | 4.335 | 3699.143 | 1849.572 | — | 1.00e+00 | 9.49e-01 | — |

## `C-h4e-2-a0.125-gpu`

GPU, n = 82601 (nu = 77269, np = 5332, 6.5% pressure), nnz = 6467877

h = 4.394e-02, α = 0.125, ε = 9.757e-02 (×1), η = α²ε²ν = 1.487e-04

δ_ekman/h = **0.39**, viscous/rotating crossover at |k| = 82.0 (1.7 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 2000 | **no** | — | — | 0.000 | 2.081 | 1.040 | — | 1.68e-02 | 6.37e-03 | 2.45e-01 |
| `btri-ilu0-mass` | 2000 | **no** | — | — | 0.706 | 39.722 | 19.861 | — | 6.63e-01 | 1.15e-02 | 1.31e-01 |
| `btri-ilu0-cc` | 474 | yes | 186 | — | 0.292 | 10.024 | 21.147 | — | 6.12e-01 | 6.14e-06 | 2.92e-06 |
| `btri-ilu0-geo-0.5` | 2000 | **no** | — | — | 0.292 | 41.786 | 20.893 | — | 9.48e-01 | 9.48e-01 | 8.92e-01 |
| `btri-ilu0-geo-1e-1` | 2000 | **no** | — | — | 0.270 | 41.322 | 20.661 | — | 9.72e-01 | 9.72e-01 | 9.45e-01 |
| `btri-ilu0-geo-1e-2` | 2000 | **no** | — | — | 0.427 | 41.342 | 20.671 | — | 9.98e-01 | 9.98e-01 | 9.97e-01 |
| `btri-ilu0-lsc` | 2000 | **no** | — | — | 0.906 | 46.394 | 23.197 | — | 8.92e-01 | 8.86e-01 | 7.54e-01 |
| `btri-krylov5-geo-1e-3` | 2000 | **no** | — | — | 0.163 | 235.465 | 117.732 | — | 9.99e-01 | 9.48e-01 | 8.97e-01 |
| `btri-krylov20-geo-1e-3` | 2000 | **no** | — | — | 0.521 | 826.831 | 413.416 | — | 1.00e+00 | 9.41e-01 | 8.78e-01 |
| `btri-lu-geo-1e-1` | 2000 | **no** | — | — | 13.318 | 208.252 | 104.126 | — | 9.98e-01 | 9.41e-01 | 8.78e-01 |

## `D-h1e-1-a0.5-eps0.125`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 1.220e-02 (×0.125), η = α²ε²ν = 3.718e-05

δ_ekman/h = **0.08**, viscous/rotating crossover at |k| = 164.0 (0.3 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | — | — | 0.000 | 22.893 | 7.631 | — | 4.02e-02 | 1.98e-02 | 7.38e-01 |
| `bdiag-mass` | 3000 | **no** | — | — | 1.187 | 164.254 | 54.751 | — | 1.61e-01 | 1.58e-01 | 9.75e-01 |
| `btri-mass` | 3000 | **no** | — | — | 0.846 | 153.189 | 51.063 | — | 9.59e-01 | 9.59e-01 | 9.14e-01 |
| `btri-stiffness` | 3000 | **no** | — | — | 1.040 | 185.529 | 61.843 | — | 1.00e+00 | 1.00e+00 | 9.99e-01 |
| `btri-cahouet-chabard` | 3000 | **no** | — | — | 1.172 | 158.619 | 52.873 | — | 1.00e+00 | 1.00e+00 | 9.99e-01 |
| `btri-geo-0.9` | 3000 | **no** | — | — | 4.890 | 166.543 | 55.514 | — | 9.78e-01 | 9.76e-01 | 9.51e-01 |
| `btri-geo-0.5` | 3000 | **no** | — | — | 2.103 | 259.036 | 86.345 | — | 1.00e+00 | 9.74e-01 | 9.49e-01 |
| `btri-geo-1e-1` | 3000 | **no** | — | — | 1.100 | 169.107 | 56.369 | — | 9.96e-01 | 9.96e-01 | 9.92e-01 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 2.165 | 169.067 | 56.356 | — | 1.00e+00 | 1.00e+00 | 9.99e-01 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 0.763 | 170.003 | 56.668 | — | 1.00e+00 | 1.00e+00 | 1.00e+00 |
| `btri-geo-full` | 3000 | **no** | — | — | 1.152 | 162.488 | 54.163 | — | 1.00e+00 | 1.00e+00 | 1.00e+00 |
| `btri-lsc` | 3000 | **no** | — | — | 1.210 | 179.914 | 59.971 | — | 1.00e+00 | 1.00e+00 | 1.00e+00 |

## `D-h1e-1-a0.5-eps0.5`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 4.878e-02 (×0.5), η = α²ε²ν = 5.950e-04

δ_ekman/h = **0.30**, viscous/rotating crossover at |k| = 41.0 (1.4 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | 2957 | — | 0.000 | 18.939 | 6.313 | — | 4.60e-02 | 9.72e-04 | 1.16e-01 |
| `bdiag-mass` | 3000 | **no** | — | — | 2.354 | 110.473 | 36.824 | — | 1.45e-01 | 2.15e-02 | 3.81e-01 |
| `btri-mass` | 3000 | **no** | — | — | 1.177 | 112.614 | 37.538 | — | 4.14e-01 | 8.21e-03 | 1.59e-01 |
| `btri-stiffness` | 3000 | **no** | — | — | 1.255 | 118.262 | 39.421 | — | 9.57e-01 | 9.57e-01 | 9.19e-01 |
| `btri-cahouet-chabard` | 615 | yes | 290 | — | 1.533 | 26.845 | 43.651 | — | 7.55e-01 | 3.35e-06 | 2.65e-05 |
| `btri-geo-0.9` | 3000 | **no** | — | — | 1.386 | 90.484 | 30.161 | — | 2.97e-01 | 4.64e-02 | 6.23e-01 |
| `btri-geo-0.5` | 3000 | **no** | — | — | 1.755 | 122.269 | 40.756 | — | 4.93e-01 | 5.02e-02 | 2.08e-01 |
| `btri-geo-1e-1` | 3000 | **no** | — | — | 1.236 | 132.411 | 44.137 | — | 9.49e-01 | 9.49e-01 | 9.19e-01 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 1.187 | 108.562 | 36.187 | — | 9.97e-01 | 9.97e-01 | 9.95e-01 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 1.876 | 131.556 | 43.852 | — | 9.99e-01 | 9.99e-01 | 9.99e-01 |
| `btri-geo-full` | 3000 | **no** | — | — | 1.511 | 213.545 | 71.182 | — | 9.98e-01 | 9.98e-01 | 9.97e-01 |
| `btri-lsc` | 3000 | **no** | — | — | 2.458 | 253.158 | 84.386 | — | 9.96e-01 | 9.88e-01 | 9.74e-01 |

## `D-h1e-1-a0.5-eps2`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 1.951e-01 (×2), η = α²ε²ν = 9.519e-03

δ_ekman/h = **1.22**, viscous/rotating crossover at |k| = 10.2 (5.4 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-1/h3` | 3000 | **no** | 2143 | — | 0.000 | 14.775 | 4.925 | — | 2.02e-02 | 3.83e-04 | 3.20e-02 |
| `bdiag-mass` | 3000 | **no** | 1199 | — | 1.854 | 146.472 | 48.824 | — | 9.23e-02 | 9.77e-06 | 1.49e-04 |
| `btri-mass` | 582 | yes | 294 | — | 1.224 | 27.947 | 48.019 | — | 3.57e-02 | 3.31e-06 | 3.58e-05 |
| `btri-stiffness` | 1055 | yes | 398 | — | 1.071 | 53.020 | 50.256 | — | 5.47e-01 | 3.36e-06 | 1.09e-04 |
| `btri-cahouet-chabard` | 79 | yes | 39 | — | 1.551 | 4.292 | 54.330 | — | 3.33e-02 | 2.97e-06 | 5.93e-06 |
| `btri-geo-0.9` | 1057 | yes | 484 | — | 1.590 | 52.901 | 50.049 | — | 9.77e-02 | 3.22e-06 | 2.21e-05 |
| `btri-geo-0.5` | 1320 | yes | 661 | — | 1.091 | 65.130 | 49.341 | — | 2.33e-01 | 3.26e-06 | 9.55e-06 |
| `btri-geo-1e-1` | 3000 | **no** | — | — | 0.900 | 148.922 | 49.641 | — | 6.07e-01 | 6.04e-01 | 6.73e-01 |
| `btri-geo-1e-2` | 3000 | **no** | — | — | 0.865 | 149.223 | 49.741 | — | 9.46e-01 | 9.03e-01 | 9.19e-01 |
| `btri-geo-1e-3` | 3000 | **no** | — | — | 1.546 | 148.917 | 49.639 | — | 9.79e-01 | 9.72e-01 | 9.78e-01 |
| `btri-geo-full` | 3000 | **no** | — | — | 1.840 | 153.026 | 51.009 | — | 7.84e-01 | 7.49e-01 | 8.15e-01 |
| `btri-lsc` | 112 | yes | 65 | — | 1.220 | 4.668 | 41.676 | — | 1.84e-02 | 2.96e-06 | 8.34e-06 |

## `G-h1e-1-a0.5-none`

CPU, n = 18419 (nu = 17184, np = 1235, 6.7% pressure), nnz = 1461715

h = 1.133e-01, α = 0.500, ε = 9.757e-02 (×1), η = α²ε²ν = 2.380e-03

δ_ekman/h = **0.61**, viscous/rotating crossover at |k| = 20.5 (2.7 cells), gauge = `none`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `btri-geo-project` | 354 | yes | 179 | — | 0.880 | 13.099 | 37.003 | — | 1.99e-01 | 3.29e-06 | 9.69e-01 |

## `G-h1e-1-a0.5-pin`

CPU, n = 18418 (nu = 17184, np = 1234, 6.7% pressure), nnz = 1461168

h = 1.133e-01, α = 0.500, ε = 9.757e-02 (×1), η = α²ε²ν = 2.380e-03

δ_ekman/h = **0.61**, viscous/rotating crossover at |k| = 20.5 (2.7 cells), gauge = `pin`

| preconditioner | iters | solved | iters@1e-3 | iters@1e-6 | setup (s) | solve (s) | ms/iter | t@1e-6 (s) | res@1s | true res | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `btri-geo-pin` | 3000 | **no** | — | — | 0.923 | 243.255 | 81.085 | — | 8.54e-01 | 8.54e-01 | 8.54e-01 |

## Mesh-robustness summary

Iterations to `1e-6` for each preconditioner across meshes. Growth with `n` is the
thing to watch: a mesh-independent preconditioner holds a flat row.

| preconditioner | B-h4e-2-a0.25-cpu | B-h4e-2-a0.25-gpu | C-h1e-1-a0.25-gpu | C-h2e-2-a0.25-gpu | C-h4e-2-a0.125-gpu |
|---|---|---|---|---|---|
| `baseline-1/h3` | (>2000) | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-ilut-mass` | (>2000) | — | — | — | — |
| `btri-ilut-cc` | (>659) | — | — | — | — |
| `btri-ilut-geo-0.5` | (>2000) | — | — | — | — |
| `btri-ilut-geo-1e-1` | (>2000) | — | — | — | — |
| `btri-ilut-geo-1e-2` | (>2000) | — | — | — | — |
| `btri-ilut-lsc` | (>2000) | — | — | — | — |
| `btri-krylov5-geo-1e-3` | (>2000) | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-krylov20-geo-1e-3` | (>2000) | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-lu-geo-1e-1` | (>2000) | (>2000) | (>2000) | — | (>2000) |
| `btri-ilu0-mass` | — | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-ilu0-cc` | — | (>268) | (>311) | (>235) | (>474) |
| `btri-ilu0-geo-0.5` | — | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-ilu0-geo-1e-1` | — | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-ilu0-geo-1e-2` | — | (>2000) | (>2000) | (>2000) | (>2000) |
| `btri-ilu0-lsc` | — | (>2000) | (>1255) | (>2000) | (>2000) |

## Schur-complement spectra (`verify_schur.jl`)

Eigenvalue spread `max|λ|/min|λ|` of `S̃⁻¹S` with the **exact** dense
`S = BA⁻¹Bᵀ`. This is the quantity that governs GMRES on the pressure block,
measured independently of any solver. `a_grid/f = η(π/h)²/f < 1` means the
whole resolved spectrum is rotation-dominated.

`spearman` is the rank correlation between `|λ|` of `Mν⁻¹S` and the vertical
fraction `ρ_z = (pᵀKz p)/(pᵀK p)` of the corresponding eigenvector. A strong
positive value confirms the diagnosis: the modes the mass matrix misprices are
the barotropic ones.

| ε_scale | η | a_grid/f | mass | stiffness | geo δ=1e-2 | geo δ=1e-3 | geo full | spearman |
|---|---|---|---|---|---|---|---|---|
| 4 | 3.81e-02 | 29.29 | 1.58e+03 | 2.01e+04 | 2.39e+04 | 9.72e+04 | 3.15e+07 | -0.243 |
| 2 | 9.52e-03 | 7.32 | 2.17e+03 | 1.97e+04 | 1.97e+04 | 5.86e+04 | 4.30e+06 | -0.021 |
| 1 | 2.38e-03 | 1.83 | 6.84e+03 | 1.96e+04 | 1.84e+04 | 4.03e+04 | 2.39e+05 | +0.189 |
| 0.5 | 5.95e-04 | 0.46 | 3.30e+04 | 1.96e+04 | 2.30e+04 | 3.23e+04 | 2.58e+04 | +0.572 |
| 0.25 | 1.49e-04 | 0.11 | 1.21e+05 | 1.97e+04 | 3.10e+04 | 3.13e+04 | 4.44e+04 | +0.908 |

## Pressure gauge: `:pin` vs `:none` + nullspace projection

`:pin` eliminates one pressure DOF. That removes the exact constant nullmode but
leaves a near-null "constant everywhere except the pinned DOF" mode carrying a
spuriously tiny Schur eigenvalue. Schur approximations built from *singular*
pressure operators (`𝒦 = Kz + δKh`, `Kf`, `BTBᵀ`) invert that mode with enormous
gain, so they are exactly the ones `:pin` penalises — while the mass matrix,
nonsingular and well scaled, barely notices. A direct solve never sees any of it.


### `gauge_h1e-01_a0.50_cpu.jld2` — h = 0.1, α = 0.5, CPU

| Schur `S̃` | `:pin` iters | `:pin` err | `:none`+proj iters | `:none`+proj err | iteration ratio |
|---|---|---|---|---|---|
| `mass` | **>3000** | 4.73e-04 | 511 | — | 5.87× |
| `cahouet-chabard` | 213 | 4.86e-06 | 213 | — | 1.00× |
| `geo-0.5` | **>3000** | 1.71e-03 | 469 | — | 6.40× |
| `geo-1e-1` | **>3000** | 8.54e-01 | 354 | — | 8.47× |
| `geo-1e-2` | **>3000** | 9.88e-01 | **>3000** | — | 1.00× |
| `lsc` | 683 | 1.81e-05 | 693 | — | 0.99× |

### `gauge_h4e-02_a0.25_gpu.jld2` — h = 0.04, α = 0.25, GPU

| Schur `S̃` | `:pin` iters | `:pin` err | `:none`+proj iters | `:none`+proj err | iteration ratio |
|---|---|---|---|---|---|
| `mass` | **>2000** | 7.55e-01 | **>2000** | — | 1.00× |
| `cahouet-chabard` | 188 | 4.15e-05 | 188 | — | 1.00× |
| `geo-0.5` | **>2000** | 9.73e-01 | **>2000** | — | 1.00× |
| `geo-1e-1` | **>2000** | 9.83e-01 | **>2000** | — | 1.00× |
| `geo-1e-2` | **>2000** | 9.99e-01 | **>2000** | — | 1.00× |
| `lsc` | **>2000** | 9.47e-01 | **>2000** | — | 1.00× |

## Preconditioner staleness (`staleness.jl`)

Preconditioner built once from the state at step 2000, then reused for the
systems assembled from later states. `‖ΔA‖/‖A‖` is how far the operator has
actually drifted. Setup cost being amortized:

- `baseline-1/h3`: 0.000 s
- `btri-cc`: 1.264 s

| step | Δsteps | ‖ΔA‖/‖A‖ | baseline-1/h3 | btri-cc |
|---|---|---|---|---|
| 2000 | 0 | 0.000e+00 | 2000 (6.85s) | 266 (10.89s) |
| 2010 | 10 | 9.457e-03 | 2000 (2.22s) | 266 (8.72s) |
| 2020 | 20 | 1.895e-02 | 2000 (2.20s) | 266 (8.76s) |
| 2050 | 50 | 5.046e-02 | 2000 (2.17s) | 283 (9.29s) |
| 2100 | 100 | 8.828e-02 | 2000 (2.15s) | 306 (10.01s) |
| 2150 | 150 | 1.112e-01 | 2000 (2.15s) | 489 (16.00s) |
| 2200 | 200 | 1.248e-01 | 2000 (2.16s) | 455 (14.89s) |
| 2260 | 260 | 1.408e-01 | 2000 (2.15s) | 565 (18.48s) |
| 2320 | 320 | 1.501e-01 | 2000 (2.16s) | 602 (19.71s) |
