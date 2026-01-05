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
    - $\alpha = 1$: 4,201
    - $\alpha = 1/2$: 15,946
    - $\alpha = 1/4$: 64,597
    - $\alpha = 1/8$: 261,736
- $f = 1 + y/2$

## $\alpha = 1$

### $\varepsilon = 1$ ($\delta/h = \varepsilon\sqrt{2}/0.2 \approx 7.07$)


- `P = I/h^3`: `niter=8402, time=8.492303261 (solved=false)` 
- `P = BlockDiagonal(lu(A))`: `niter=41, time=0.113108237`
- `P = BlockDiagonal(lu(A_no_f))`: `niter=41, time=7.955e-02`
- `P = BlockDiagonal(kp_ilu0(A_no_f))`: `niter=41, time=5.196e+00`

### $\varepsilon = 1/2$ ($\delta/h = \varepsilon\sqrt{2}/0.2 \approx 3.54$)

- `P = I/h^3`: `niter=8189, time=4.004770284999999`
- `P = BlockDiagonal(lu(A))`: `niter=61, time=0.210336316`
- `P = BlockDiagonal(lu(A_no_f))`: `niter=61, time=1.668e-01`
- `P = BlockDiagonal(kp_ilu0(A_no_f))`: `niter=61, time=7.887e+00`

### $\varepsilon = 1/4$ ($\delta/h \approx 1.77$)
 
- `P = I/h^3`: `niter=2043, time=1.023896494`
- `P = BlockDiagonal(lu(A))`: `niter=101, time=0.31602084399999997`
- `P = BlockDiagonal(lu(A_no_f))`: `niter=521, time=1.034e+00`
- `P = BlockDiagonal(kp_ilu0(A_no_f))`: `niter=341, time=4.367e+01`

## $\alpha = 1/2$

### $\varepsilon = 1$

- `P = I/h^3`: `niter=31892, time=16.070321899 (solved=false)`
- `P = BlockDiagonal(lu(A))`: `niter=41, time=10.705225942`
- `P = BlockDiagonal(kp_ilu0(A_no_f))`: `niter=41, time=1.938e+01`

### $\varepsilon = 1/2$

- `P = I/h^3`: `niter=11973, time=5.9686630780000005`
- `P = BlockDiagonal(lu(A))`: `niter=121, time=31.244824391`
- `P = BlockDiagonal(kp_ilu0(A_no_f))`: `niter=381, time=1.870e+02`

### $\varepsilon = 1/4$

- `P = I/h^3`: `niter=3356, time=1.6807564229999998`
- `P = BlockDiagonal(lu(A))`: `niter=1581, time=406.867088986`

## $\alpha = 1/4$

### $\varepsilon = 1$

- `P = I/h^3`: `niter=36265, time=21.393152276`

### $\varepsilon = 1/2$

- `P = I/h^3`: `niter=17303, time=10.389196127`

### $\varepsilon = 1/4$

- `P = I/h^3`: `niter=6794, time=3.9851888`