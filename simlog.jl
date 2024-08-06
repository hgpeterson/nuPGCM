"""
################################################################################

sim000

Mesh:
    • 2D bowl (thin) h=0.01 
Parameters:
    • ε² = 1e-3
    • μϱ = 1
    • γ = 1
    • f = 1
Forcing:
    • ν = 1
    • κ = 1e-2 + exp(-(z + H)/0.1)
Horizontal Diffusion: FALSE
Timestep:
    • Diffusion → Crank-Nicolson
    • Advection → Forward Euler
    • Δt = 1e-4*μϱ/ε²
    • T = 500Δt
Iterative Solvers:
    • Inversion → DQGMRES, memory=20, tol=1e-8, P=I
    • Evolution → CG, tol=1e-8, P=I
Notes:
    • Good reference for 3D simulations with same setup

################################################################################

sim001

Same as sim000 but with no advection.
Notes:
    • Clear difference in `b` profiles by the end of the simulation 
        → Advection is doing something.

################################################################################

sim002

Mesh:
    • 3D bowl (thin) h=0.02 
Parameters:
    • ε² = 1
    • μϱ = 1
    • γ = 1
    • f = 1
Forcing:
    • ν = 1
    • κ = 1e-2 + exp(-(z + H)/0.1)
Horizontal Diffusion: FALSE
Timestep:
    • Diffusion → Crank-Nicolson
    • Advection → Forward Euler
    • Δt = 1e-4*μϱ/ε²
    • T = 500Δt
Iterative Solvers:
    • Inversion → DQGMRES, memory=20, tol=1e-7, P=I
    • Evolution → CG, tol=1e-7, P=I
Notes:
    • This was a test to see if a lower tolerance (1e-7) in the iterative 
      solvers would make a difference in the solution.
    • There is a little bit more noise in the beginning of the simulation, but 
      it ultimately smooths out.

################################################################################

sim003

Mesh:
    • 3D bowl (thin) h=0.01 
Parameters:
    • ε² = 1e-3
    • μϱ = 1
    • γ = 1
    • f = 1
Forcing:
    • ν = 1
    • κ = 1e-2 + exp(-(z + H)/0.1)
Horizontal Diffusion: FALSE
Timestep:
    • Diffusion → Crank-Nicolson
    • Advection → Forward Euler
    • Δt = 1e-4*μϱ/ε²
    • T = 500Δt
Iterative Solvers:
    • Inversion → DQGMRES, memory=20, tol=1e-7, P=I
    • Evolution → CG, tol=1e-7, P=I
Simulation Info:
    • Hardware = One 80G h100 GPU
    • Simulation time = 16 hrs
Notes:
    • Only profiles were saved, but it seems like a wave develops before the 
      solution crashes at about t = 25.

################################################################################

sim004

Same as sim003 but with horizontal diffusion.
Notes:
    • With γ = 1, the horizontal diffusion is strong enough to completely
      stabilize the solution (no waves).

################################################################################

sim005

Same as sim004 but with γ = 1/8.
Notes:
    • Due to a bug in the code, γ was 1/8 for the inversion and the RHS of the
      evolution, but for the LHS of the evolution it was still 1. This caused
      the solution to be just as smooth as sim004.

################################################################################

sim006

Redo sim005, this time with γ actually equal to 1/8.

################################################################################

sim007

Same as sim006 but without horizontal diffusion.

################################################################################
"""