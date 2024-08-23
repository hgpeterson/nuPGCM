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
    • Inversion → DQGMRES, memory=20, tol=1e-8, P=I
    • Evolution → CG, tol=1e-8, P=I
Simulation Info:
    • Hardware = One 80G h100 GPU
    • Simulation time = 16 hrs
Notes:
    • Only profiles were saved, but it seems like a wave develops before the 
      solution crashes at about t = 25.
    • See sim009: this was probably just a CFL blow up and not a real 
      instability.

################################################################################

sim004

Same as sim003 but with horizontal diffusion and tol=1e-7.
Notes:
    • Solution is stable (not sure if it's because of the diffusion or the 
      lower tolerance - based on sim007, it's the tol). 

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
Notes:
    • Solution is stable and qualitatively similar to sim007.

################################################################################

sim007

Same as sim006 but without horizontal diffusion.
Notes:
    • Solution is stable even without horizontal diffusion. Must have been the
      lower tolerance that made the difference in sim004.

################################################################################

sim008

Redo sim003 to see exactly how instability forms.
Notes:
    • No instability! Only difference is tol=1e-8 in sim003.

################################################################################

sim009

Same as sim008 but with tol=1e-8. (This is the same as sim003.)
Notes:
    • I don't think the "instability" was really an instability at all. It just
      looks like a CFL blow up or something in a particular spot.

################################################################################

sim010

Mesh:
    • 3D bowl h=0.02 
Parameters:
    • ε² = 1e-2
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
    • Inversion → BiCgStab, tol=1e-8, P=I
    • Evolution → CG, tol=1e-8, P=I
Simulation Info:
    • Hardware = One 16G p100 GPU
    • Simulation time = 4 hrs
Notes:
    • This should now be a direct comparison to ../sim002 with the νPGCM 
      (except with ν = 1).
    • No instability...

################################################################################

sim011

Same as sim010 but with γ = 1/8, DQGMRES, itmax=500.
Notes:
    • Sim time: 4 hrs
    • Also no instability!!!

################################################################################

sim012

Same as sim011 but without advection.
Notes:
    • Clearly a difference in buoyancy profiles (just looking at last frame).

################################################################################

sim013 

Same setup as sim011 but in 2D bowl.
Notes:
    • Profiles between 2D and 3D are very similar! Good sign.
    • But looking at the global 2D solution, there is clearly a bit of
      gridscale noise.

################################################################################

sim014
Same as sim013 but using exact LU factorization for inversion.
Notes:
    • Gridscale noise goes away → iterative solver is not fully converged.

################################################################################

sim015
Same as sim011 but with GMRES, restart=true, itmax=0.
Notes:
    • Sim time: 5 hrs
    • Looks almost exactly the same as sim011.

################################################################################

sim016
Mesh:
    • 3D bowl h=0.01
Parameters:
    • ε² = 1e-4
    • μϱ = 1
    • γ = 1/8
    • f = 1
Forcing:
    • ν = 1
    • κ = 1e-2 + exp(-(z + H)/0.1)
Horizontal Diffusion: FALSE
Timestep:
    • Diffusion → Crank-Nicolson
    • Advection → Forward Euler
    • Δt = 0.1
    • T = 500Δt
Iterative Solvers:
    • Inversion → GMRES, memory=20, restart=true, tol=1e-8, P=I
    • Evolution → CG, tol=1e-8, P=I
Simulation Info:
    • Hardware = One 80G h100 GPU
    • Simulation time = 24+ hrs
Notes:
    • Working on a 3D sim that's close to the parameters we want.
    • Looks like CG starts taking only 1 or even 0 iterations after a while...
      I think this is because the norm of the residual is small due to the 
      small timestep. Maybe preconditioning would fix this, because it changes
      the norm. (see sim018)

################################################################################

sim017

2D simulations equivalent to sim016 for varying γ to look at convergence.

################################################################################

sim018

Same as sim016 but only diffusion. Want to see why nothing was happening with
Δt = 0.01.
Notes:
    • It does seem like it has to do with the norm of the residual being small.
      Using a preconditioner of P = diag(M) fixes things!

################################################################################

sim019

Same as sim016 but with γ = 1.
Notes:
    • CFL blow up at around t = 37.

################################################################################

sim020

Same as sim016 but with P_evolution = diag(M_b) and Δt = 0.05.

################################################################################
"""