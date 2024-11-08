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

Same as sim019 but with P_evolution = diag(M_b) and Δt = 0.05.
Notes:
    • Looking pretty good, although for some reason there is quite a bit of 
      noise in the interior. Perhaps a better quality mesh would help?
    • Sim makes it to t = 64 after 24 hrs. Main bottleneck now is RHS setup
      for both inversion and evolution.

################################################################################

sim021

Same as sim020 but with β = 1.

################################################################################

sim022

Same as sim021 but with γ = 1/4 and horizontal diffusion in the evolution
equation.

################################################################################

sim023

Same as sim022 but with μϱ = 1e-4, Δt = 1e-4*μϱ/ε² = 1e-4, 
T = 5e-2*μϱ/ε² = 5e-2.

################################################################################

sim024

Just a quick low-res simulation to use for code-testing.
h = 0.05, ε² = 1e-1, μϱ = 1, γ = 1/4, f = 1, β = 1.

################################################################################

sim025

Same as sim024 but with β = 0.

################################################################################

sim026

Same as sim022 but with β = 0.

Notes:
    • CFL blow up at around t = 37 again (cf sim019). Difference between this
      sim and sim020 is γ = 1/4 here.
    • Sort of makes sense that CFL blow up happens when β = 0 since the flow is
      stronger.

################################################################################

sim027

Re-running sim025 but with `b` as the buoyancy _perturbation_ about a background
linear profile.

Notes:
    • Leads to is much smoother inversions and no wait time at the beginning!

################################################################################

sim028

Re-running sim022 but with `b` as perturbation to background and no horizontal
diffusion in inversion _or_ evolution.

Notes:
    • Beautiful, smooth solution, but it blows up at t = 10... perhaps horiz.
      diff. is needed after all. (should look in to smaller/higher-order time
      steps, too, though)

################################################################################

sim029 

Same as sim027 but with no horizontal diffusion in inversion _or_ evolution.

################################################################################

sim030

Same as sim028 but putting the horizontal diffusion back in.

################################################################################

sim031 

2D simulation equivalent to sim030 testing out new `run.jl` script that can
handle 2D/3D and also looking at b as perturbation.

################################################################################

sim032 

Same as sim031 but with a non-uniform mesh.

Note:
    • I was hoping that with the new buoyancy perturbation setup we would be
      able to use a mesh with a higher resolution near the bottom, but it seems
      like we still get some noise in the interior.
    • The solution was stable, though, but it actually took longer to run (3.5
      hrs) than sim031 (2.5 hrs), presumably because GMRES had to take more
      iterations.

################################################################################

sim033

Same as sim030 but with β = 0.5.

################################################################################

sim034

Same as sim030 but with β = 0.

################################################################################

sim035

Same as sim034 but with μϱ = 1e-4, Δt = 1e-4*μϱ/ε² = 1e-4. 

################################################################################

sim036

Same as sim033 but with μϱ = 1e-4, Δt = 1e-4*μϱ/ε² = 1e-4.

################################################################################

sim037

Same as sim030 but with μϱ = 1e-4, Δt = 1e-4*μϱ/ε² = 1e-4.

################################################################################

sim038

Same as sim034 (β = 0, μϱ = 1) but with Δt = 0.01.

################################################################################

sim039

2D simulation using new time integration scheme: Strang splitting with CN for 
diffusion and midpoint method for adv. 

    • Parameters: μϱ = 1, ε² = 1e-4, Δt = 0.05, γ = 1/4
    • Direct solvers for inversion, evolution
    • T = 500, h = 0.01, sim time = 1.5 hrs
     
###############################################################################

sim040 

Redo 2D sim031 with new time integration scheme and direct solvers.

###############################################################################

sim041 

Redo beta-plane sim030 with new time integration scheme.

###############################################################################

sim042

Redo f-plane sim034 (Δt = 0.05) with new time integration scheme.

###############################################################################

sim043

Redo f-plane sim038 (Δt = 0.01) with new time integration scheme.

###############################################################################
"""