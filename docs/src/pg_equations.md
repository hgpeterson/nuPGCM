# PG Equations

As discussed in the [overview](index.md), we aim to describe the large-scale ocean circulation using the PG equations, which are derived from the Boussinesq equations by assuming large horizontal scales and small Rossby numbers.
In index notation, the dimensional PG equations in Cartesian space ``(x_1, x_2, x_3)`` read
```math
\begin{aligned}
     2 e_{ijk} \Omega_j u_k &= - \frac{\partial p}{\partial x_i} + b z_i + \frac{\partial}{\partial x_j} \left( 2 \nu \sigma_{ij} \right),\\
     \frac{\partial u_i}{\partial x_i} &= 0,\\
     \frac{\partial b}{\partial t} + u_i \frac{\partial b}{\partial x_i} &= \frac{\partial }{\partial x_i} \left( \kappa \frac{\partial b}{\partial x_i} \right),
\end{aligned}
```
where ``\vec{u} = (u_1, u_2, u_3)`` is the velocity vector, ``p`` is the pressure, and ``b`` is the buoyancy.
The first term in the momentum equation represents the Coriolis acceleration, with ``e_{ijk}`` the Levi--Civita symbol for the cross product and ``vec{\Omega}`` the rotation vector of the volume; we do not make the traditional approximation.
The buoyancy force acts only in the direction opposite to gravity, defined by the unit vector $\vec{z}(\vec{x})$. 
We parameterize turbulent mixing of buoyancy by a down-gradient flux proportional to the turbulent diffusivity ``\kappa``.
The Eliassen--Palm fluxes parameterizing mixing due to eddies contribute a diffusion term in the momentum equations with eddy viscosity ``\nu``.
To account for spatially varying ``\nu``, this friction term is written in terms of the rank-2 strain rate tensor,
```math
\sigma_{ij} = \sigma(\vec{u})_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right).
```
For constant~$\nu$, this term is equivalent to the regular Laplacian of ``\vec{u}``.
We apply a no-slip condition on the flow at the bottom boundary of the domain ``\Gamma_{\mathrm{B}``, and at the surface ``\Gamma_\mathrm{S}`` we demand no normal flow and that the stress in the local horizontal direction is set by the wind stress:
```math
\begin{aligned}
    \vec{u} = 0 \quad &\mathrm{on} \quad \Gamma_\mathrm{B},\\
    u_i n_i = 0 \quad \mathrm{and} \quad \nu \sigma_{ij} n_i = \tau_{ij} n_i \quad &\mathrm{on} \quad \Gamma_\mathrm{S},
\end{aligned}
```
where ``\vec{n}`` is the normal vector to the boundary.
The buoyancy flux through the bottom is set to ``\mathcal{G}`` (typically zero unless accounting for geothermal heating effects), and either the buoyancy or the buoyancy flux can be set at the surface:
```math
\begin{aligned}
    -\kappa \frac{\partial b}{\partial x_i} n_i = \mathcal{G} \quad &\mathrm{on} \quad \Gamma_\mathrm{B},\\
    b = b_\mathrm{S} \quad \mathrm{or} \quad -\kappa \frac{\partial b}{\partial x_i} n_i = \mathcal{F} \quad &\mathrm{on} \quad \Gamma_\mathrm{S}.
\end{aligned}
``` 

These PG dynamics can be viewed separately as an evolution equation for buoyancy and an inversion statement for the flow.