# PG Equations

As discussed in the [overview](../index.md), we aim to describe the large-scale ocean circulation using the PG equations, which are derived from the Boussinesq equations by assuming large horizontal scales and small Rossby numbers.
In index notation, the dimensional PG equations in Cartesian space $(x, y, z)$ read
```math
\begin{aligned}
    f \vec{z} \times \vec{u} &= -\nabla p + b \vec{z} + \nabla \cdot \left( 2 \nu \sigma(\vec{u}) \right), \hphantom{\frac12}\\
    \nabla \cdot \vec{u} &= 0, \hphantom{\frac12}\\
    \pder{{b}}{{t}} + \vec{u} \cdot \nabla b &= \nabla \cdot \left( \kappa \nabla {b} \right),
\end{aligned}
```
where $\vec{u} = (u, v, w)$ is the velocity vector, $p$ is the pressure, and $b$ is the buoyancy.
The first term in the momentum equation represents the Coriolis acceleration, with $\vec{z} = (0, 0, 1)$ being the unit vector in the vertical and $f$ the Coriolis parameter.
The buoyancy force acts only in the vertical.
We parameterize turbulent mixing of buoyancy by a down-gradient flux proportional to the turbulent diffusivity $\kappa$.
The Eliassen--Palm fluxes parameterizing mixing due to eddies contribute a diffusion term in the momentum equations with eddy viscosity $\nu$.
To account for spatially varying $\nu$, this friction term is written in terms of the rank-2 strain rate tensor $\sigma(\vec{u}) = \frac12 \left[ \left(\nabla \vec{u}\right) + \left(\nabla \vec{u}\right)^T \right]$ or, in index notation,
```math
\sigma(\vec{u})_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right).
```
For constant $\nu$, this term is equivalent to the regular Laplacian of $\vec{u}$.
We apply no-slip and no-normal flow conditions on at the bottom $z = -H$, and at the surface $z = 0$ we demand no normal flow and a stress set by the wind:
```math
\begin{aligned}
    \vec{u} = 0 \quad &\mathrm{at} \quad z = -H,\\
    w = 0 \quad \mathrm{and} \quad \nu \pder{\vec{u}_\perp}{z} = \vec{\tau} \quad &\mathrm{at} \quad z = 0,
\end{aligned}
```
where $\vec{u}_\perp = (u, v)$ and $\vec{\tau} = (\tau^x, \tau^y)$ is the normal vector to the boundary.
The buoyancy flux through the bottom is set to $\mathcal{G}$ (typically zero unless accounting for geothermal heating effects), and either the buoyancy or the buoyancy flux can be set at the surface:
```math
\begin{aligned}
    \kappa \pder{b}{z} = \mathcal{G} \quad &\mathrm{at} \quad z = -H,\\
    b = b_\mathrm{S} \quad \mathrm{or} \quad \kappa \pder{b}{z} = \mathcal{F} \quad &\mathrm{at} \quad z = 0.
\end{aligned}
``` 

These PG dynamics can be viewed separately as an evolution equation for buoyancy and an inversion statement for the flow.