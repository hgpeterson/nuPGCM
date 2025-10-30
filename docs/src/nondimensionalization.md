# Nondimensionalization

To isolate role of the aspect ratio in this problem, the ``\nu``PGCM ultimately solves the nondimensional PG equations, derived below.
We scale all spatial coordinates by the natural length scale of the domain ``L`` (e.g., the radius of the planet or the width of the basin) and all velocities by the same scale:
```math
x_i = L \nd{x}_i \quad \mathrm{and} \quad u_i = U \nd{u}_i,
```
for ``i = 1, 2, 3``.
Although the code supports arbitrary initial conditions, we typically initialize all simulations with flat isopycnals aligned with gravity and constant stratification ``N^2`` such that ``\partial_z b = N^2`` at ``t = 0`` (using the notation ``\partial_z \equiv z_i \partial_{x_i}``).
A natural scaling for buoyancy, therefore, would be ``b \sim N^2 H`` for some depth scale of the ocean ``H``.
Unlike in quasi-geostrophic theory, however, the PG equations do not explicitly impose a background stratification so that, in general, a representative scale for ``N^2`` in the abyssal ocean will depend on the context of the problem.
We additionally define characteristic scales for the rotation rate and mixing coefficients:
```math
\Omega_i \sim \Omega, \quad \nu \sim \nu_0, \quad \kappa \sim \kappa_0.
```
Finally, we assume that the pressure gradient term in the momentum equation scales with the Coriolis term, that the buoyancy also scales with the pressure scale divided by ``H`` from hydrostatic balance, and that time scales advectively:
```math
p \sim \Omega U L, \quad b \sim \frac{\Omega U L}{H} = N^2 H, \quad t \sim \frac{L}{U}.
```
Applying these scales yields the following nondimensional PG equations:
```math
\begin{aligned}
    2 e_{ijk} \nd{\Omega}_j \nd{u}_k &= -\frac{\partial \nd{p}}{\partial \nd{x}_i} + \alpha^{-1} \nd{b} z_i + \alpha^2 \varepsilon^2 \frac{\partial }{\partial \nd{x}_j} \left( 2 \nd{\nu} \nd{\sigma}_{ij} \right),\\
    \frac{\partial \nd{u}_i}{\partial \nd{x}_i} &= 0,\\
    \mu\varrho \left( \frac{\partial \nd{b}}{\partial \nd{t}} + \nd{u}_i \frac{\partial \nd{b}}{\partial \nd{x}_i} \right) &= \alpha^2 \varepsilon^2 \frac{\partial }{\partial \nd{x}_i} \left( \nd{\kappa} \frac{\partial \nd{b}}{\partial \nd{x}_i} \right),
\end{aligned}
```
where ``\alpha = H / L`` is the aspect ratio, ``\varepsilon^2 = \nu_0 / \Omega H^2`` is the Ekman number, ``\varrho = N^2 H^2 / \Omega^2 L^2`` is the Burger number, and ``\mu = \nu_0/\kappa_0`` is the turbulent Prandtl number.

*Figure here?*

With all three spatial coordinates scaled by $L$, the effect of the aspect ratio $\alpha$ on the dynamics is made explicit and the domain $\mathcal{D}$ itself must have an aspect ratio of $\alpha$. 
For an isolated basin with $x_3$ being the vertical coordinate aligned with gravity, this implies that $-\alpha \le x_3 \le 0$ (Fig. \ref{fig:alpha}a).
If instead $\mathcal{D}$ is the entire ocean, $\alpha$ naturally becomes the thickness of the shell relative to the radius of the sphere (Fig. \ref{fig:alpha}b).
This scaling guarantees that the viscous friction term in \eqref{eq:inversion-nd} is spatially isotropic, a desirable property for computing numerical solutions.
More importantly, for $\alpha > 0$ hydrostatic balance is not exactly required, as can be seen by dotting \eqref{eq:inversion-nd} with the local vertical $\vec{z}$:
```math
\pder{p}{z} = \alpha^{-1} b + \alpha^2 \varepsilon^2 \pder{}{x_j} \left( 2 \nu \sigma_{ij} \right) z_i,
```
where again $\partial_z \equiv z_i\partial_{x_i}$.
For the ocean, typical order-of-magnitude length scales are $H \approx 10^3$ m and $L \approx 10^6$ m, implying $\alpha \approx 10^{-3}$.
Hence, the small aspect ratio assumption is often made, eliminating the diffusion term in \eqref{eq:nonhydrostatic}.
While standard finite element techniques may be used to solve the Stokes problem with rotation, they become brittle under this approximation (e.g., [guillen-gonzalez_analysis_2015](@cite)).
To leverage established methods, we will therefore keep $\alpha$ larger than zero but small enough to capture the qualitative dynamics of the ocean.
A similar approach was taken by [salmon_simplified_1986](@cite) for a PG model with Rayleigh drag.

## Boundary conditions

Now let's work out how the boundary conditions must change under this particular nondimensionalization.
Of course, the no-slip and no-normal-flow conditions at the bottom still imply $\nd{\vec{u}} = 0$ at $\nd{z} = -\nd{H}$.
Since the buoyancy flux and wind stress boundary conditions involve vertical derivatives, though, we must take some care to ensure that they scale properly in the limit as $\alpha \to 0$. 

### Surface wind stress

To illustrate this point, it helps to think about the net meridional transport $\psi$, defined such that
```math
\pder{\psi}{z} = v \quad \text{and} \quad -\pder{\psi}{x} = w.
```
Recall that the velocity scale $U = N^2 H_0^2 / f_0 L$.
As with buoyancy, we want $\psi$ to scale like $U H_0$ (not $U L$), so we have
```math
\alpha \pder{\nd{\psi}}{\nd{z}} = \nd{v}.
``` 
As $\alpha \to 0$, the vertical derivative terms in the $x$-momentum equation should dominate such that
```math
-\nd{f} \nd{v} = -\pder{\nd{p}}{\nd{x}} + \alpha^2 \varepsilon^2 \pder{}{\nd{z}} \left( \nd{\nu} \pder{\nd{u}}{\nd{z}} \right).
```
Integrating in $z$, this gives
```math
-\prettyint{\nd{z}}{0}{\nd{f}\nd{v}}{\nd{z}'} = \alpha \nd{f} \nd{\psi} \sim \alpha^2 \varepsilon^2 \nd{\nu} \pder{\nd{u}}{\nd{z}} \Big|_{\nd{z} = 0},
```
neglecting the pressure gradient term and shear in the interior.
Here's where the limit comes in: as $\alpha \to 0$, we want this interior transport to be equal to the wind-driven circulation, i.e., $\nd{\psi} \sim \nd{\tau}^x/\nd{f}$.
For this to be true, we must have
```math
\alpha^2 \varepsilon^2 \nd{\nu} \pder{\nd{u}}{\nd{z}} = \alpha \nd{\tau}^x \quad \text{at} \quad \nd{z} = 0,
```
or, more generally,
```math
\boxed{\alpha^2 \varepsilon^2 \nd{\nu} \pder{\nd{\vec{u}}_\perp}{\nd{z}} = \alpha \nd{\vec{\tau}} \quad \text{at} \quad \nd{z} = 0.}
```

### Surface buoyancy flux

What about the surface buoyancy flux condition?
Here we want to ensure that the integrated buoyancy tendency remains fixed as $\alpha \to 0$.
Integrating the buoyancy equation with advection neglected and diffusion only in the vertical, we have
```math
\mu\varrho \prettyint{\nd{z}}{0}{\pder{\nd{b}}{\nd{t}}}{\nd{z}'} = \alpha^2 \varepsilon^2 \nd{\kappa}\pder{\nd{b}}{\nd{z}} \Big|_{\nd{z} = 0},
```
again neglecting the flux in the interior.
For this to be equal to the net buoyancy input at the surface even as $\alpha \to 0$, we must have
```math
\boxed{\alpha^2 \varepsilon^2 \nd{\kappa}\pder{\nd{b}}{\nd{z}} = \nd{F} \quad \text{at} \quad \nd{z} = 0.}
```

## Scales and parameter values

The parameters $\varepsilon$, $\mu$, and $\varrho$ are free to be chosen to match the physical context.
The Coriolis parameter for Earth is about $f_0 \approx 10^{-4}$ s$^{-1}$ and the stratification in the deep ocean is around $N \approx 10^{-3}$ s$^{-1}$, yielding a Burger number of $\varrho \approx 10^{-4}$.
Over rough topography, one might expect strong turbulence associated with a turbulent diffusivity on the order of $\kappa_0 \approx 10^{-3}$ m$^2$ s$^{-1}$ (e.g., [waterhouse_global_2014](@cite)).
The magnitude of the turbulent viscosity depends on whether or not a parameterization of eddies is considered.
Without eddies, it is reasonable to assume that, for weakly stratified abyssal waters, small-scale mixing of buoyancy would occur on similar scales to the mixing of momentum, implying that $\nu_0 \sim \kappa_0$, or $\mu \sim 1$ (e.g., [caulfield_layering_2021](@cite)).
An Eliassen--Palm flux equivalent to an eddy diffusivity of $K \approx 10^3$ m$^2$ s$^{-1}$ [gent_isopycnal_1990](@cite), however, would require an enhanced viscosity of $\nu_0 \approx 10$ m$^2$ s$^{-1}$, or $\mu \approx 10^4$.
In the first case, where $\nu_0 \approx 10^{-3}$ m$^2$ s$^{-1}$, the Ekman number $\varepsilon$ is on the order of $10^{-3}$.
In the eddy-parameterizing case of $\nu_0 \approx 10$ m$^2$ s$^{-1}$, on the other hand, it is enhanced to $\varepsilon \approx 10^{-1}$.
Given that the nondimensional bottom Ekman layer thickness $\delta/L = \sqrt{2 \nu_0 / (f_0 L^2)} = \sqrt{2} \, \alpha \varepsilon$, these differences are crucial in setting the minimum resolution required for the simulation.

From the wind stress boundary condition, we must have...

### References

```@bibliography
```