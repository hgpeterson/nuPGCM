# 1D Model

## Slope-rotated coordinates

To understand some of the basic physics of the PG model, it can be useful to consider a 1D model over a uniform slope 
at an angle $\theta$ with the horizontal (see [Peterson2022,Peterson2023,Peterson2026](@cite)). 
The governing equations for the cross-slope flow $u$, along-slope flow $v$, and buoyancy perturbation $b$ to a background
(nondimensional) stratification $N^2$ are
```math
\begin{aligned}
-f v \cos\theta &= -P_x + \alpha^{-1} b \sin\theta + \alpha^2 \varepsilon^2 \pder{}{z} \left( \nu \pder{u}{z} \right),\\
 f u \cos\theta &= -P_y + \hphantom{aaaaaaaaaa,}     \alpha^2 \varepsilon^2 \pder{}{z} \left( \nu \pder{v}{z} \right),\\
 \mu\varrho \left( \pder{b}{t} + u N^2 \sin\theta \right) &= \alpha^2 \varepsilon^2 \pder{}{z} \left[ \kappa \left( N^2\cos\theta + \pder{b}{z} \right) \right],
\end{aligned}
```
where $P_x$ and $P_y$ are barotropic pressure gradients [Peterson2022,Peterson2026](@cite).
The boundary conditions on the flow are no slip at the bottom, no stress at the top, and a constraint on the barotropic 
transport:
```math
\begin{aligned}
u = v = 0 \quad &\text{at} \quad z = -H, \vphantom{\frac12}\\
\pder{u}{z} = \pder{v}{z} = 0 \quad &\text{at} \quad z = 0,\\
\int_{-H}^0 u \; \text{d}z = \int_{-H}^0 v \; \text{d}z &= 0
\end{aligned}
```
The buoyancy perturbation satisfies
```math
\begin{aligned}
N^2\cos\theta + \pder{b}{z} = 0 \quad &\text{at} \quad z = -H,\\
b = 0 \quad &\text{at} \quad z = 0. \vphantom{\frac12}
\end{aligned}
```

### Boundary layer scale

We define a streamfunction such that $\partial_z \chi = u$.
Then the $y$-momentum equation becomes
```math
f \partial_z \chi \cos\theta = -P_y + \alpha^2 \varepsilon^2 \partial_z ( \nu \partial_z v).
```
Integrating from the top $z = 0$ down to some level $z$, this implies
```math
f \chi \cos\theta = -P_y z + \alpha^2 \varepsilon^2 \nu \partial_z v.
```
Differentiating the $x$-momentum equation and substituting this in, we get
```math
\alpha^2 \varepsilon^2 \partial^2_z ( \nu \partial^2_z \chi) + \frac{f^2 \chi \cos^2\theta + f P_y z \cos\theta}{\alpha^2 \varepsilon^2 \nu} = -\alpha^{-1} \partial_z b' \sin\theta.
```
In the boundary layer (BL), we have
```math
\begin{aligned}
\alpha^2 \varepsilon^2 \nu_B \partial^4_z \chi_B + \frac{f^2 \cos^2\theta}{\alpha^2 \varepsilon^2 \nu_B} \chi_B &= -\alpha^{-1} \partial_z b'_B \sin\theta,\\
\mu\varrho \chi_B N^2 \sin\theta &= \alpha^2 \varepsilon^2 \kappa_B \partial_z b'_B, \vphantom{\frac12}
\end{aligned}
```
where the second equation comes from assuming a steady flow in the BL and integrating.
Substituting $\partial_z b'_B$ from the buoyancy equation into the inversion, we get
```math
\partial^4_z \chi_B + 4q^4 \chi_B = 0,\\
```
where
```math
\boxed{(\delta q)^4 = 1 + \frac{\mu\varrho}{\alpha} \frac{\nu_B}{\kappa_B} \frac{N^2 \tan^2\theta}{f^2} }
```
where
```math
\delta = \alpha \varepsilon \sqrt{\frac{2\nu_B}{f}}
```
is the typical flat-bottom Ekman BL scale.
The BL thickness is then $q^{-1}$.
Note that since $N^2 \sim \alpha^{-1}$ and $\tan \theta \sim \alpha$, the scaling for $\delta q$ is independent of
$\alpha$.

## Slope-aligned coordinates

To understand some of the basic physics of the PG model, it can be useful to consider a 1D model over a uniform slope 
at an angle $\theta$ with the horizontal (see [Peterson2022,Peterson2023,Peterson2026](@cite)). 
The governing equations for the cross-slope flow $u$, along-slope flow $v$, and buoyancy perturbation $b$ to a background
(nondimensional) stratification $N^2$ are
```math
\begin{aligned}
-f v &= -P_x + \alpha^{-1} b \tan\theta + \alpha^2 \varepsilon^2 \sec^2\theta \pder{}{z} \left( \nu \pder{u}{z} \right),\\
 f u &= -P_y + \hphantom{aaaaaaaaaa,}     \alpha^2 \varepsilon^2 \sec^2\theta \pder{}{z} \left( \nu \pder{v}{z} \right),\\
 \mu\varrho \left( \pder{b}{t} + u N^2 \tan\theta \right) &= \alpha^2 \varepsilon^2 \pder{}{z} \left[ \kappa \left( N^2 + \sec^2\theta \pder{b}{z} \right) \right],
\end{aligned}
```
where $P_x$ and $P_y$ are barotropic pressure gradients [Peterson2022,Peterson2026](@cite).
The boundary conditions on the flow are no slip at the bottom, no stress at the top, and a constraint on the barotropic 
transport:
```math
\begin{aligned}
u = v = 0 \quad &\text{at} \quad z = -H, \vphantom{\frac12}\\
\pder{u}{z} = \pder{v}{z} = 0 \quad &\text{at} \quad z = 0,\\
\int_{-H}^0 u \; \text{d}z = \int_{-H}^0 v \; \text{d}z &= 0
\end{aligned}
```
The buoyancy perturbation satisfies
```math
\begin{aligned}
N^2 + \sec^2\theta \pder{b}{z} = 0 \quad &\text{at} \quad z = -H,\\
b = 0 \quad &\text{at} \quad z = 0. \vphantom{\frac12}
\end{aligned}
```

### Derivation

We begin with the nondimensional PG equations from [before](nondimensionalization.md), now written in index notation:
```math
\begin{aligned}
    f e_{ijk} z^j u^k &= -\partial_i p + \alpha^{-1} b z_i + \alpha^2 \varepsilon^2 \partial_j \left( 2 \nu \sigma^{ij} \right),\\
    \partial_i u^i &= 0,\\
    \mu\varrho \left( \partial_t b + u^i \partial_i b \right) &= \alpha^2\varepsilon^2 \partial_i \left( \kappa \partial^i b \right),
\end{aligned}
```
where $e_{ijk}$ is the Levi--Cevita symbol.
As in [Peterson2022,Peterson2023,Peterson2026](@cite), we assume a uniform bottom slope of $\theta$ and transform these 
equations to the (non-orthogonal) coordinates
```math
\xi = x, \quad \eta = y, \quad \zeta = z - x \tan\theta,
```
with the inverse map
```math
x = \xi, \quad y = \eta, \quad z = \zeta + \xi \tan\theta.
```
and assume **no variations in the cross- or along-slope directions $\xi$ and $\eta$**.
The (constant) covariant basis vectors are then
```math
\vec{e}_\xi = (1, 0, \tan\theta)^T, \quad \vec{e}_\eta = (0, 1, 0)^T, \quad \vec{e}_\zeta = (0, 0, 1)^T,
```
which implies a (constant) covariant metric tensor of
```math
g_{ij} = \vec{e}_i \cdot \vec{e}_j =
\left(
\begin{array}{ccc}
    \sec^2\theta & 0 & \tan\theta\\
    0 & 1 & 0\\
    \tan\theta & 0 & 1
\end{array}
\right),
```
using $1 + \tan^2\theta = \sec^2 \theta$.
It's inverse is the contravariant metric tensor:
```math
g^{ij} = 
\left(
\begin{array}{ccc}
    1 & 0 & -\tan\theta\\
    0 & 1 & 0\\
    -\tan\theta & 0 & \sec^2\theta
\end{array}
\right).
```
The volume element $\sqrt{g} = 1$ is the square root of the determinant of $g_{ij}$.
The partial derivatives transform as
```math
\partial_x = \partial_\xi - \tan\theta \partial_\zeta \to -\tan\theta \partial_\zeta, \quad \partial_y = \partial_\eta \to 0, \quad \partial_z = \partial_\zeta,
```
and the contravariant derivatives (found by computing $\partial^i = g^{ij}\partial_j$) are
```math
\partial^\xi = \partial_\xi - \tan\theta \partial_\zeta \to - \tan\theta \partial_\zeta, 
\quad \partial^\eta = \partial_\eta \to 0, 
\quad \partial^\zeta = -\tan\theta \partial_\xi + \sec^2\theta \partial_\zeta \to \sec^2\theta \partial_\zeta.
```
The contravariant velocity components are of the form
```math
u^\xi = u^x, \quad u^\eta = u^y, \quad u^\zeta = u^z - u^x\tan\theta.
```
Continuity then implies that 
```math
\partial_i u^i = 0 \to \partial_\zeta u^\zeta = 0 \implies \boxed{u^\zeta = 0}
```
and therefore $u^z = u^x \tan\theta = u^\xi \tan\theta$.

Let us now transform the momentum equations to these coordinates term by term.
For the Coriolis term, we have that the contravariant components of the vertical unit vector are $z^\xi = 0$, $z^\eta = 0$,
and $z^\zeta = 1$, so the covariant components of the cross product $\vec{z} \times \vec{u}$ are
```math
e_{\xi j k} z^j u^k = -u^\eta, \quad e_{\eta j k} z^j u^k = u^\xi, \quad e_{\zeta j k} z^j u^k = 0,
```
The covariant components of the buoyancy term are
```math
\alpha^{-1} b z_\xi = \alpha^{-1} b \tan\theta , \quad \alpha^{-1} b z_\eta= 0, \quad \alpha^{-1} b z_\zeta = \alpha^{-1} b.
```
Now for the friction term. 
Each of the nine components of the strain rate tensor are
```math
\begin{aligned}
    \sigma^{ij} = \frac12 (\partial^i u^j + \partial^j u^i) &= 
    \frac12 \left(
    \begin{array}{ccc}
    2\partial^\xi u^\xi & \partial^\xi u^\eta + \partial^\eta u^\xi & \partial^\xi u^\zeta + \partial^\zeta u^\xi\\
    \partial^\eta u^\xi + \partial^\xi u^\eta & 2 \partial^\eta u^\eta & \partial^\eta u^\zeta + \partial^\zeta u^\eta\\
    \partial^\zeta u^\xi + \partial^\xi u^\zeta & \partial^\zeta u^\eta + \partial^\eta u^\zeta & 2 \partial^\zeta u^\zeta
    \end{array}
    \right)\\
    &\to
    \frac12 \left(
    \begin{array}{ccc}
    -2\tan\theta \partial_\zeta u^\xi & -\tan\theta \partial_\zeta u^\eta & \sec^2\theta \partial_\zeta u^\xi\\
    -\tan\theta \partial_\zeta u^\eta & 0 & \sec^2\theta \partial_\zeta u^\eta\\
    \sec^2\theta \partial_\zeta u^\xi & \sec^2\theta \partial_\zeta u^\eta & 0
    \end{array}
    \right)
\end{aligned}
```
Since $\partial_\xi$ and $\partial_\eta$ are neglected, all we need is $\sigma^{i\zeta}$ (the third column) for the 
divergence.
Thus, we get the terms
```math
\alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\xi ) \quad \text{and} \quad
\alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\eta )
```
in the $\xi$- and $\eta$-momentum equations.
Lastly, since pressure is a scalar, the components of its covariant derivative are simply partial derivatives.
We allow for barotropic pressure gradients so that $\partial_\xi p \to P_x$ and $\partial_\eta \to P_y$
[Peterson2022,Peterson2026](@cite).
Putting all of this together, our 1D $\xi$- and $\eta$-momentum equations are
```math
\boxed{
\begin{aligned}
-f u^\eta &= -P_x + \alpha^{-1} b' \tan\theta + \alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\xi),\\
 f u^\xi  &= -P_y + \hphantom{aaaaaaaaaaa}      \alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\eta).
\end{aligned}
}
```
Here the buoyancy has been split into a background and perturbation (note that this is the *nondimensional* $N^2$):
```math
b = N^2 z + b' = N^2 (\zeta + \xi \tan\theta) + b'.
```

Now let's work on the buoyancy equation.
The covariant gradient of the background buoyancy is
```math
\partial_i [N^2 (\zeta + \xi \tan\theta)] = (N^2 \tan \theta, 0, N^2)^T,
```
so, since $u^\zeta = 0$ and we neglect $\partial_\xi b'$ and $\partial_\eta b'$, the advection term is simply
```math
u^i \partial_i b = u^\xi N^2 \tan\theta.
```
The contravariant gradient of $b$ is
```math
\begin{aligned}
\partial^\xi b &= (\partial_\xi - \tan\theta \partial_\zeta) b' \to -\tan\theta \partial_\zeta b',\\
\partial^\eta b &= \partial_\eta b' \to 0,\\
\partial^\zeta b &= N^2 + (-\tan\theta \partial_\xi + \sec^2\theta \partial_\zeta) b' \to N^2 + \sec^2\theta \partial_\zeta b'.
\end{aligned}
```
The Laplacian term then reduces to
```math
\partial_i (\kappa \partial^i b) \to \partial_\zeta [\kappa (N^2 + \sec^2\theta \partial_\zeta b' )],
```
and the no-flux boundary condition becomes
```math
n_i \partial^i b = \cos\theta (N^2 + \sec^2\theta \partial_\zeta b') = 0 \quad \text{at} \quad z = -H,
```
since the bottom-normal vector $n^i$ is simply $(0, 0, \cos\theta)^T$ in the slope-aligned coordinates.
In summary, the 1D buoyancy equation reads:
```math
\boxed{\mu\varrho ( \partial_t b + u^\xi N^2 \tan\theta ) = \alpha^2 \varepsilon^2 \partial_\zeta [ \kappa (N^2 + \sec^2\theta \partial_\zeta b') ]}
```