# 1D Model

To understand some of the basic physics, it can be useful to consider a 1D model over a slope with variations only in
the vertical (see [Peterson2022,Peterson2023,Peterson2026](@cite)). 
```math
\begin{aligned}
-f v &= -P_x + \alpha^{-1} b \tan\theta + \alpha^2 \varepsilon^2 \sec^2\theta \partial_z ( \nu \partial_z u),\\
 f u &= -P_y +                             \alpha^2 \varepsilon^2 \sec^2\theta \partial_z ( \nu \partial_z v),\\
 \mu\varrho ( \partial_t b + u N^2 \tan\theta ) &= \alpha^2 \varepsilon^2 \sec^2\theta \partial_z [ \kappa (N^2 + \partial_z b) ],
\end{aligned}
```

## Derivation

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
and assume **no variations in the cross- or along-slope directions $\xi$ and $\eta$**.
The contravariant velocity components in the new coordinate system are of the form
```math
u^\xi = u^x, \quad u^\eta = u^y, \quad u^\zeta = u^z - u^x\tan\theta.
```
The partial derivatives transform as
```math
\partial_x = \partial_\xi - \tan\theta \partial_\zeta \to -\tan\theta \partial_\zeta, \quad \partial_y = \partial_\eta \to 0, \quad \partial_z = \partial_\zeta,
```
and the contravariant derivatives (found by computing $\partial^i = g^{ij}\partial_j$ where $g^{ij}$ is the contravariant
metric tensor) are
```math
\partial^\xi = \partial_\xi - \tan\theta \partial_\zeta \to - \tan\theta \partial_\zeta, 
\quad \partial^\eta = \partial_\eta, 
\quad \partial^\zeta = -\tan\theta \partial_\xi + \sec^2\theta \partial_\zeta \to \sec^2\theta \partial_\zeta.
```
Since $\partial_i u^i \to \partial_\zeta u^\zeta$ in 1D, continuity implies that $u^\zeta = 0$ and therefore 
$u^z = u^x \tan\theta = u^\xi \tan\theta$.
Each of the nine strain components are
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
Putting all of this together, our 1D model equations are 
```math
\begin{aligned}
-f u^\eta &= -P_x + \alpha^{-1} b' \tan\theta + \alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\xi),\\
 f u^\xi  &= -P_y +                            \alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta ( \nu \partial_\zeta u^\eta),\\
 \mu\varrho ( \partial_t b + u^\xi N^2 \tan\theta ) &= \alpha^2 \varepsilon^2 \sec^2\theta \partial_\zeta [ \kappa (N^2 + \partial_\zeta b') ],
\end{aligned}
```
where we have allowed for barotropic pressure gradients $P_x$ and $P_y$ (see [Peterson2022](@cite)) and the buoyancy has
been split into a background an perturbation (note that this is the *nondimensional* $N^2$):
```math
b = N^2 z + b' = N^2 (\zeta + \xi \tan\theta) + b'.
```
The $\xi$-momentum equation picks up a $\alpha^{-1} b' \tan\theta$ term because $z = \xi \tan\theta + \zeta$.

Drop funny notation:
$$ -f v = -P_x + \alpha^{-1} b' \tan\theta + \alpha^2\varepsilon^2\Gamma^2 \partial^2_z u $$
$$ f u = \alpha^2\varepsilon^2 \Gamma \partial^2_z v $$
$$ \mu\varrho (\partial_t b' + u N^2 \tan\theta) = \alpha^2\varepsilon^2 \Gamma \partial^2_z b'$$

Streamfunction $\partial_z \chi = u$ implies $f (\chi - U) = \alpha^2 \varepsilon^2 \Gamma \partial_z v$ by $y$-momentum.
Set $U = 0$.
Inversion:
$$ \alpha^2\varepsilon^2\Gamma^2 \partial^4_z \chi + \frac{f^2}{\alpha^2\varepsilon^2\Gamma} \chi = -\alpha^{-1} \partial_z b' \tan\theta $$

BL: 
$$ \mu\varrho \partial_z \chi_B N^2 \tan\theta = \alpha^2\varepsilon^2\Gamma \partial^2_z b'_B \to \partial_z b'_B = \frac{\mu\varrho N^2\tan\theta}{\alpha^2\varepsilon^2\Gamma} \chi_B $$
Substitute:
$$ \partial^4_z \chi_B + \left(\frac{f^2}{\alpha^4\varepsilon^4\Gamma^3} + \frac{\mu\varrho N^2 \tan^2\theta}{\alpha\alpha^4\varepsilon^4\Gamma^3} \right) \chi_B = 0$$
So
$$ (\delta q)^4 = 1 + \frac{\mu\varrho}{\alpha} \frac{N^2 \tan^2 \theta}{f^2}$$
where
$$ \delta = \alpha\varepsilon\Gamma^{3/4}\sqrt{\frac{2}{f}} $$


## other stuff
with the inverse map
```math
x = \xi, \quad y = \eta, \quad z = \zeta + \xi \tan\theta.
```
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

Let us now transform the momentum equation to these coordinates.
For the Coriolis term, we have that the contravariant components of the vertical unit vector are $z^\xi = 0$, $z^\eta = 0$,
and $z^\zeta = 1$, so the covariant components of the cross product $\vec{z} \times \vec{u}$ are
```math
\varepsilon_{\xi j k} z^j u^k = -u^\eta, \quad \varepsilon_{\eta j k} z^j u^k = u^\xi, \quad \varepsilon_{\zeta j k} z^j u^k = 0,
```
where here $\varepsilon$ is the Levi--Cevita symbol (not the Ekman number) and the contravariant velocity components are 
of the form
```math
u^\xi = u^x, \quad u^\eta = u^y, \quad u^\zeta = u^z - u^x\tan\theta.
```
Since pressure is a scalar, the components of its covariant derivative are simply partial derivatives.
The covariant components of the buoyancy term are
```math
\alpha^{-1} b z_\xi = \alpha^{-1} b \tan\theta , \quad \alpha^{-1} b z_\eta= 0, \quad \alpha^{-1} b z_\zeta = \alpha^{-1} b.
```
Now for the friction term. 
Since $g_{ij}$ is constant, the Christoffel symbols vanish, so the strain rate tensor in our new coordinates is simply
```math
\sigma^{ij} = \frac12 \left( \partial^i u^j + \partial^j u^i \right) = \frac12 \left( g^{ij} \partial_i u^j + g^{ij} \partial_j u^i \right).
```
The covariant derivatives are
```math
\partial_x = \partial_\xi - \tan\theta \pder{}{\zeta}, \quad \pder{}{y} = \pder{}{\eta}, \quad \pder{}{z} = \pder{}{\zeta}.
```
The divergence of the stress (with no Jacobian factor since $\sqrt{g} = 1) is then
```math
\partial_j\left( 2 \nu \sigma^{ij}\right).
```

And now the part that makes this model 1D: we assume **no variations in the cross- or along-slope directions $\xi$ and $\eta$**.
Thus, continuity implies that $u^\zeta = 0$:
```math
\pder{u^\xi}{\xi} + \pder{u^\eta}{\eta} + \pder{u^\zeta}{\zeta} = 0 \implies \pder{u^\zeta}{\zeta} = 0 \implies u^\zeta = 0,
```
and therefore $u^z = u^x \tan\theta = u^\xi \tan\theta$.

Let's see what happens to one of those pesky friction terms under this assumption that $\partial_\xi,\, \partial_\eta \to 0.$
For the first term in the brackets, we have
```math
\begin{aligned}
\nabla \cdot (\nu \nabla A) &= \left(\pder{}{\xi} - \tan\theta \pder{}{\zeta}\right) \left[ \nu \left(\pder{A}{\xi} - \tan\theta \pder{A}{\zeta} \right) \right]
+ \pder{}{\eta} \left( \nu \pder{A}{\eta} \right)
+ \pder{}{\zeta} \left( \nu \pder{A}{\zeta} \right)\\
&= (1 + \tan^2\theta) \pder{}{\zeta} \left( \nu \pder{A}{\zeta} \right).
\end{aligned}
```
To clean things up, we will define $\Gamma \equiv 1 + \tan^2 \theta$.
For the second term, we have
```math
\begin{aligned}
\nabla \cdot \left(\nu \pder{\vec{u}}{x} \right) &= \pder{}{x} \left(\nu \pder{u^x}{x} )
\end{aligned}
```