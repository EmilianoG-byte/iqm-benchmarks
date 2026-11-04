# Derivations for the identifiability and Cramér–Rao notes

Companion to [README.md](README.md). It writes out the steps the README states briefly. Each section names the README section it supports. Notation is that of [README Section 3](README.md#3-setting-and-notation). Full references are in [README Section 8](README.md#8-sources).

Provenance tags are the same as in the README:

- **[Source: …]**: a result from the literature. **(checked)** means the statement and its number were compared with the source text on 2026-11-04. Otherwise the reference is from memory.
- **[Derived here]**: a calculation written out in this file.
- **[Numerical: script]**: our own computation, with a script in [`scripts/`](scripts/).

## Contents

- [A. Generic rank of the Jacobian](#a-generic-rank-of-the-jacobian) (README 4.2, 4.3)
- [B. The Riemannian Hessian in the tangent basis](#b-the-riemannian-hessian-in-the-tangent-basis) (README 4.4)
- [C. Fisher information on the tangent space](#c-fisher-information-on-the-tangent-space) (README 4.5)
- [D. Scalar bound and curvature terms](#d-scalar-bound-and-curvature-terms) (README 4.6)
- [E. The van Trees bound](#e-the-van-trees-bound) (README 4.6)
- [F. The per-direction test statistic](#f-the-per-direction-test-statistic) (README 5.5)

---

## A. Generic rank of the Jacobian

Supports README Section 4.3, "Generic rank".

**Claim.** Fix the design (the list of circuits). Then $\operatorname{rank}J_T(\theta)$ has the same value $r_{\rm gen}$ at every $\theta\in\mathcal M$ except on a closed set of measure zero. For our designs, $r_{\rm gen} = \min(3N, 624)$.

Steps A.1–A.6 prove this. A.7 describes the exceptional points.

### A.1 The objects

| Object | Size | Definition |
|---|---|---|
| $\theta$ | 928 | real and imaginary parts of all entries of $K$, $A$, $B$ |
| $p(\theta)$ | $4N$ | the formula of README Section 3, evaluated at any $\theta\in\mathbb R^{928}$. Off $\mathcal M$ the entries need not be probabilities. |
| $J(\theta) = \partial p/\partial\theta$ | $4N\times928$ | $J_{(c,i),a} = \partial p_{ci}/\partial\theta_a$. For any direction $v\in\mathbb R^{928}$, $Jv = \frac{d}{dt}p(\theta+tv)\big\rvert_{t=0}$. |
| $h(\theta)$, $Dh(\theta)$ | 225; $225\times928$ | constraint function (README 4.1) and its Jacobian. On $\mathcal M$, $\operatorname{rank}Dh = 113$. |
| $T(\theta)$ | $928\times815$ | orthonormal basis of the tangent space $T_\theta\mathcal M = \ker Dh(\theta)$ [Source: Boumal 2023, Thm. 3.15 (checked)], from the singular value decomposition (SVD) of $Dh$ |
| $J_T = JT$ | $4N\times815$ | column $k$ is $Jt_k$, the change of all probabilities along the tangent basis vector $t_k$. For a tangent vector $v = Tz$: $Jv = J_Tz$. |

$J$ describes how the probabilities respond to *any* change of the 928 numbers, including changes that leave $\mathcal M$ (for example scaling one Kraus operator, so that $\sum_r K_r^\dagger K_r\ne I$). $J_T$ keeps only the changes that stay on $\mathcal M$ to first order and writes them in 815 coordinates. Therefore

$$\operatorname{rank}J_T = \dim\{Jv : v\in T_\theta\mathcal M\},$$

the number of independent ways in which the probabilities can change, to first order, when the gate set moves along $\mathcal M$.

**Polynomial structure.** Write every entry of $K$, $A$, $B$ as $x + iy$. Then $p_{ci} = \operatorname{tr}[A_i^\dagger A_i\,\mathcal G_{g_L}\circ\dots\circ\mathcal G_{g_1}(BB^\dagger)]$ is a sum of products of entries of $K$, $\bar K$, $A$, $\bar A$, $B$, $\bar B$. It is real, so it is a real polynomial in $\theta$ of degree $2L + 4$. Derivatives of polynomials are polynomials, so every entry of $J(\theta)$ is a polynomial in $\theta$. $h$ is quadratic, so every entry of $Dh(\theta)$ is linear in $\theta$.

$T(\theta)$ is not polynomial in $\theta$. It comes out of an SVD, and in general no continuous choice of orthonormal tangent bases exists over the whole manifold (Boumal 2013, p. 8, makes this point for the sphere via the hairy ball theorem). Step A.2 removes $T$.

### A.2 A rank formula without $T$

**Lemma.** $\displaystyle\operatorname{rank}J_T = \operatorname{rank}\begin{pmatrix}J\\ Dh\end{pmatrix} - \operatorname{rank}Dh.$

**Proof.** [Derived here] We compare kernels and use rank–nullity (rank + dimension of the kernel = number of columns).

1. A vector $v$ is in the kernel of the stacked matrix iff $Jv = 0$ and $Dh\,v = 0$. So $\ker\begin{pmatrix}J\\Dh\end{pmatrix} = \ker J\cap\ker Dh$, and $\operatorname{rank}\begin{pmatrix}J\\Dh\end{pmatrix} = 928 - \dim(\ker J\cap\ker Dh)$.
2. $\operatorname{rank}Dh = 928 - \dim\ker Dh$.
3. $J_Tz = 0$ iff $Tz\in\ker J$. The map $z\mapsto Tz$ is injective (orthonormal columns) with image $\ker Dh$, so it maps $\ker J_T$ one-to-one onto $\ker J\cap\ker Dh$. Hence $\operatorname{rank}J_T = 815 - \dim(\ker J\cap\ker Dh) = \dim\ker Dh - \dim(\ker J\cap\ker Dh)$.

Subtracting 2 from 1 gives $\dim\ker Dh - \dim(\ker J\cap\ker Dh)$, which is the expression in 3. ∎

On $\mathcal M$, $\operatorname{rank}Dh = 113$ at every point (README 4.1). So

$$\operatorname{rank}J_T(\theta) = \operatorname{rank}X(\theta) - 113,\qquad X(\theta) = \begin{pmatrix}J(\theta)\\ Dh(\theta)\end{pmatrix},$$

and $X$ has polynomial entries. $\operatorname{rank}J_T$ is low exactly where $\operatorname{rank}X$ is low.

### A.3 Rank and minors

A *minor of order $r$* of a matrix is the determinant of an $r\times r$ submatrix (pick any $r$ rows and any $r$ columns).

**Lemma.** A matrix $X$ has rank $\ge r$ iff at least one minor of order $r$ is nonzero.

**Proof.** [Derived here]
- (⇐) Suppose an $r\times r$ submatrix is invertible. Then its $r$ columns are linearly independent. The corresponding full columns of $X$ are then also independent, because a linear relation among the full columns would hold in particular on the selected rows. So $\operatorname{rank}X\ge r$.
- (⇒) Suppose $\operatorname{rank}X\ge r$. Pick $r$ independent columns. They form a matrix with $r$ columns and rank $r$. Row rank equals column rank, so it has $r$ independent rows. The $r\times r$ submatrix on those rows and columns has independent rows, so it is invertible and its determinant is nonzero. ∎

**Consequence.** Let $r^* = \max_{\theta\in\mathcal M}\operatorname{rank}X(\theta)$. By the lemma, $\operatorname{rank}X(\theta) < r^*$ iff every minor of order $r^*$ vanishes at $\theta$. A minor is a sum of products of entries, so it is a polynomial in $\theta$. Collect them into one polynomial:

$$g(\theta) = \sum_{\text{minors of order } r^*}(\text{minor})^2 .$$

A sum of squares of real numbers is zero iff each term is zero. Therefore

$$Z := \{\theta\in\mathcal M : \operatorname{rank}J_T(\theta) < r^* - 113\} = \{\theta\in\mathcal M : g(\theta) = 0\}.$$

$Z$ is closed, because it is the zero set of a continuous function. $g$ is not identically zero on $\mathcal M$, because at a point where the rank $r^*$ is reached some minor is nonzero.

### A.4 The manifold is connected

$\mathcal M$ is a product of complex Stiefel manifolds $\mathrm{St}(n,p) = \{X\in\mathbb C^{n\times p}: X^\dagger X = I_p\}$. [Derived here]

1. **$U(n)$ is connected.** By the spectral theorem every unitary is $U = Q\operatorname{diag}(e^{i\varphi_1},\dots,e^{i\varphi_n})Q^\dagger$ with $Q$ unitary. The path $U(t) = Q\operatorname{diag}(e^{it\varphi_1},\dots,e^{it\varphi_n})Q^\dagger$, $t\in[0,1]$, consists of unitaries and joins $I$ to $U$.
2. **$U(n)$ maps onto $\mathrm{St}(n,p)$.** The map "keep the first $p$ columns" is continuous. It is onto: the columns of any $X\in\mathrm{St}(n,p)$ are orthonormal, and Gram–Schmidt completes them to an orthonormal basis of $\mathbb C^n$, i.e. to a unitary $[X, X_\perp]$.
3. The continuous image of a connected space is connected, and a product of connected spaces is connected. ∎

(For real matrices this would fail at $p = n$: $O(n)$ has two components. The complex case has no such problem.)

### A.5 Zero sets of analytic functions are small

A function is *real-analytic* if near every point it equals a convergent power series. Polynomials are real-analytic.

**$\mathcal M$ is a real-analytic manifold.** 112 of the 225 outputs of $h$ are identically zero (imaginary parts of diagonal entries) or identical, up to sign, to another output (the lower triangle copies the upper triangle of the Hermitian matrix $K^\dagger K - I$). Keeping the other 113 outputs gives a polynomial map $h_{113}$ with the same zero set $\mathcal M$ and a full-rank differential on $\mathcal M$. The real-analytic implicit function theorem [Source: Krantz & Parks 2002, Sec. 2.3] then gives charts around every point in which $\mathcal M$ is the graph of a real-analytic function. Composed with such a chart, $g$ is a real-analytic function of 815 variables. The chart domains can be taken to be balls, which are connected.

Two facts are needed.

**(F1)** If $G$ is real-analytic on a connected open set $U\subset\mathbb R^n$ and not identically zero, then $\{x\in U: G(x) = 0\}$ has Lebesgue measure zero. [Source: Mityagin 2020 (statement checked in the abstract)]

**(F2) Identity theorem.** If $g$ is real-analytic on a connected real-analytic manifold, then either $g\equiv0$, or $g$ vanishes on no nonempty open subset.

*Proof of F2.* [Derived here] Let $O$ be the set of points near which $g$ vanishes identically. $O$ is open by definition. $O$ is also closed. Take $\theta_n\in O$ with $\theta_n\to\theta$ and work in an analytic chart around $\theta$. All partial derivatives of $g$ vanish at each $\theta_n$, since $g\equiv 0$ near $\theta_n$. By continuity they all vanish at $\theta$. So the Taylor series of $g$ at $\theta$ is zero, and since it converges to $g$ near $\theta$, $g\equiv0$ near $\theta$, i.e. $\theta\in O$. A connected space has no subsets that are open and closed other than $\emptyset$ and the whole space, so $O = \emptyset$ or $O = \mathcal M$. ∎

**Conclusion.** $g\not\equiv0$ on $\mathcal M$ (A.3), so by F2 it is not identically zero on any chart ball. By F1, $Z$ has measure zero in each chart. $\mathcal M$ is compact, so finitely many charts cover it. A subset of a manifold has measure zero if its image in every chart has measure zero [Source: Lee 2012, Ch. 6, "Sets of measure zero"]. Hence $Z$ has measure zero, which is the same as zero Riemannian volume. $Z$ is closed with empty interior, so its complement, where the rank is maximal, is open and dense.

### A.6 Why the generic rank is 624

Let $r_{\rm gen} = r^* - 113$ be the maximal rank of $J_T$ on $\mathcal M$ for a given design. By A.5 it is attained at almost every point. Take $N\ge 208$ (for smaller $N$ replace 624 by $3N$).

**$r_{\rm gen}\ge624$.** At the truth, the computed $J_T$ has $\sigma_{624}/\sigma_{\max}\approx 10^{-6}$ at $N = 400$, while the zero singular values sit at $\approx10^{-15}$ (README 5.2). Rounding cannot create rank. By Weyl's inequality for singular values, $\lvert\sigma_k(J_T + E) - \sigma_k(J_T)\rvert\le\lVert E\rVert_2$ for a perturbation $E$ [Source: Golub & Van Loan 2013, Cor. 8.6.2]. A computed singular value far above the rounding error $\lVert E\rVert_2\sim10^{-15}\sigma_{\max}$ therefore belongs to a nonzero exact singular value. So the exact rank at the truth is at least 624, and $r_{\rm gen}\ge624$.

**$r_{\rm gen}\le624$.** The gauge transformations do not change any probability, so their infinitesimal generators are tangent vectors in $\ker J_T$. Wherever 191 of them are linearly independent, $\operatorname{rank}J_T\le815 - 191 = 624$. The generators are linear in $\theta$: for example Kraus mixing with Hermitian generator $H$ gives $\delta K_{gr} = i\sum_s H_{rs}K_{gs}$. So the $928\times192$ matrix $G(\theta)$ of all generators (README 4.1, with the global phase counted twice on purpose) has polynomial entries, and A.3–A.5 apply to it as well. At the truth, $\operatorname{rank}G = 191$: $\sigma_{191}/\sigma_{\max} = 6.8\times10^{-4}$ and $\sigma_{192}/\sigma_{\max} = 1.7\times10^{-16}$. The generators are tangent ($\max\lvert Dh\,G\rvert = 1.5\times10^{-15}$) and change no probability ($\lVert JG\rVert/(\lVert J\rVert\lVert G\rVert) = 6\times10^{-17}$). [Numerical: 12] So $\operatorname{rank}G = 191$ at almost every point. Two sets of measure zero do not cover $\mathcal M$, so some point has both $\operatorname{rank}J_T = r_{\rm gen}$ and 191 independent gauge directions. There, $r_{\rm gen}\le624$.

Together, $r_{\rm gen} = 624$.

**What this says about designs: nothing.** The argument is for one fixed design. Each design gives a different polynomial matrix $X(\theta)$ with its own generic rank. Designs form a finite set, so "almost every design" has no meaning. That all 30 random designs per $N$ reached $\min(3N, 624)$ (README 5.2) is an empirical result.

### A.7 The exceptional points

The rank can be lower only on the closed measure-zero set $Z$. Examples:

- **Exactly unitary gates** (rank 137 at $N = 400$). A unitary gate has one nonzero Kraus operator, $K_{g1} = U$, and $K_{g2} = K_{g3} = K_{g4} = 0$. The channel depends on each $K_{gr}$ only through $K_{gr}XK_{gr}^\dagger$, which is quadratic. Moving $K_{gr}$ from $0$ to $tV$ changes the channel by $t^2\,VXV^\dagger$, whose derivative at $t=0$ is zero. These moves are tangent: the constraint differential $K_g^\dagger\delta K_g + \delta K_g^\dagger K_g = U^\dagger\delta K_{g1} + \delta K_{g1}^\dagger U$ does not involve $\delta K_{g2},\delta K_{g3},\delta K_{g4}$. So $3\times32 = 96$ real tangent directions per gate lie in $\ker J_T$.
- **Maximum-likelihood estimates** (rank 599–620). This is the same mechanism. Some Kraus components of the estimates are nearly zero (README 5.4), so the derivatives along them nearly vanish.
- Exact relations between gates; zero probabilities.

These are first-order statements. At a unitary gate the directions above *do* change the probabilities, at order $t^2$. A rank drop at a special point therefore does not prove that the gate set is unidentifiable there. The equivalence between full rank and local identifiability [Source: Rothenberg 1971] assumes that the rank is constant in a neighbourhood.

---

## B. The Riemannian Hessian in the tangent basis

Supports README Section 4.4.

**Notation map to Boumal 2023, Sec. 7.7 (checked).**

| README | Boumal |
|---|---|
| $\mu$ | $\lambda(x)$ |
| $\nabla\bar f$, $\nabla^2\bar f$ | $\operatorname{grad}\bar f(x)$, $\operatorname{Hess}\bar f(x)$ |
| $Dh^\top$ | $Dh(x)^*$ (the adjoint) |
| rows of $Dh$ | $\operatorname{grad}h_i(x)$ |
| $P = TT^\top$ | $\operatorname{Proj}_x$ |

### B.1 Normal space and the multipliers $\mu$

1. **The normal space is spanned by the constraint gradients.** For any matrix, $(\ker Dh)^\perp = \operatorname{range}Dh^\top$ (fundamental theorem of linear algebra). So $N_\theta\mathcal M = (T_\theta\mathcal M)^\perp = \operatorname{range}Dh^\top$ [Source: Boumal 2023, eq. (7.73) (checked)].
2. **Every vector splits into tangent and normal parts.** $\mathbb R^{928} = T_\theta\mathcal M\oplus N_\theta\mathcal M$, so for any $g$ there is a $\mu$ with $g = Pg + Dh^\top\mu$ (Boumal eq. 7.74).
3. **The normal part is a least-squares fit.** $Dh^\top\mu$ is the orthogonal projection of $g$ onto $\operatorname{range}Dh^\top$, i.e. the point of that subspace closest to $g$:

$$\mu = \arg\min_\mu\lVert g - Dh^\top\mu\rVert^2 = (Dh^\top)^+g .$$

   Boumal writes exactly this in eq. (7.75) for a general vector $v$ and coefficients $\alpha$. Eq. (7.78) applies it to $v = \operatorname{grad}\bar f(x)$ and calls the result $\lambda(x)$. The normal equations are $Dh\,Dh^\top\mu = Dh\,g$, so for full row rank $\mu = (Dh\,Dh^\top)^{-1}Dh\,g$.
4. **The Riemannian gradient** is $\operatorname{grad}f = P\nabla\bar f = \nabla\bar f - Dh^\top\mu$ [Source: Boumal 2023, Prop. 3.61 and eq. (7.77) (checked)].
5. **These are Lagrange multipliers.** At a constrained critical point, $\operatorname{grad}f = 0$, so $\nabla\bar f = \sum_j\mu_j\nabla h_j$ exactly. That is the first-order condition for the Lagrangian $\bar f(\theta) - \mu^\top h(\theta)$ [Source: Boumal 2023, Exercise 7.10 (checked); Nocedal & Wright 2006, Ch. 12]. Away from critical points no $\mu$ satisfies this exactly, and the least-squares $\mu$ is the best fit.

**Redundant constraints.** Our $Dh$ has 225 rows and rank 113, so $Dh\,Dh^\top$ is singular and $\mu$ is not unique. $Dh^\top\mu$ is unique, because it is a projection. $\sum_j\mu_j\nabla^2h_j$ is also unique. Two solutions differ by some $\delta\mu$ with $Dh^\top\delta\mu = 0$. Since $\operatorname{rank}Dh = 113$ everywhere on $\mathcal M$, these $\delta\mu$ are spanned by the 112 structural relations (an output that is identically zero, or a copy of another one). Those relations hold for all $\theta$, so the Hessians $\nabla^2h_j$ satisfy them too, and $\sum_j\delta\mu_j\nabla^2h_j = 0$. The code uses the minimum-norm solution, `np.linalg.lstsq(Dg.T, egrad)`.

### B.2 The Hessian formula

**Tool** [Source: Boumal 2023, Cor. 5.16 (checked)]: on a Riemannian submanifold of a Euclidean space, $\operatorname{Hess}f(x)[u] = \operatorname{Proj}_x\big(D\bar G(x)[u]\big)$, where $\bar G$ is any smooth extension of $\operatorname{grad}f$ to a neighbourhood of $\mathcal M$.

**Derivation** (Boumal's own, leading to eq. 7.79). [Derived here, following Boumal]
1. Extend the gradient by $\bar G(\theta) = \nabla\bar f(\theta) - \sum_j\mu_j(\theta)\nabla h_j(\theta)$. This is defined near $\mathcal M$ (use $h_{113}$ of A.5 so that $Dh$ has full rank there) and equals $\operatorname{grad}f$ on $\mathcal M$.
2. Product rule:

$$D\bar G(\theta)[u] = \nabla^2\bar f\,u - \sum_j\mu_j\nabla^2h_j\,u - \sum_j\big(D\mu_j(\theta)[u]\big)\nabla h_j .$$

3. The last term is a combination of the $\nabla h_j$, so it lies in the normal space and $P$ removes it. Hence, for tangent $u$,

$$\operatorname{Hess}f(\theta)[u] = P\,L\,u,\qquad L = \nabla^2\bar f(\theta) - \sum_j\mu_j\nabla^2h_j(\theta).$$

The derivative of $\mu$ never has to be computed. $L$ is the Hessian of the Lagrangian with respect to $\theta$.

### B.3 Which matrix: $T^\top LT$ or $PLP$?

The Hessian is a linear map from $T_\theta\mathcal M$ to itself. A linear map has a different matrix in each coordinate system.

- **Tangent coordinates (815×815).** In the orthonormal basis $t_1,\dots,t_{815}$, entry $(k,l)$ is $\langle t_k,\operatorname{Hess}f[t_l]\rangle = t_k^\top P\,L\,t_l = t_k^\top L\,t_l$, because $Pt_k = t_k$ and $P$ is symmetric. So the matrix is

$$H = T^\top L\,T .$$

- **Ambient coordinates (928×928).** Boumal's eq. (7.81) writes $\operatorname{Hess}f(x) = \operatorname{Proj}_x\circ(\dots)\circ\operatorname{Proj}_x$, "with the understanding that the linear map … is restricted to $T_x\mathcal M$". As a $928\times928$ matrix this is $PLP$. It acts like the Hessian on tangent vectors and sends normal vectors to zero.

The two are the same map:

$$PLP = TT^\top L\,TT^\top = T\,H\,T^\top .$$

If $Hy = \lambda y$, then $(PLP)(Ty) = THT^\top Ty = THy = \lambda\,Ty$. So $PLP$ has the eigenvalues of $H$, plus 113 zeros along the normal directions. Eq. (7.79) has a projector only on the left because it is applied to a tangent $v$, and $\operatorname{Proj}_x v = v$. `riemannian_hessian_check.py` builds $H$ (`H = T.T @ H_amb @ T`).

**Why the basis must be orthonormal.** With a non-orthonormal tangent basis $B$, the matrix $B^\top LB$ is congruent to $H$. By Sylvester's law of inertia it keeps the rank and the signs of the eigenvalues but not their values.

### B.4 Least squares and Gauss–Newton

For $\bar f(\theta) = \tfrac12\lVert p(\theta) - y\rVert^2$ with residual $r = p - y$: $\nabla\bar f = J^\top r$ and $\nabla^2\bar f = J^\top J + \sum_k r_k\nabla^2p_k$. Then

$$H = J_T^\top J_T + T^\top\Big(\sum_k r_k\nabla^2p_k - \sum_j\mu_j\nabla^2h_j\Big)T .$$

With exact data at the truth, $r = 0$. Then $\nabla\bar f = 0$, so $\mu = 0$ (the least-squares fit of the zero vector), and $H = J_T^\top J_T$. With sampled data both extra terms are nonzero and depend on the data.

---

## C. Fisher information on the tangent space

Supports README Section 4.5.

**What comes from where.** The definition of the Fisher information through directional derivatives of the log-likelihood along tangent vectors, and the statement that on a submanifold it is the restriction to the tangent space, are in Boumal 2013, eqs. (2)–(3) and Sec. II (checked). The multinomial calculation (C.4–C.7) is not in Boumal. It is the standard calculation for our measurement model, written out here.

### C.1 The statistical model

Each circuit $c$ is run $m$ times ("shots"). Each shot gives one of 4 outcomes. If the true gate set is $\theta$, shots are independent and give outcome $i$ with probability $p_{ci}(\theta)$. The counts $n_c = (n_{c1},\dots,n_{c4})$, with $\sum_in_{ci} = m$, are multinomial, and different circuits are independent:

$$P(n\mid\theta) = \prod_c\frac{m!}{n_{c1}!\cdots n_{c4}!}\prod_i p_{ci}(\theta)^{n_{ci}} .$$

The log-likelihood is

$$\ell(\theta) = \log P(n\mid\theta) = \sum_{c,i}n_{ci}\log p_{ci}(\theta) + \text{const},$$

where the constant $\sum_c\log\frac{m!}{\prod_in_{ci}!}$ does not depend on $\theta$ and drops out of every derivative. For fixed data, $\ell$ says how well each candidate gate set explains the data. For fixed $\theta$, it is random through $n$.

### C.2 Coordinates on the manifold, and why $T$ appears

The Fisher information and the Cramér–Rao bound are defined with derivatives with respect to parameters that can be varied independently [Source: Kay 1993, Ch. 3]. The 928 coordinates of $\theta$ cannot. Off $\mathcal M$, $p(\theta)$ is not a probability distribution: for example $\sum_ip_{ci}\neq1$ when $\sum_iA_i^\dagger A_i\ne I$. $\mathcal M$ has dimension 815, so we need 815 independent coordinates. We build them in four steps.

1. At $\theta$, the tangent space is an 815-dimensional subspace of $\mathbb R^{928}$ with orthonormal basis $t_1,\dots,t_{815}$, the columns of $T$.
2. A vector $z\in\mathbb R^{815}$ names the tangent vector $Tz = \sum_kz_kt_k$. **This is the only job of $T$:** it turns 815 coordinates into a 928-dimensional direction that is tangent to $\mathcal M$. Because $T^\top T = I$, $\lVert Tz\rVert = \lVert z\rVert$, so $z$ measures Euclidean distances in $(K, A, B)$ correctly.
3. The straight step $\theta + Tz$ is tangent to $\mathcal M$ but leaves it at second order (README 4.2: the constraint violation scales as $\varepsilon^2$). A *retraction* $R_\theta$ maps it back onto $\mathcal M$. By definition $R_\theta(0) = \theta$ and the derivative of $R_\theta$ at $0$ is the identity [Source: Boumal 2023, Def. 3.47 (checked)]. For one Stiefel factor, the polar retraction is $R_X(V) = (X+V)(I + V^\dagger V)^{-1/2}$ [Source: Boumal 2023, Sec. 7.3, eq. (7.24) (checked); the book states the real case, and for complex matrices the transpose becomes the conjugate transpose].
4. Define $\theta(z) = R_\theta(Tz)$. Then $\theta(z)\in\mathcal M$ for all $z$, $\theta(0) = \theta$, and

$$\theta(z) = \theta + Tz + O(\lVert z\rVert^2).$$

Near $z = 0$, the map $z\mapsto\theta(z)$ is a one-to-one smooth parametrization of a neighbourhood of $\theta$ in $\mathcal M$ (a chart), by the inverse function theorem, since its derivative $T$ is injective. In these coordinates, $z\mapsto P(n\mid\theta(z))$ is an ordinary statistical model with 815 free real parameters, and the textbook definitions apply.

### C.3 Derivative of the probabilities: exact, not approximate

By the chain rule, at $z = 0$:

$$\frac{\partial p_{ci}(\theta(z))}{\partial z_k}\Big|_{z=0} = \sum_{a=1}^{928}J_{ci,a}\,\frac{\partial\theta_a}{\partial z_k}\Big|_{z=0} = \sum_aJ_{ci,a}T_{a,k} = (J_T)_{ci,k},$$

because $\partial\theta/\partial z\rvert_{z=0} = T$ by step 4. **This is exact.** The $O(\lVert z\rVert^2)$ part of the retraction changes second derivatives at $z = 0$, and first derivatives only away from $z = 0$. Everything below uses first derivatives at $z = 0$ only, so the choice of retraction never matters.

### C.4 The score

**Definition** [Source: Kay 1993, Ch. 3; Boumal 2013, eq. (2) (checked)]. The score is the gradient of the log-likelihood with respect to the parameters, at the parameter value of interest (here $z = 0$, the true gate set):

$$s_k = \frac{\partial\,\ell(\theta(z))}{\partial z_k}\Big|_{z=0},\qquad k = 1,\dots,815 .$$

In Boumal's notation, $s_k = DL(\theta)[t_k]$, the directional derivative of the log-likelihood along the tangent vector $t_k$.

**Meaning.**
- $s_k > 0$ means that the data actually observed would become more likely if the gate set moved slightly along $+t_k$.
- The score is random: each dataset gives a different one.
- Under the true $\theta$ it fluctuates around zero (C.5).
- The larger its fluctuations along a direction, the more sharply the likelihood depends on that direction, so the more the data say about it. The Fisher information is the covariance of the score (C.7).

**Formula.** Use $\partial\log p/\partial z = (1/p)\,\partial p/\partial z$ and C.3:

$$s_k = \sum_{c,i}n_{ci}\,\frac{1}{p_{ci}}\,\frac{\partial p_{ci}}{\partial z_k} = \sum_{c,i}n_{ci}\,\frac{(J_T)_{ci,k}}{p_{ci}} .$$

### C.5 The score has mean zero

Each of the $m$ shots gives outcome $i$ with probability $p_{ci}$, so $\mathbb E[n_{ci}] = mp_{ci}$. By linearity,

$$\mathbb E[s_k] = \sum_{c,i}mp_{ci}\frac{(J_T)_{ci,k}}{p_{ci}} = m\sum_c\sum_i(J_T)_{ci,k} .$$

For each circuit,

$$\sum_i(J_T)_{ci,k} = \frac{\partial}{\partial z_k}\sum_ip_{ci}(\theta(z))\Big|_{z=0} = \frac{\partial}{\partial z_k}1 = 0,$$

because $\theta(z)\in\mathcal M$ for all $z$, and every gate set on $\mathcal M$ gives probabilities that sum to 1. **This is where staying on $\mathcal M$ matters.** Along a direction that leaves $\mathcal M$ (for example scaling all $A_i$), $\sum_ip_{ci}$ changes, and the analogous quantity would not have mean zero. In matrix form: $\mathbf 1^\top J_T^{(c)} = 0$ for the $4\times815$ block $J_T^{(c)}$ of every circuit.

### C.6 Covariance of the counts

Take one circuit and drop the index $c$. Let $x_{t,i} = 1$ if shot $t$ gives outcome $i$ and $0$ otherwise, so $n_i = \sum_{t=1}^mx_{t,i}$.

1. $\mathbb E[x_{t,i}] = p_i$.
2. A shot gives exactly one outcome, so $x_{t,i}x_{t,j} = \delta_{ij}x_{t,i}$ and $\mathbb E[x_{t,i}x_{t,j}] = \delta_{ij}p_i$.
3. Therefore $\mathrm{Cov}(x_{t,i},x_{t,j}) = \delta_{ij}p_i - p_ip_j$.
4. Shots are independent, so covariances add over the $m$ shots:

$$\mathrm{Cov}(n_{ci},n_{cj}) = m\,(p_{ci}\delta_{ij} - p_{ci}p_{cj}).$$

5. Different circuits are independent: $\mathrm{Cov}(n_{ci},n_{c'j}) = 0$ for $c\ne c'$.

The negative off-diagonal term says that counts of one circuit compete: they must add up to $m$.

### C.7 The Fisher information

**Definition** [Source: Kay 1993, Ch. 3; Boumal 2013, eq. (3) (checked)]: $F_{kl} = \mathbb E[s_ks_l]$. Since $\mathbb E[s] = 0$ (C.5), this is also $\mathrm{Cov}(s_k,s_l)$.

Write $s_k = \sum_{c,i}a_{ci,k}\,n_{ci}$ with the constants $a_{ci,k} = (J_T)_{ci,k}/p_{ci}$. They depend on $\theta$ but not on the data. Then, line by line:

$$
\begin{aligned}
F_{kl} &= \sum_{c,i}\sum_{c',j}a_{ci,k}\,a_{c'j,l}\,\mathrm{Cov}(n_{ci},n_{c'j}) && \text{(covariance is bilinear)}\\
&= \sum_c\sum_{i,j}a_{ci,k}\,a_{cj,l}\;m\,(p_{ci}\delta_{ij} - p_{ci}p_{cj}) && \text{(C.6; only } c = c' \text{ survives)}\\
&= m\sum_c\Big[\sum_ia_{ci,k}a_{ci,l}\,p_{ci} - \Big(\sum_ia_{ci,k}p_{ci}\Big)\Big(\sum_ja_{cj,l}p_{cj}\Big)\Big] && \text{(split the two terms)}\\
&= m\sum_c\Big[\sum_i\frac{(J_T)_{ci,k}(J_T)_{ci,l}}{p_{ci}} - \Big(\sum_i(J_T)_{ci,k}\Big)\Big(\sum_j(J_T)_{cj,l}\Big)\Big] && (a_{ci,k}\,p_{ci} = (J_T)_{ci,k})\\
&= m\sum_{c,i}\frac{(J_T)_{ci,k}(J_T)_{ci,l}}{p_{ci}} && \text{(C.5: per-circuit sums vanish)}\\
&= m\,\big(J_T^\top\operatorname{diag}(p)^{-1}J_T\big)_{kl}.
\end{aligned}
$$

**Second route: the expected Hessian.** Under the usual regularity conditions, $F = -\mathbb E[\partial^2\ell/\partial z\,\partial z^\top]$ [Source: Kay 1993, Ch. 3]. Directly:

$$-\mathbb E\Big[\frac{\partial^2\ell}{\partial z_k\partial z_l}\Big] = -\sum_{c,i}mp_{ci}\Big[\frac{\partial_k\partial_lp_{ci}}{p_{ci}} - \frac{\partial_kp_{ci}\,\partial_lp_{ci}}{p_{ci}^2}\Big] = -m\sum_c\partial_k\partial_l\underbrace{\sum_ip_{ci}}_{=1} + m\big(J_T^\top\operatorname{diag}(p)^{-1}J_T\big)_{kl}.$$

The second derivatives $\partial_k\partial_lp_{ci}$ do contain the second-order part of the retraction, but only through $\sum_ip_{ci}\equiv1$, whose derivatives vanish. So the two routes agree for every retraction.

### C.8 Check: one circuit, two outcomes, one parameter

Let the outcome probabilities be $(q(z), 1 - q(z))$. Then $J_T = (q', -q')^\top$ and

$$F = m\Big(\frac{q'^2}{q} + \frac{q'^2}{1-q}\Big) = \frac{m\,q'^2}{q(1-q)} .$$

If the parameter is $q$ itself ($q' = 1$), then $F = m/(q(1-q))$, and $1/F = q(1-q)/m$ is the variance of the observed frequency, the familiar binomial result. The observed frequency is unbiased and attains this bound.

### C.9 What is exact and what is approximate

- $F$ is exact. It involves only first derivatives at the true point, and no Taylor approximation is made.
- The Cramér–Rao inequality (section D) is exact for any number of shots, for unbiased estimators.
- Two statements are approximate:
  - that the maximum-likelihood estimate attains the bound. This holds only asymptotically ($m\to\infty$), when the errors are small enough for $p(\theta(z))$ to be linear in $z$ over their size;
  - linearized readings such as $p(\hat\theta) - p(\theta^*)\approx J_Tz$ (F.4).

  This is where "first order" is an approximation. It fails for weakly determined directions (README 5.5).

### C.10 The choices made do not matter

- **Retraction.** Only $\partial\theta/\partial z\rvert_0 = T$ entered (C.3), and second-order terms cancel in the second route (C.7).
- **Orthonormal basis.** Another orthonormal basis is $T' = TQ$ with $Q$ an orthogonal $815\times815$ matrix. Then $J_T' = J_TQ$ and $F' = Q^\top FQ$: the same eigenvalues, with eigenvectors $v_k' = Q^\top v_k$ that describe the same tangent vectors, $T'v_k' = Tv_k$.
- **Metric.** The eigenvalues do depend on the metric: here the Euclidean metric of $(K, A, B)$, encoded in the orthonormality of $T$ (README 4.2).

### C.11 Relation to Boumal 2013

Boumal's setting is a parameter space $\mathcal P$ with a submanifold $\bar{\mathcal P}$; here $\bar{\mathcal P} = \mathcal M$.
- Eq. (2) defines the Fisher information as a bilinear form on tangent vectors, $F[u,v] = \mathbb E\{DL(\theta)[u]\cdot DL(\theta)[v]\}$. Eq. (3) gives its matrix in an orthonormal basis.
- Sec. II says that on a submanifold the Fisher information "is simply the restriction" to the tangent space.
- Our $F$ is that matrix in the basis $t_1,\dots,t_{815}$ (Boumal's $\bar e$).
- Boumal's Theorem 2 (eq. 8) writes the resulting bound in ambient coordinates as $(P_eF_eP_e)^\dagger$, with $P_e = TT^\top$ in our notation. That equals $TF^{-1}T^\top$ (his eq. 12), the constrained bound of Gorman & Hero quoted in README 4.6.

---

## D. Scalar bound and curvature terms

Supports README Section 4.6 ("Singular $F$" and "Curvature terms").

### D.1 Statement

Let $\varphi:\mathcal M\to\mathbb R$ be smooth and gauge invariant, for example the predicted probability of a circuit. Let $\hat\varphi$ be any function of the data. Suppose $\hat\varphi$ is unbiased near the truth:

$$\mathbb E_{\theta(z)}[\hat\varphi] = \varphi(\theta(z))\quad\text{for all } z \text{ in a neighbourhood of } 0 .$$

Let $\mathbf d = \nabla_z\varphi(\theta(z))\rvert_{z=0} = T^\top\nabla\varphi(\theta)$ (chain rule, as in C.3). Then:

- if $\mathbf d\in\operatorname{range}F$: $\mathrm{Var}(\hat\varphi)\ge\mathbf d^\top F^+\mathbf d$;
- if $\mathbf d\notin\operatorname{range}F$: no such unbiased estimator exists.

With $F$ invertible this is the classical bound for a function of the parameters [Source: Kay 1993, Ch. 3]. The singular case is from Stoica & Marzetta 2001.

### D.2 Proof

[Derived here]

1. **Unbiasedness gives a covariance.** Differentiate the unbiasedness condition at $z = 0$. The left side gives

$$\nabla_z\mathbb E_z[\hat\varphi] = \sum_n\hat\varphi(n)\,\nabla_zP(n\mid z) = \sum_n\hat\varphi(n)\,P(n\mid z)\,\nabla_z\log P(n\mid z) = \mathbb E[\hat\varphi\,s].$$

   Exchanging derivative and sum is allowed because there are finitely many possible count tables. The right side gives $\mathbf d$. Since $\mathbb E[s] = 0$, $\mathbb E[\hat\varphi s] = \mathrm{Cov}(\hat\varphi, s)$. So

$$\mathrm{Cov}(\hat\varphi, s) = \mathbf d .$$

2. **Cauchy–Schwarz.** For any $a\in\mathbb R^{815}$, $\mathrm{Cov}(\hat\varphi, a^\top s)^2\le\mathrm{Var}(\hat\varphi)\,\mathrm{Var}(a^\top s)$, i.e.

$$(a^\top\mathbf d)^2\le\mathrm{Var}(\hat\varphi)\;a^\top Fa .$$

3. **Choose $a$.**
   - If $\mathbf d\in\operatorname{range}F$, take $a = F^+\mathbf d$. Then $a^\top\mathbf d = \mathbf d^\top F^+\mathbf d$ and $a^\top Fa = \mathbf d^\top F^+FF^+\mathbf d = \mathbf d^\top F^+\mathbf d$ (a property of the pseudo-inverse). Dividing gives $\mathrm{Var}(\hat\varphi)\ge\mathbf d^\top F^+\mathbf d$.
   - If $\mathbf d\notin\operatorname{range}F = (\ker F)^\perp$, there is $a\in\ker F$ with $a^\top\mathbf d\ne0$. Then $\mathrm{Var}(a^\top s) = a^\top Fa = 0$, so $a^\top s = 0$ with probability 1. Then $\mathrm{Cov}(\hat\varphi, a^\top s) = 0\neq a^\top\mathbf d$, which contradicts step 1. ∎

**When is $\mathbf d\in\operatorname{range}F$?** $\ker F = \ker J_T$, because $F = mW^\top W$ with $W = \operatorname{diag}(p)^{-1/2}J_T$. At rank 624, $\ker J_T$ is spanned by the 191 gauge directions (A.6). A gauge-invariant $\varphi$ does not change along them, so $\mathbf d$ is orthogonal to $\ker F$, i.e. $\mathbf d\in\operatorname{range}F$. In the eigenbasis, $F^+ = \sum_{\lambda_k>0}v_kv_k^\top/\lambda_k$, so $\mathbf d^\top F^+\mathbf d = \sum_k(\mathbf d\cdot v_k)^2/\lambda_k$.

### D.3 What the proof did not need

No logarithmic map, no geodesics, no curvature, and no assumption of many shots. The error $\hat\varphi - \varphi(\theta)$ is a real number, and the real line is flat. The manifold enters only through the coordinates $z\mapsto\theta(z)$, and only through their first derivative $T$ at $z = 0$ (C.3).

### D.4 Where the curvature terms of the intrinsic bounds come from

[Source: Boumal 2013, Secs. I–IV (checked); Smith 2005]

- **The object estimated.** Intrinsic bounds are about the estimated *point* $\hat\theta$ (for the quotient, its gauge class).
- **The error.** It is the tangent vector $\xi = \operatorname{Log}_\theta(\hat\theta)$, the initial velocity of the geodesic from $\theta$ to $\hat\theta$. For the quotient it is the horizontal lift of $\operatorname{Log}_{[\theta]}([\hat\theta])$ (eq. 20).
- **Unbiasedness and covariance.** Unbiased means $\mathbb E[\xi] = 0$ (eq. 4), and the covariance is $C = \mathbb E[\xi\xi^\top]$ (eq. 5).
- **The bounds.** Theorem 3 (eq. 22) gives $C\succeq F^\dagger$ + curvature terms. Theorem 5 (eq. 51) writes them out as $C\succeq F^\dagger - \frac13\big(R_m(F^\dagger)F^\dagger + F^\dagger R_m(F^\dagger)\big)$, where $R_m$ is linear and built from the Riemann curvature tensor of the quotient (eqs. 46–50).

Repeating D.2 with $\xi$ in place of $\hat\varphi - \varphi$, step 1 must differentiate $\mathbb E[\operatorname{Log}_\theta\hat\theta] = 0$ with respect to $\theta$. In flat space $\partial(\hat\theta-\theta)/\partial\theta = -I$. On a curved manifold, Smith (2005, p. 1615), quoted in Boumal 2013, Sec. IV, writes that the covariant derivative of $\operatorname{Log}_\theta(\hat\theta)$ is "$-I$ + second- and higher-order terms involving the manifold's sectional and Riemannian curvatures". Averaged over the estimator, these terms give $R_m$.

$R_m(F^\dagger)$ is linear in $F^\dagger$ and in the curvature. So the correction relative to $F^\dagger$ is of order (sectional curvature) × (variance), roughly $K/\lambda$. Boumal's eq. (32) turns this into a condition: the terms are negligible when

$$\operatorname{dist}(\theta,\hat\theta)\ll\frac1{\sqrt{K_{\max}}},$$

where $K_{\max}$ bounds the absolute sectional curvature at $\theta$. The "large signal-to-noise ratio" provision (Boumal 2013, pp. 4–5) has a second reason as well: $\operatorname{Log}$ is defined only near $\theta$.

**Relation to the scalar bound.** If $\hat\theta$ is intrinsically unbiased and the errors are small, then $\varphi(\hat\theta) - \varphi(\theta)\approx\langle\nabla\varphi,\xi\rangle$. So $\mathrm{Var}(\varphi(\hat\theta))\approx\mathbf d^\top C\mathbf d\ge\mathbf d^\top F^\dagger\mathbf d$ + curvature corrections, with the same leading term as D.1. The assumptions differ: $\mathbb E[\xi] = 0$ versus $\mathbb E[\hat\varphi] = \varphi(\theta)$. On a curved space neither implies the other.

### D.5 How large is the curvature here? (order of magnitude)

[Derived here; heuristic]

Consider one gate. Put its Kraus operators, vectorized, as the columns of $Y\in\mathbb C^{16\times4}$. Kraus mixing is $Y\to YV^\top$ with $V$ unitary, and $YY^\dagger$ is the (unnormalized) Choi matrix. So the gauge quotient contains, for each gate, a manifold of rank-4 Choi matrices, with the metric inherited from the Euclidean metric on $Y$.

The real analogue is $\mathbb R^{n\times p}_*$ modulo $O(p)$, the positive-semidefinite matrices of rank $p$ with the Bures–Wasserstein metric. At a point whose factor $Y$ has singular values $\sigma_1\ge\dots\ge\sigma_p$, its largest sectional curvature is

$$K_{\max} = \frac{3}{\sigma_{p-1}^2 + \sigma_p^2},$$

which diverges as the two smallest singular values go to zero [Source: Massart, Hendrickx & Absil 2019, Prop. 2 (checked)].

For our gates, $\sum_r\sigma_r^2 = \sum_r\lVert K_r\rVert_F^2 = \operatorname{tr}I_4 = 4$. So $\sigma_r^2$ is 4 times the normalized Choi eigenvalue. For the gate quoted in README 5.1, $\sigma_3^2+\sigma_4^2\approx4\times(2.2+1.1)\times10^{-4}\approx1.3\times10^{-3}$. That gives $K_{\max}\approx2.3\times10^3$ and $1/\sqrt{K_{\max}}\approx0.02$.

**Caveats.** This uses the real case instead of the complex one. It ignores the trace-preservation constraint and the other factors of the quotient (POVM, state, Hilbert-space gauge).

**Reading.** If the intrinsic bound were used, its curvature terms would be negligible for directions with Cramér–Rao standard deviation well below 0.02, roughly the first $\sim135$ ($1/\sqrt{\lambda_k} < 0.01$), and not beyond. That is the same range where the estimator stops attaining the bound (README 5.5). It fits mechanism 2 there: the curvature is large near the positivity boundary, where Kraus operators are small. No number in the README uses the intrinsic bound, so none is affected.

---

## E. The van Trees bound

Supports README Section 4.6, "Bayesian bound".

### E.1 Statement

**Setting.**
- Parameters are tangent coordinates $z\in\mathbb R^D$ ($D = 815$) around a reference gate set (the truth in our scripts).
- A *prior* is a probability density $\pi(z)$ on $\mathbb R^D$. It must be smooth, positive, and decay fast enough at infinity that the boundary terms below vanish; a Gaussian is fine.
- The joint experiment: draw $z\sim\pi$, then draw data $n\sim P(n\mid\theta(z))$.

**Statement.** For **any** estimator $\hat\varphi(n)$,

$$\mathbb E_\pi\,\mathbb E_z\big[(\hat\varphi - \varphi(\theta(z)))^2\big]\;\ge\;\bar{\mathbf d}^\top\big(\mathbb E_\pi[F(z)] + F_\pi\big)^{-1}\bar{\mathbf d},$$

where
- $F(z)$ is the Fisher information of the data at $\theta(z)$ (C.7);
- $F_\pi = \mathbb E_\pi[\nabla\log\pi\,\nabla\log\pi^\top]$ is the information carried by the prior;
- $\bar{\mathbf d} = \mathbb E_\pi[\nabla_z\varphi(\theta(z))]$.

[Source: Van Trees 1968; Gill & Levit 1995. Gill & Levit give a general form with a vector function $C(\theta)$; the constant vector $c$ below is a special case.]

### E.2 Proof

[Derived here] Let $q(n,z) = P(n\mid z)\pi(z)$ be the joint density. Fix $c\in\mathbb R^D$ and define

$$U = c^\top\nabla_z\log q(n,z) = c^\top s(n;z) + c^\top\nabla\log\pi(z).$$

1. **Integration by parts replaces unbiasedness.** Using $q\nabla\log q = \nabla q$:

$$\mathbb E[(\hat\varphi - \varphi)U] = \sum_n\int(\hat\varphi(n) - \varphi(z))\,c^\top\nabla_zq(n,z)\,dz = -\sum_n\int c^\top\nabla_z\big(\hat\varphi(n) - \varphi(z)\big)\,q(n,z)\,dz = c^\top\bar{\mathbf d}.$$

   The middle step integrates by parts in $z$. The boundary terms vanish because $q\to0$ at infinity. $\hat\varphi(n)$ does not depend on $z$, so its gradient is zero. **This is why every estimator gives the same answer.** In D.2, unbiasedness was needed to get $\mathrm{Cov}(\hat\varphi, s) = \mathbf d$ at one fixed $\theta$. Here, averaging over $z$ does that job for any $\hat\varphi$.
2. **Information adds.** Expanding the square,

$$\mathbb E[U^2] = c^\top\mathbb E_\pi\big[\mathbb E_z[ss^\top]\big]c + 2\,\mathbb E_\pi\big[c^\top\mathbb E_z[s]\;c^\top\nabla\log\pi\big] + c^\top F_\pi c = c^\top\big(\mathbb E_\pi[F] + F_\pi\big)c,$$

   because $\mathbb E_z[s] = 0$ (C.5).
3. **Cauchy–Schwarz.** $(c^\top\bar{\mathbf d})^2 = \mathbb E[(\hat\varphi-\varphi)U]^2\le\mathbb E[(\hat\varphi-\varphi)^2]\,\mathbb E[U^2]$. So for every $c$,

$$\mathbb E[(\hat\varphi - \varphi)^2]\ge\frac{(c^\top\bar{\mathbf d})^2}{c^\top(\mathbb E_\pi[F] + F_\pi)c}.$$

   The right side is largest for $c = (\mathbb E_\pi[F] + F_\pi)^{-1}\bar{\mathbf d}$, which gives the bound. ∎

### E.3 Gaussian prior: the formula in the scripts

For $\pi(z)\propto\exp(-\lVert z\rVert^2/2\tau^2)$:

$$\nabla\log\pi = -z/\tau^2,\qquad F_\pi = \mathbb E[zz^\top]/\tau^4 = I/\tau^2 .$$

**Our approximation:** $\mathbb E_\pi[F]\approx F(0)$ and $\bar{\mathbf d}\approx\mathbf d(0)$. With $F = \sum_k\lambda_kv_kv_k^\top$ over a complete orthonormal eigenbasis (the gauge directions have $\lambda = 0$):

$$(F + I/\tau^2)^{-1} = \sum_k\frac{v_kv_k^\top}{\lambda_k + 1/\tau^2},\qquad \mathbf d^\top(F + I/\tau^2)^{-1}\mathbf d = \sum_k\frac{(\mathbf d\cdot v_k)^2}{\lambda_k + 1/\tau^2}.$$

The gauge terms vanish because $\mathbf d$ is orthogonal to the gauge directions. This is `van_trees_var` in script 09.

**Limits of the approximation.**
- *Distance.* A Gaussian sample in 815 dimensions lies at distance $\approx\tau\sqrt{815}\approx29\tau$ from the centre. Each Kraus operator (32 real numbers) moves by $\approx\tau\sqrt{32}\approx6\tau$. At $\tau = 0.01$ that is already larger than the smallest true Kraus operators ($\approx0.02$), so prior samples differ from the truth in their weak components.
- *Why it is still plausible.* The strong directions' $\lambda_k$ hardly change when weak components change (README 5.4: within 2–4% at the estimates for $k\le100$). The weak directions are dominated by $1/\tau^2$ anyway.
- *Status.* It has not been checked. The exact bound averages $F(z)$ and $\nabla\varphi$ over samples $z\sim\pi$.
- *Chart.* $\theta(z)$ via the polar retraction is defined for every $z$, so the inequality itself holds for every $\tau$. For $\tau\gtrsim0.03$ the prior spreads over a large part of the compact manifold, and its meaning as "plausible deviations from the reference" weakens.

---

## F. The per-direction test statistic

Supports README Section 5.5.

### F.1 What we want to test, and why not in parameter space

The bound says that an unbiased estimator must scatter along $v_k$ with standard deviation at least $1/\sqrt{\lambda_k}$. We want to know, for each $k$, whether the 40 maximum-likelihood estimates scatter that much. Measuring the scatter of $\hat\theta - \theta^*$ along $v_k$ directly runs into two problems.

1. **Gauge.** $\hat\theta$ sits at an arbitrary point of its gauge orbit. Moving it along the orbit changes its component along $v_k$, because $v_k$ is orthogonal to the gauge directions only at $\theta^*$ and only to first order.
2. **Coordinates.** "The component of $\hat\theta - \theta^*$ along $v_k$" needs tangent coordinates of $\hat\theta$, i.e. an inverse retraction or a logarithm, plus a gauge fixing.

Predicted probabilities avoid both problems. They are gauge invariant and computed directly from $\hat\theta$.

### F.2 The simplest case: one circuit, two outcomes, one direction

Let the outcome probabilities be $(q, 1-q)$, and let $q'$ be the derivative of $q$ along the direction.

1. **Jacobian.** $J_T = (q', -q')^\top$.
2. **Whitened Jacobian.** $W = \operatorname{diag}(p)^{-1/2}J_T = (q'/\sqrt q,\; -q'/\sqrt{1-q})^\top$.
3. **Singular value and Fisher information.** $s = \lVert W\rVert = \lvert q'\rvert\sqrt{1/q + 1/(1-q)} = \lvert q'\rvert/\sqrt{q(1-q)}$, so $\lambda = ms^2 = mq'^2/(q(1-q))$, as in C.8.
4. **Left singular vector.** $u = W/s = \operatorname{sign}(q')\,(\sqrt{1-q},\,-\sqrt q)^\top$.
5. **Prediction error.** $\delta = (\hat q - q,\; -(\hat q - q))^\top$.
6. **The statistic.**

$$e = \sqrt m\,u^\top\operatorname{diag}(p)^{-1/2}\delta = \operatorname{sign}(q')\sqrt m\,(\hat q-q)\Big(\frac{\sqrt{1-q}}{\sqrt q} + \frac{\sqrt q}{\sqrt{1-q}}\Big) = \operatorname{sign}(q')\,\frac{\hat q - q}{\sqrt{q(1-q)/m}} .$$

So **$e$ is the error of the predicted probability divided by its binomial shot-noise standard deviation**: a z-score. Since $\hat q - q\approx q'\times(\text{error along the direction})$, we also have $e\approx\sqrt\lambda\times(\text{error along the direction})$, the error divided by the Cramér–Rao standard deviation $1/\sqrt\lambda$. If the estimate reproduces the observed frequency $f$ (here the maximum-likelihood estimate does, and it is efficient), then $\mathbb E[e^2] = \mathrm{Var}(f)\big/\big(q(1-q)/m\big) = 1$ exactly.

### F.3 The general construction

At the truth, take the thin SVD of the whitened Jacobian of the training circuits, keeping the 624 nonzero singular values:

$$W = \operatorname{diag}(p^*)^{-1/2}J_T = USV^\top .$$

- **Parameter directions.** The columns $v_k\in\mathbb R^{815}$ of $V$ are orthonormal and are the eigenvectors of $F$, with $\lambda_k = ms_k^2$.
- **Data directions.** The columns $u_k\in\mathbb R^{1600}$ of $U$ are orthonormal directions in the space of whitened predictions.
- **The link** is $Wv_k = s_ku_k$. A unit step of the gate set along $v_k$ moves the whitened predictions by $s_k$ along $u_k$ and along no other $u_l$. So $u_k$ is the "fingerprint" of $v_k$ in data space, and the fingerprints of different directions are orthogonal.

Then $e_k$ is computed in three steps:

1. Prediction error: $\delta = p_{\rm train}(\hat\theta) - p_{\rm train}(\theta^*)\in\mathbb R^{1600}$. This is gauge invariant.
2. Whitening: $w = \sqrt m\operatorname{diag}(p^*)^{-1/2}\delta$. Each entry is now in units of the shot-noise scale $\sqrt{p^*/m}$ of that outcome.
3. Read off the fingerprint of $v_k$:

$$e_k = u_k^\top w = \sqrt m\,u_k^\top\operatorname{diag}(p^*)^{-1/2}\big(p_{\rm train}(\hat\theta) - p_{\rm train}(\theta^*)\big).$$

In code (script 09): `e = sqrt(m) * ((P_train - p) / sqrt(p)) @ U`.

### F.4 Reading 1: error along $v_k$ in units of the bound

Let $z$ be tangent coordinates of $\hat\theta$ around $\theta^*$ (after any gauge transformation). To first order, $\delta\approx J_Tz$, so

$$w\approx\sqrt m\,Wz = \sqrt m\,USV^\top z = \sum_l\sqrt m\,s_l\,(v_l^\top z)\,u_l = \sum_l\sqrt{\lambda_l}\,(v_l^\top z)\,u_l .$$

The $u_l$ are orthonormal, so

$$e_k = u_k^\top w\approx\sqrt{\lambda_k}\,(v_k^\top z) = \frac{\text{error along } v_k}{1/\sqrt{\lambda_k}} .$$

A gauge component of $z$ drops out, since $W$ maps it to zero. Exactly, $e_k$ does not depend on the gauge at all, because it uses only predicted probabilities.

### F.5 Reading 2: how much of the shot noise the estimate absorbs

1. **The whitened noise.** The measured frequencies $f_c = n_c/m$ have covariance $(\operatorname{diag}(p_c) - p_cp_c^\top)/m$ (C.6). Let $y = \sqrt m\operatorname{diag}(p^*)^{-1/2}(f - p^*)$. Per circuit,

$$\mathrm{Cov}(y_c) = \operatorname{diag}(p_c)^{-1/2}\big(\operatorname{diag}(p_c) - p_cp_c^\top\big)\operatorname{diag}(p_c)^{-1/2} = I_4 - q_cq_c^\top,\qquad q_c = \sqrt{p_c}\ \text{(entrywise)} .$$

   $q_c$ is a unit vector because $\sum_ip_{ci} = 1$. So $\mathrm{Cov}(y)$ is the orthogonal projector onto the $3N$-dimensional subspace orthogonal to all the $q_c$. In that subspace the noise has unit variance and no correlations ("white"). Along each $q_c$ it is exactly zero, because $f_c$ and $p_c$ both sum to 1.
2. **The $u_k$ lie in the noisy subspace.** Block $c$ of $Wv_k$ is $\operatorname{diag}(p_c)^{-1/2}J_T^{(c)}v_k$, and $q_c^\top\operatorname{diag}(p_c)^{-1/2}J_T^{(c)}v_k = \mathbf 1^\top J_T^{(c)}v_k = 0$ (C.5). So $u_k\perp q_c$ for every $c$, and

$$\mathrm{Var}(u_k^\top y) = 1,\qquad \mathrm{Cov}(u_k^\top y,\,u_l^\top y) = \delta_{kl}.$$

3. **Maximum likelihood is approximately weighted least squares.** For one circuit, with $n_i = mf_i$, expand $\log(p_i/f_i) = \log(1 + (p_i - f_i)/f_i)$ to second order:

$$\sum_in_i\log p_i = \sum_in_i\log f_i + m\sum_i(p_i - f_i) - \frac m2\sum_i\frac{(p_i - f_i)^2}{f_i} + O(\lvert p-f\rvert^3).$$

   The middle term is zero, since both sum to 1. So near the data, maximizing the likelihood means minimizing $\frac m2\sum_{c,i}(p_{ci} - f_{ci})^2/f_{ci}$. For small noise $f\approx p^*$, and this is $\tfrac12\lVert y - w(\theta)\rVert^2$ with $w(\theta) = \sqrt m\operatorname{diag}(p^*)^{-1/2}(p(\theta) - p^*)$.
4. **In a model linear in $z$**, $p(\theta(z)) = p^* + J_Tz$, the predictions $w(z) = \sqrt m\,Wz$ range over $\operatorname{span}\{u_1,\dots,u_{624}\}$. The least-squares fit picks the closest point, the orthogonal projection:

$$w(\hat z) = \sum_{k\le624}(u_k^\top y)\,u_k\quad\Longrightarrow\quad e_k = u_k^\top w(\hat z) = u_k^\top y .$$

   With step 2: $\mathbb E[e_k] = 0$ and $\mathbb E[e_k^2] = 1$. The estimate absorbs all of the shot noise along $u_k$.
5. **Meaning.** $\mathbb E[e_k^2]$ measures how much of the shot noise along $u_k$ the estimate picks up. An estimate that does not respond to the data along $u_k$, for example because it is stuck at a boundary, has $e_k\approx0$ whatever the data.

### F.6 Why the bound is exactly 1

With $u_k$ and $p^*$ held fixed, define

$$\varphi_k(\theta) = \sqrt m\,u_k^\top\operatorname{diag}(p^*)^{-1/2}\big(p_{\rm train}(\theta) - p_{\rm train}(\theta^*)\big).$$

It is gauge invariant (it uses only predicted probabilities), $\varphi_k(\theta^*) = 0$, and $e_k = \varphi_k(\hat\theta)$. So $e_k$ is exactly the error of the plug-in estimator $\varphi_k(\hat\theta)$, with no linearization. Its gradient in tangent coordinates at $\theta^*$ is

$$\mathbf d_{\varphi_k} = \sqrt m\,J_T^\top\operatorname{diag}(p^*)^{-1/2}u_k = \sqrt m\,W^\top u_k = \sqrt m\,VSU^\top u_k = \sqrt m\,s_k\,v_k .$$

By D.1, any estimator that is unbiased for $\varphi_k$ near $\theta^*$ satisfies

$$\mathrm{Var}(e_k)\ge\mathbf d_{\varphi_k}^\top F^+\mathbf d_{\varphi_k} = m\,s_k^2\;v_k^\top F^+v_k = \frac{m\,s_k^2}{\lambda_k} = 1 .$$

### F.7 What the measured mean of $e_k^2$ tells us

The truth is known and $\varphi_k(\theta^*) = 0$, so the script averages $e_k^2$ without subtracting a mean:

$$\overline{e_k^2}\approx\mathbb E[e_k^2] = \mathrm{Var}(e_k) + (\mathbb E[e_k])^2\ \ge\ \mathrm{Var}(e_k).$$

- If the estimator were unbiased for $\varphi_k$ near $\theta^*$, then $\mathbb E[e_k^2]\ge1$ (F.6).
- Observing $\mathbb E[e_k^2]\ll1$ therefore rules out unbiasedness near $\theta^*$. It does not say that $\mathbb E[e_k]\ne0$ at $\theta^*$ itself. It says that the average estimate does not follow $\varphi_k$ when the truth moves. In the biased bound of README 4.6, $\mathrm{Var}\ge(\mathbf d + \mathbf d_b)^\top F^+(\mathbf d + \mathbf d_b)$, and a value near 0 requires $\mathbf d_b\approx-\mathbf d$ along $v_k$.
- **Statistical uncertainty.** If $e_k\sim\mathcal N(0,\sigma^2)$, the average of $e_k^2$ over $n$ datasets has standard deviation $\sigma^2\sqrt{2/n}$, i.e. $\pm0.22$ for $n = 40$ and $\sigma = 1$. Medians over bins of $\sim50$ directions are good to about $\pm0.04$, assuming the directions are roughly independent.
