# Precision of Pauli transfer matrix elements: theory, weak directions and experiment design

Companion to [README.md](README.md), written 2026-11-09 with Claude Code. It collects in one place, for one use case, what README Sections 4.5–4.7, 5.5–5.6 and 6 say in several places:

> For a design with $N$ circuits, $m$ shots per circuit and maximum depth $L_{\max}$, how precisely are the gauge-fixed Pauli transfer matrix elements determined? How do the weakly determined Fisher directions enter? Which designs reach a target error $\delta$?

**Provenance convention** (as in the README): **[Source: …]** literature, marked **(checked)** if compared with the text; **[Derived here]** a calculation written out in this file; **[Numerical: script]** our computation, reproducible with the script named in [Section 7](#7-files).

---

## Contents

0. [Terms and abbreviations](#0-terms-and-abbreviations)
1. [Answers in brief](#1-answers-in-brief)
2. [Derivation of the interval for a Pauli transfer matrix element](#2-derivation-of-the-interval-for-a-pauli-transfer-matrix-element)
3. [How the weak directions enter](#3-how-the-weak-directions-enter)
4. [Scaling with $m$, $N$ and $L_{\max}$](#4-scaling-with-m-n-and-l_max)
5. [Numerical tests](#5-numerical-tests)
6. [Caveats and open questions](#6-caveats-and-open-questions)
7. [Files](#7-files)

---

## 0. Terms and abbreviations

| Term | Meaning |
|---|---|
| PTM | Pauli transfer matrix of gate $g$: $R_{g,ij} = \operatorname{tr}(P_i\,\mathcal G_g(P_j))/d$, with $P_0 = I, P_1,\dots,P_{15}$ the two-qubit Pauli matrices and $d = 4$ |
| LAN | local asymptotic normality: the log-likelihood is approximately quadratic near the truth and the maximum-likelihood estimate is approximately Gaussian with covariance $F^{+}$ ([Section 2.5](#25-why-the-estimate-is-approximately-gaussian-lan)) |
| SPAM | state preparation and measurement |
| gauge fixing $\Pi$ | the Hilbert-space unitary that brings the gate PTMs closest to the target PTMs in Frobenius norm, applied to the gate set (README 6.3) |
| $f$ | one gauge-fixed PTM element, $f = R_{g,ij}\circ\Pi$ |
| $\mathbf d_f$ | gradient of $f$ in the tangent coordinates $z$ (815 numbers) |
| $D$ | the matrix whose rows are $\mathbf d_f^\top$ for the 1440 non-constant elements |
| $\sigma_f$ | LAN standard deviation of $f(\hat\theta)$: $\sigma_f^2 = \mathbf d_f^\top F^{+}\mathbf d_f$ |
| $c_k$ | contribution of Fisher direction $k$ to $\sigma_f^2$: $c_k = (\mathbf d_f\cdot v_k)^2/\lambda_k$ |
| $\Sigma_R$ | LAN covariance matrix of all PTM elements, $DF^{+}D^\top$ |
| $\delta$ | target 95% half-width of a PTM element |
| weak share | fraction of $\sigma_f^2$ that comes from directions with $1/\sqrt{\lambda_k} > \varepsilon$, i.e. beyond $n_{\rm eff}(\varepsilon)$; $\varepsilon = 0.01$ unless stated |
| van Trees width $\tau$ | prior standard deviation per tangent direction in the van Trees bound (README 4.6) |
| dominant / small Kraus operators | in the canonical Kraus decomposition of a gate (orthogonal Kraus operators), the one with the largest norm / the other three. They carry the coherent / incoherent part of the gate |
| $L_{\max}$ | maximum sequence length of the design (`seq_len_list = [1, L_cut, L_max]`) |
| $\Phi$ | cumulative distribution function of the standard normal distribution |
| $T$, $F$, $\lambda_k$, $v_k$, $n_{\rm eff}(\varepsilon)$, $J_T$, $p$ | as in README Sections 3, 4.2, 4.5 and 4.7 |

---

## 1. Answers in brief

**Q1. What does $\delta f = 1.96\sqrt{\mathbf d_f^\top F^+\mathbf d_f}$ look like for a PTM element?** ([Section 2](#2-derivation-of-the-interval-for-a-pauli-transfer-matrix-element))

$$
\delta R_{g,ij} \;=\; 1.96\,\sqrt{\sum_{k=1}^{624}\frac{(\mathbf d\cdot v_k)^2}{\lambda_k}}
\;=\;\frac{1.96}{\sqrt m}\sqrt{\mathbf d^\top\big(J_T^\top\operatorname{diag}(p)^{-1}J_T\big)^{+}\mathbf d},
\qquad
\mathbf d = T^\top\Big[\nabla R_{g,ij} - \big(\partial_h R_{g,ij}\big)\,H_{hh}^{-1}H_{h\theta}\Big].
$$

- The bracket is the gradient of the **gauge-fixed** element, obtained with the implicit function theorem ([Section 2.2](#22-gauge-fixing-and-the-gradient-of-fcircpi)). Without a gauge fixing, a PTM element has no finite bound and the formula is meaningless.
- Before the gauge fixing, the gradient with respect to the Kraus operator $K_{gr}$ is simply $(2/d)\,P_iK_{gr}P_j$ ([Section 2.3](#23-explicit-gradient-before-the-gauge-fixing)).
- 1.96 $=\Phi^{-1}(0.975)$; it is valid under LAN ([Section 2.5](#25-why-the-estimate-is-approximately-gaussian-lan)). It is pyGSTi's $\sqrt{C_1}$.
- For all 1440 elements together: covariance $\Sigma_R = DF^+D^\top$, and $z = 4.14$ instead of 1.96 if every element must be within $\delta$ at once ([Section 2.7](#27-all-elements-covariance-and-simultaneous-intervals)).

**Q2. Is $F$ the Riemannian Fisher information?** Yes, in the sense of Boumal 2013. The Fisher information is a bilinear form on the tangent space, and $F$ is its matrix in the basis $T$, which is orthonormal for the Euclidean metric of $(K, A, B)$ (README 4.5). The bilinear form needs no metric, but its **eigenvalues** do. So $\lambda_k$ and $n_{\rm eff}(\varepsilon)$ depend on the metric, and $\sigma_f$ does not ([Section 3.2](#32-eigenvalues-depend-on-the-metric-sigma_f-does-not)). Numerically, a random change of metric moves $n_{\rm eff}(0.01)$ from 135 to 181, and $\sigma_f^2$ by $10^{-11}$ (relative).

**Q3. Can we know analytically how the weak directions affect the PTM error?** Partly ([Section 3](#3-how-the-weak-directions-enter)).

- **Under LAN, exactly.** $\sigma_f^2 = \sum_k c_k$, so each direction's share is a number you can compute. The spectrum alone cannot give it: it depends on the overlaps $\mathbf d_f\cdot v_k$, and $n_{\rm eff}$ depends on the metric.
- **For the actual estimator, not from $F$ alone.** Here the weak directions ($k\gtrsim100$) are the small Kraus operators, i.e. the incoherent part of the gates (79–95% of their weight). The estimate does not move along them (README 5.5). So they contribute a **bias**, not shot noise, and that bias is limited by the size of the incoherent part. Collapsing the incoherent part of one gate changes each of its PTM elements by at most $3(1-\mu_1)$ to first order, where $1-\mu_1$ is the gate's non-dominant (normalized) Choi weight: about $2\times10^{-3}$ here ([Section 3.5](#35-what-the-estimator-does-along-them-a-bias-limited-by-the-incoherent-part)).
- **Consequences at the reference design** ($N = 400$, $m = 1000$, $L_{\max} = 14$) [Numerical: 15]:
  - The LAN $\sigma_f$ overestimates the actual error about 4× (median rmse/$\sigma_f$ = 0.24). 94% (median) of $\sigma_f^2$ comes from directions beyond $n_{\rm eff}(0.01) = 135$, but 91% of the actual squared error comes from the first 100.
  - Keeping only the first 135 directions **underestimates** the error for a quarter of the elements: 74% coverage at nominal 95%.
  - The van Trees bound with $\tau\approx0.01$ is calibrated (median rmse/$\sigma$ 0.93–0.96, coverage 95–96%). An a-priori width computed from the incoherent weight, $\tau_0 = 5.4\times10^{-3}$, is about a factor 2 too small.
  - So an eigenvalue count can err in **both** directions. Use per-element numbers, not $n_{\rm eff}$.

**Q4. Which designs reach a target $\delta$?** ([Sections 4](#4-scaling-with-m-n-and-l_max) and [5.1](#51-results-of-the-design-scan); [Numerical: 16], evaluated at a prior guess, not the truth)

- **Bracket, not a single number.** Per design we get two values of the required shots $m$: the van Trees width (optimistic, close to reality) and the LAN width (conservative). At the reference design ($L_{\max} = 14$, $N = 400$, $\delta = 10^{-2}$ per element) they are $m\approx750$ and $m\approx3700$. The 40 real refits reached $\delta$ for the worst element at $m = 1000$.
- **Depth is the cheapest lever.** Going from $L_{\max} = 14$ to $23$ cuts the required $m$ by about 2.5 for both widths: the Fisher eigenvalues of the strong directions grow like $L_{\max}^{0.7-1.0}$ and those of the weakest like $L_{\max}^{1.8}$.
- **More circuits beat more shots for the conservative width.** Above the threshold the van Trees width depends on $Nm$ only ($\sigma\propto N^{-0.5}$), but the LAN width falls like $N^{-1.2}$ to $N^{-1.7}$, because the weakest directions become much better conditioned away from $N\approx208$.
- **Below $N = 208$ no PTM element is estimable** (LAN width infinite for all 1440); any finite number there rests on the prior.
- Example for $\delta = 10^{-2}$ per element: $L_{\max} = 23$ with $N = 600$ needs $m\approx200$–$450$ (total $0.12$–$0.27\times10^6$ shots), against $m\approx750$–$3700$ at $L_{\max} = 14$, $N = 400$ ($0.3$–$1.5\times10^6$). Requiring all 1440 elements within $\delta$ at once costs 4.5–6 times more.
- These are predictions. Test T3 ([Section 5](#5-numerical-tests)) refits simulated data at the chosen design to confirm them.

---

## 2. Derivation of the interval for a Pauli transfer matrix element

### 2.1 The quantity, and why it needs a gauge fixing

[Derived here] With $\mathcal G_g(X) = \sum_r K_{gr}XK_{gr}^\dagger$,

$$
R_{g,ij}(\theta) = \frac1d\sum_{r=1}^{4}\operatorname{tr}\big(P_iK_{gr}P_jK_{gr}^\dagger\big),\qquad i,j = 0,\dots,15 . \tag{1}
$$

- **Real.** $\mathcal G_g(P_j)$ is Hermitian and so is $P_i$, so $\overline{\operatorname{tr}(P_i\mathcal G_g(P_j))} = \operatorname{tr}(\mathcal G_g(P_j)P_i) = \operatorname{tr}(P_i\mathcal G_g(P_j))$.
- **First row constant.** $R_{g,0j} = \frac1d\operatorname{tr}\big(\sum_rK_{gr}^\dagger K_{gr}P_j\big) = \frac1d\operatorname{tr}P_j = \delta_{j0}$ on the model, by trace preservation. That removes $6\times16 = 96$ of the $1536$ elements; **1440** remain. (Off the model this row is not constant, so in code it vanishes only after projecting with $T$.)
- **Kraus mixing and the purifications leave $R$ unchanged**: $\sum_r K_rXK_r^\dagger$ is invariant under $K_r\to\sum_sV_{rs}K_s$ with $V$ unitary, and the purifications act on $A$, $B$ only.
- **The Hilbert-space gauge changes $R$.** $K\to UKU^\dagger$ gives $\mathcal G\to\mathcal U\circ\mathcal G\circ\mathcal U^{-1}$ with $\mathcal U(X) = UXU^\dagger$, hence $R_g\to R_UR_gR_U^\top$ with $(R_U)_{ij} = \operatorname{tr}(P_iUP_jU^\dagger)/d$. $R_U$ is orthogonal, because $\mathcal U$ preserves the Hilbert–Schmidt inner product and $\{P_i/\sqrt d\}$ is an orthonormal basis.

So a raw PTM element changes along the 15 Hilbert-space gauge directions, which are in $\ker F$. By DERIVATIONS.md §D.1–D.2 its gradient is then not in $\operatorname{range}F$: no unbiased estimator exists, and its LAN variance is infinite. A PTM element becomes a well-defined target only after a gauge fixing.

### 2.2 Gauge fixing and the gradient of $f\circ\Pi$

**The gauge fixing** (as in scripts 13–16; README 6.3):

$$
U^*(\theta) = \arg\min_{U}\; c(U,\theta),\qquad c(U,\theta) = \sum_g\big\lVert R_UR_g(\theta)R_U^\top - R_g^{\rm target}\big\rVert_F^2,
\qquad \Pi(\theta) = U^*(\theta)\cdot\theta ,
$$

where $U\cdot\theta$ applies $K\to UKU^\dagger$, $A_i\to A_iU^\dagger$, $B\to UB$. It agrees with mGST's `perform_gauge` to $2.5\times10^{-5}$ in the superoperators (README 6.3).

**$f\circ\Pi$ is gauge invariant** [Derived here]. If $\theta' = V\cdot\theta$, then $c(U, \theta') = c(UV, \theta)$, so the minimizer is $U^*(\theta') = U^*(\theta)V^\dagger$ (assuming it is unique up to a global phase). Hence $\Pi(\theta')$ and $\Pi(\theta)$ have the same PTMs. Kraus mixing and the purifications change neither $R$ nor $c$.

**Its gradient** [Derived here; the existence of $h^*$ is the implicit function theorem]. Parametrize $U(h) = \exp\big(\tfrac i2\sum_{a=1}^{15}h_aP_a\big)$ and write $c(h,\theta)$. Evaluate at a gauge-fixed point $\theta_0$, so that $h^*(\theta_0) = 0$ and $\nabla_hc(0,\theta_0) = 0$.

1. If $H_{hh} = \partial^2c/\partial h^2$ ($15\times15$) is invertible, the implicit function theorem gives a smooth $h^*(\theta)$ near $\theta_0$ with $\nabla_hc(h^*(\theta),\theta) \equiv 0$.
2. Differentiate this identity with respect to $\theta$: $H_{hh}\,\partial h^*/\partial\theta + H_{h\theta} = 0$, with $H_{h\theta} = \partial(\nabla_hc)/\partial\theta$ ($15\times928$). So $\partial h^*/\partial\theta = -H_{hh}^{-1}H_{h\theta}$.
3. $\Pi(\theta) = U(h^*(\theta))\cdot\theta$, and $U(0)\cdot\theta = \theta$. The chain rule at $\theta_0$ gives

$$
\nabla(f\circ\Pi) = \nabla f - (\partial_hf)\,H_{hh}^{-1}H_{h\theta},\qquad \partial_hf = \frac{\partial}{\partial h}f\big(U(h)\cdot\theta_0\big)\Big|_{h=0}\ (1\times15). \tag{2}
$$

Check: (2) has no component along the Hilbert-space gauge directions, $1.3\times10^{-15}$ relative [Numerical: 13]. In code: `ptm_gradient` in [`scripts/ptm_tools.py`](scripts/ptm_tools.py).

### 2.3 Explicit gradient before the gauge fixing

[Derived here] Vary $K_{gr}\to K_{gr}+\delta K_r$ in (1). The first-order change is $\frac1d\sum_r\big[\operatorname{tr}(P_i\,\delta K_r\,P_jK_r^\dagger) + \operatorname{tr}(P_iK_rP_j\,\delta K_r^\dagger)\big]$. The second term is the complex conjugate of the first (take the adjoint inside the trace and use cyclicity), so

$$
\delta R_{g,ij} = \frac2d\sum_r\operatorname{Re}\operatorname{tr}\big(P_i\,\delta K_r\,P_jK_r^\dagger\big) = \frac2d\sum_r\big\langle P_iK_rP_j,\;\delta K_r\big\rangle,
\qquad \langle X,Y\rangle = \operatorname{Re}\operatorname{tr}(X^\dagger Y),
$$

using $\operatorname{tr}(P_i\,\delta K\,P_jK^\dagger) = \operatorname{tr}\big((P_iKP_j)^\dagger\delta K\big)$. The metric $\langle\cdot,\cdot\rangle$ is the dot product of the 928 real coordinates (README 4.2). Hence

$$
\nabla_{K_{gr}}R_{g,ij} = \frac2d\,P_iK_{gr}P_j,\qquad \big\lVert\nabla_{K_{gr}}R_{g,ij}\big\rVert_F = \frac2d\lVert K_{gr}\rVert_F . \tag{3}
$$

- Over the four Kraus operators of the gate, $\lVert\nabla_KR_{g,ij}\rVert = \frac2d\big(\sum_r\lVert K_{gr}\rVert_F^2\big)^{1/2} = \frac2d\sqrt{\operatorname{tr}I_d} = 2/\sqrt d = 1$.
- **The amplitude factor:** the part of the gradient on Kraus operator $r$ is proportional to $\lVert K_{gr}\rVert$. For our truth the dominant Kraus operators have $\lVert K\rVert^2\approx3.998$ and the small ones $4\times10^{-4}$–$1.5\times10^{-3}$ [Numerical: 15]. So a unit step of a small Kraus operator changes a PTM element by at most $\frac2d\lVert K_r\rVert\approx0.01$–$0.02$, against $\approx1$ for the dominant one. This is what makes the weak directions special ([Section 3.4](#34-amplitude-coordinates-small-lambda_k-but-ptm-errors-that-do-not-shrink-with-it)).

### 2.4 Tangent coordinates and estimability

$\mathbf d_f = T^\top\nabla(f\circ\Pi)$. As in README 4.5 (§C.3), the derivative of $z\mapsto f(\Pi(\theta(z)))$ at $z = 0$ is the ambient gradient applied to $T$, exactly; only first derivatives enter.

**When is the bound finite?** [Derived here, from DERIVATIONS.md §D.2]
- At rank 624, $\ker F$ consists of the 191 gauge directions (README §A.6). $f\circ\Pi$ is gauge invariant, so $\mathbf d_f\perp\ker F$, i.e. $\mathbf d_f\in\operatorname{range}F$: the bound is finite.
- Below the identifiability threshold ($3N < 624$), $\ker F$ also contains non-gauge directions. If $\mathbf d_f$ has a component there, $f$ is not estimable to first order: the LAN variance is infinite. **$F^+$ silently drops that component**, so code must check $\lVert(\ker F\text{ basis})^\top\mathbf d_f\rVert/\lVert\mathbf d_f\rVert\approx0$. At $N = 150$, none of the 1440 elements passes this check [Numerical: 16].

### 2.5 Why the estimate is approximately Gaussian (LAN)

This is the assumption behind the 1.96 (pyGSTi states it as "we assume that local asymptotic normality (LAN) applies", Nielsen et al. 2021, Sec. 6.3.1 (checked)). Here is where it comes from, and what it needs. [Derived here: the standard heuristic argument for asymptotic normality of the maximum-likelihood estimate. A rigorous version is in textbooks, e.g. van der Vaart, *Asymptotic Statistics* (1998), Ch. 5 and 7; not checked.]

Use the 624 coordinates $y$ along the eigenvectors with $\lambda_k > 0$, $z = Vy$. They cover the directions transverse to the gauge orbits, and in them $F_y = V^\top FV = \operatorname{diag}(\lambda)$ is invertible.

1. **Stationarity.** If the estimate $\hat y$ is an interior maximum of the log-likelihood $\ell$, then $\nabla\ell(\hat y) = 0$.
2. **Taylor expansion around the truth** ($y = 0$): $0 = \nabla\ell(0) + \nabla^2\ell(0)\,\hat y + O(\lVert\hat y\rVert^2)$.
3. **The score** $s = \nabla\ell(0)$ has mean 0 and covariance $F_y$ (README 4.5). It is a sum of independent contributions of the $Nm$ shots, so by the central limit theorem it is approximately $\mathcal N(0, F_y)$.
4. **The Hessian.** $\mathbb E[-\nabla^2\ell(0)] = F_y$ (README 4.5, "Properties"). For many shots, $-\nabla^2\ell(0)$ is close to its mean (law of large numbers).
5. **Solve.** $\hat y\approx F_y^{-1}s\sim\mathcal N\big(0,\,F_y^{-1}F_yF_y^{-1}\big) = \mathcal N\big(0,\operatorname{diag}(1/\lambda)\big)$. In tangent coordinates, $\hat z = V\hat y\sim\mathcal N(0, F^+)$.

**What each step needs, and where it fails here:**

| Assumption | Fails here when | Evidence |
|---|---|---|
| A1: interior maximum (step 1) | the truth is near the positivity boundary | 81% of estimated gates have a Choi eigenvalue below $10^{-6}$ (README 5.4) |
| A2: quadratic term small (step 2) | $1/\sqrt{\lambda_k}$ is comparable to the size of the model along $v_k$ | $1/\sqrt{\lambda_k} > 0.08$ for $k > 400$, larger than the incoherent part (README 5.5) |
| A3: Hessian close to its mean (step 4) | few shots | not tested separately |
| A4: the optimizer finds the global maximum | local minima, early stopping | fraction of converged runs (README 5.1) |

### 2.6 Delta method and the interval

[Derived here]
1. Taylor in $z$: $f(\Pi(\theta(\hat z))) - f(\Pi(\theta^*)) = \mathbf d_f^\top\hat z + O(\lVert\hat z\rVert^2)$.
2. The gauge part of $\hat z$ is arbitrary (wherever the optimizer drifted), but $\mathbf d_f\perp$ gauge, so it drops out.
3. With Section 2.5: $f(\hat\theta) - f(\theta^*)\approx\mathcal N(0,\sigma_f^2)$, with $\sigma_f^2 = \mathbf d_f^\top F^+\mathbf d_f$.
4. $P\big(\lvert\mathcal N(0,\sigma^2)\rvert\le z\sigma\big) = 2\Phi(z) - 1 = 0.95$ for $z = \Phi^{-1}(0.975) = 1.96$. Equivalently $z^2 = 3.84$ is the 95% quantile of $\chi^2_1$, pyGSTi's $C_1$ (README 6.1, their eq. 78).

So the 95% interval of a PTM element is

$$
f(\hat\theta)\;\pm\;1.96\,\sigma_f,\qquad
\sigma_f^2 = \sum_{k=1}^{624}\frac{(\mathbf d_f\cdot v_k)^2}{\lambda_k} = \frac1m\,\mathbf d_f^\top\big(J_T^\top\operatorname{diag}(p)^{-1}J_T\big)^{+}\mathbf d_f . \tag{4}
$$

The second form uses $F = mJ_T^\top\operatorname{diag}(p)^{-1}J_T$ (README 4.5) and $(mA)^+ = A^+/m$. With real data, $F$, $T$ and $\mathbf d_f$ are evaluated at the gauge-fixed estimate ("plug-in"); for PTM elements plug-in and oracle gave the same widths (README 6.3), and the rank deficiency of the raw estimate does not matter for them ([Section 6](#6-caveats-and-open-questions)).

### 2.7 All elements: covariance and simultaneous intervals

[Derived here] Stack the rows $\mathbf d_f^\top$ into $D$ ($1440\times815$). Under LAN the vector of errors is $\Delta R\approx D\hat z\sim\mathcal N(0,\Sigma_R)$ with

$$
\Sigma_R = DF^+D^\top = \sum_{k=1}^{624}\frac{(Dv_k)(Dv_k)^\top}{\lambda_k}. \tag{5}
$$

- Its diagonal is $\sigma_f^2$. Its off-diagonal entries are the correlations between elements. Any derived quantity $\varphi(R)$ (a fidelity, a diamond-norm proxy) has $\sigma_\varphi^2 = \nabla\varphi^\top\Sigma_R\nabla\varphi$.
- $\mathbb E\lVert\Delta R\rVert_F^2 = \operatorname{tr}\Sigma_R$.
- **Every element within $\delta$ at once.** By the union (Bonferroni) bound, $P(\text{some }\lvert\Delta R_f\rvert > z\sigma_f)\le n\cdot2(1-\Phi(z))$. Setting this to $0.05$ with $n = 1440$ gives $z = \Phi^{-1}(1 - 0.05/2880) = 4.14$. It ignores correlations, so it is conservative. Compared with 1.96 it costs $(4.14/1.96)^2 = 4.5$ times more shots.
- Do **not** use the $\chi^2$ ellipsoid of the whole parameter vector (pyGSTi's confidence region) for individual elements. Its radius would be $\sqrt{\chi^2_{624}(0.95)} = 26$ instead of 4.14.

### 2.8 Code

```python
import numpy as np
import ptm_tools as pt                       # scripts/ptm_tools.py
theta, _, _ = pt.gauge_fix(pt.to_vec(K, A, B))               # Pi; then evaluate everything at theta
fi = pt.fisher(theta, gates, m)                               # T, lam (decreasing), V, kernel basis of ker F
D = pt.ptm_gradient(theta) @ fi["T"]                          # eq. (2), then T^T: rows d_f
D = D[np.linalg.norm(D, axis=1) > 1e-10]                      # drop the 96 constant first-row elements
kernel_share = np.linalg.norm(D @ fi["kernel"], axis=1) / np.linalg.norm(D, axis=1)
assert kernel_share.max() < 1e-6, "some elements are not estimable with this design (Section 2.4)"
G = D @ fi["V"]                                               # d_f . v_k
c = G ** 2 / fi["lam"]                                        # c_k per element
sigma_lan = np.sqrt(c.sum(axis=1))                            # eq. (4)
half_width = 1.96 * sigma_lan
weak_share = c[:, fi["lam"] < 1 / eps ** 2].sum(axis=1) / c.sum(axis=1)
sigma_vt = np.sqrt((G ** 2 / (fi["lam"] + 1 / tau ** 2)).sum(axis=1))   # van Trees, Section 3.7
```

---

## 3. How the weak directions enter

### 3.1 Per-direction decomposition, and what the spectrum alone cannot give

[Derived here] By (4), $\sigma_f^2 = \sum_kc_k$. The cumulative share $S_f(K) = \sum_{k\le K}c_k/\sigma_f^2$ says where an element's variance comes from.

**From the eigenvalues alone**, the only bounds are: $\mathbf d_f\in\operatorname{span}\{v_k\}$, so $\sum_k(\mathbf d_f\cdot v_k)^2 = \lVert\mathbf d_f\rVert^2$ and

$$
\frac{\lVert\mathbf d_f\rVert^2}{\lambda_1}\;\le\;\sigma_f^2\;\le\;\frac{\lVert\mathbf d_f\rVert^2}{\lambda_{624}} .
$$

With a spectrum that spans 11 decades (README 5.3) this says nothing. The count $n_{\rm eff}(\varepsilon)$ predicts $\sigma_f$ only if $\mathbf d_f$ lies in the span of the first $n_{\rm eff}$ directions, and checking that means computing $\mathbf d_f\cdot v_k$, i.e. computing $\sigma_f$.

### 3.2 Eigenvalues depend on the metric, $\sigma_f$ does not

[Derived here] Change the tangent coordinates by any invertible $Q$: $z = Qz'$. Then $F' = Q^\top FQ$ and $\mathbf d' = Q^\top\mathbf d$.

1. **$\sigma_f$ is unchanged.** For a symmetric positive semidefinite $A$ and $\mathbf b\in\operatorname{range}A$,

   $$\mathbf b^\top A^+\mathbf b = \sup_{x:\,x^\top Ax>0}\frac{(\mathbf b^\top x)^2}{x^\top Ax}.$$

   *Proof.* $\mathbf b = AA^+\mathbf b$, so $\mathbf b^\top x = (A^+\mathbf b)^\top Ax\le\sqrt{\mathbf b^\top A^+\mathbf b}\sqrt{x^\top Ax}$ (Cauchy–Schwarz in the inner product $\langle u,w\rangle_A = u^\top Aw$; $A^+AA^+ = A^+$), with equality at $x = A^+\mathbf b$. For $x\in\ker A$, $\mathbf b^\top x = 0$ because $\operatorname{range}A\perp\ker A$, so those $x$ do not matter. ∎

   Substituting $x = Qx'$ maps the ratio for $(F, \mathbf d)$ one-to-one onto the ratio for $(F', \mathbf d')$, so both suprema are equal: $\mathbf d'^\top F'^+\mathbf d' = \mathbf d^\top F^+\mathbf d$.
2. **The eigenvalues change.** $Q^\top FQ$ is not similar to $F$ unless $Q$ is orthogonal. An eigenvalue of a bilinear form is defined relative to an inner product $M$ ($Fv = \lambda Mv$); here $M$ is the Euclidean metric of $(K, A, B)$.
3. A different *parametrization* (for example Choi matrices or error rates instead of Kraus operators) acts on tangent vectors through its Jacobian at the point, so locally it is such a $Q$. Where that Jacobian is singular (Kraus amplitude $\to0$, Section 3.4) the spectra differ the most.

[Numerical: 15] With $Q = O\operatorname{diag}(e^{g})O^\top$, $g_i\sim\mathcal N(0,1)$, $O$ random orthogonal: $\sigma_f^2$ changes by at most $8.7\times10^{-12}$ (relative), while $n_{\rm eff}(0.01)$ goes from 135 to 181 and $n_{\rm eff}(0.003)$ from 64 to 98.

**A count in PTM units.** The eigenvalues of $\Sigma_R$ are metric-free statements in PTM units. At the reference design, 131 principal standard deviations of $\Sigma_R$ exceed $10^{-2}$, whereas 489 Fisher directions have $1/\sqrt{\lambda_k} > 10^{-2}$ [Numerical: 15].

### 3.3 What the weak directions are here

[Numerical: 15] The truth's Kraus operators are canonical (their Gram matrix is diagonal to $3\times10^{-18}$), so "dominant" and "small" are well defined. Median weight $\lVert\cdot\rVert^2$ of $v_k$ on each block, $N = 400$, $m = 1000$:

| $k$ | $1/\sqrt{\lambda_k}$ (median) | dominant Kraus | small Kraus | SPAM | $\lVert Dv_k\rVert$ | $\lVert Dv_k\rVert/\sqrt{\lambda_k}$ | share of $\operatorname{tr}\Sigma_R$ | mean $e_k^2$ |
|---|---|---|---|---|---|---|---|---|
| 1–50 | $2.0\times10^{-3}$ | 0.84 | 0.01 | 0.14 | 2.6 | $5.5\times10^{-3}$ | 0.024 | 0.90 |
| 51–100 | $3.3\times10^{-3}$ | 0.76 | 0.03 | 0.17 | 2.4 | $7.9\times10^{-3}$ | 0.061 | 0.91 |
| 101–135 | $8.1\times10^{-3}$ | 0.01 | 0.79 | 0.20 | 0.23 | $1.8\times10^{-3}$ | 0.003 | 0.50 |
| 136–200 | $1.4\times10^{-2}$ | 0.00 | 0.87 | 0.13 | 0.12 | $1.7\times10^{-3}$ | 0.003 | 0.32 |
| 201–300 | $2.4\times10^{-2}$ | 0.00 | 0.91 | 0.09 | 0.08 | $2.0\times10^{-3}$ | 0.007 | 0.16 |
| 301–400 | $5.0\times10^{-2}$ | 0.00 | 0.94 | 0.06 | 0.06 | $3.0\times10^{-3}$ | 0.018 | 0.07 |
| 401–500 | $0.13$ | 0.00 | 0.95 | 0.05 | 0.05 | $6.6\times10^{-3}$ | 0.094 | 0.02 |
| 501–624 | $0.38$ | 0.00 | 0.95 | 0.05 | 0.04 | $1.6\times10^{-2}$ | 0.790 | 0.01 |

- $\lVert Dv_k\rVert$ is the Frobenius norm of the change of all 1440 elements per unit step along $v_k$; a random unit tangent direction gives 0.97.
- $\lVert Dv_k\rVert/\sqrt{\lambda_k}$ is the LAN standard deviation of direction $k$ in PTM units.
- $e_k$ is the per-direction efficiency statistic of README 5.5 (1 = the estimate moves as much as LAN says; 0 = it does not move).

**Reading.**
- The split between strong and weak directions **is** the split between the coherent part (dominant Kraus operators) plus SPAM and the incoherent part (small Kraus operators).
- Weak directions move the PTMs 10–60 times less per unit step than strong ones: the amplitude factor of (3).
- But their $1/\sqrt{\lambda_k}$ is larger by more than that, so in PTM units their LAN standard deviation is not smaller (last bins: $1.6\times10^{-2}$). Under LAN they would dominate: 88% of $\operatorname{tr}\Sigma_R$ comes from $k > 400$.
- The estimate does not move along them ($e_k^2\le0.02$ for $k > 400$).

### 3.4 Amplitude coordinates: small $\lambda_k$, but PTM errors that do not shrink with it

[Derived here; a one-parameter toy model] Let a small Kraus operator have amplitude $a$ and the corresponding error rate (Choi weight) $r = a^2$, as in (1), where Kraus operators enter quadratically. Near the truth, a circuit probability and a PTM element depend on the rate linearly: $p = p_0 + \beta r$, $R = R_0 + \gamma r$, with $p_0\in(0,1)$.

1. **Fisher information.** For a two-outcome circuit, $F_x = m\,(\partial p/\partial x)^2/(p(1-p))$ (README 4.5, §C.8). With $\partial p/\partial r = \beta$ and $\partial p/\partial a = 2\beta a$:

   $$F_a = 4a^2F_r .$$

   In Kraus coordinates the direction looks weak, and weaker the smaller the true amplitude: $1/\sqrt{F_a} = 1/(2a\sqrt{F_r})\to\infty$ as $a\to0$.
2. **PTM error under LAN.** $\partial R/\partial a = 2\gamma a$, so

   $$\sigma_R = \frac{\lvert\partial R/\partial a\rvert}{\sqrt{F_a}} = \frac{2\lvert\gamma\rvert a}{2a\sqrt{F_r}} = \frac{\lvert\gamma\rvert}{\sqrt{F_r}} ,$$

   independent of $a$, and the same as in rate coordinates (an instance of Section 3.2). So a small $\lambda_k$ in Kraus coordinates does not by itself mean a large PTM error: the amplitude factor of (3) cancels the $a$ in $1/\sqrt{F_a}$.
3. **When LAN holds.** The Gaussian approximation needs the error in $a$ to be small compared with $a$ itself (otherwise the estimate reaches $a = 0$): $1/(2a\sqrt{F_r})\ll a$, i.e.

   $$\sigma_r = 1/\sqrt{F_r}\;\ll\;2r .$$

   **The rate must be resolved relative to its own size.** Here the rates are $\sim10^{-4}$ per Kraus operator while the last bins of Section 3.3 have PTM-unit standard deviations near $10^{-2}$: the incoherent part is far from resolved, and LAN fails for it.

### 3.5 What the estimator does along them: a bias limited by the incoherent part

When the data cannot resolve a rate, the constrained estimate stops at or near $a = 0$ (README 5.4–5.5). The error along those directions is then roughly the truth's own component there: a bias, which does not shrink with $m$ until the rate is resolved. How large can it be in a PTM element? [Derived here]

Fix a gate, let $s = \sum_{r\,\text{small}}\lVert K_r\rVert_F^2 = d\,(1-\mu_1)$, where $\mu_1$ is its dominant normalized Choi eigenvalue, and let $S = \sum_{r\,\text{small}}K_r^\dagger K_r\succeq0$, so $\operatorname{tr}S = s$.

1. **The small Kraus operators' own contribution.** By Cauchy–Schwarz for the Hilbert–Schmidt inner product and unitarity of the Paulis, $\lvert\operatorname{tr}(P_iK_rP_jK_r^\dagger)\rvert\le\lVert P_iK_r\rVert_F\lVert P_jK_r^\dagger\rVert_F = \lVert K_r\rVert_F^2$. So removing them changes $R_{ij}$ by at most $s/d$.
2. **Restoring trace preservation.** Now $K_1^\dagger K_1 = I - S$. The polar retraction gives $K_1' = K_1(I-S)^{-1/2} = K_1 + \tfrac12K_1S + O(s^2)$. By (3), $\Delta R_{ij} = \frac1d\operatorname{Re}\operatorname{tr}(P_iK_1SP_jK_1^\dagger) + O(s^2)$, and $\lvert\operatorname{tr}(P_iK_1S\,P_jK_1^\dagger)\rvert\le\lVert K_1S\rVert_F\lVert K_1\rVert_F\le\lVert S\rVert_F\sqrt d\le s\sqrt d$, using $\lVert K_1\rVert_{\rm op}\le1$ and $\lVert S\rVert_F\le\operatorname{tr}S$ for $S\succeq0$. So this step changes $R_{ij}$ by at most $s/\sqrt d$.
3. **Total:** collapsing the whole incoherent part of a gate changes each of its PTM elements by at most

   $$\Big(\frac1d + \frac1{\sqrt d}\Big)s + O(s^2) = 3(1-\mu_1) + O(s^2)\quad(d = 4).$$

For our truth, $\max_gs_g/d = 7.1\times10^{-4}$, so the bound is $\approx2.1\times10^{-3}$ [Numerical: 15]. Caveats: the estimator collapses the incoherent part only partially, and the bound ignores how a change of one gate propagates to the others through the gauge fixing $\Pi$.

**Consequence for design.** If $\delta$ is well above a few times the incoherent infidelity, unresolved incoherent directions cannot by themselves spoil the target; what must be resolved is the coherent part and SPAM. If $\delta$ is comparable to the incoherent infidelity or below it, the design must resolve the rates, $\sigma_r\ll r$ (Section 3.4), and only then is the LAN width trustworthy.

### 3.6 The numbers at the reference design

[Numerical: 15] $N = 400$, $m = 1000$, design seed 12345, 40 estimates (default settings of script 07). Actual error: median rmse over elements $1.2\times10^{-3}$, largest $5.2\times10^{-3}$; largest single error $1.45\times10^{-2}$. "ratio" = rmse / $\sigma$ per element.

| Prediction | median $\sigma$ | max $\sigma$ | median ratio | 5%–95% of ratio | coverage of $\pm1.96\sigma$ |
|---|---|---|---|---|---|
| first 50 directions | $3.3\times10^{-4}$ | $1.9\times10^{-3}$ | 7.5 | 1.35–64 | 39.9% |
| first 100 | $1.1\times10^{-3}$ | $4.3\times10^{-3}$ | 1.91 | 0.83–15 | 58.5% |
| first 135 $= n_{\rm eff}(0.01)$ | $1.1\times10^{-3}$ | $4.5\times10^{-3}$ | 1.24 | 0.82–4.4 | 73.6% |
| first 200 | $1.2\times10^{-3}$ | $4.6\times10^{-3}$ | 1.01 | 0.78–1.7 | 91.3% |
| first 300 | $1.4\times10^{-3}$ | $4.8\times10^{-3}$ | 0.81 | 0.45–1.07 | 97.8% |
| all 624 (LAN, eq. 4) | $6.1\times10^{-3}$ | $1.0\times10^{-2}$ | 0.24 | 0.03–0.54 | 100% |
| van Trees $\tau = 0.1$ | $2.5\times10^{-3}$ | $5.4\times10^{-3}$ | 0.52 | 0.11–0.88 | 99.4% |
| van Trees $\tau = 0.02$ | $1.4\times10^{-3}$ | $4.6\times10^{-3}$ | 0.77 | 0.36–1.07 | 97.9% |
| van Trees $\tau = 2\tau_0 = 0.011$ | $1.2\times10^{-3}$ | $4.2\times10^{-3}$ | 0.93 | 0.61–1.21 | 95.9% |
| van Trees $\tau = 0.01$ | $1.2\times10^{-3}$ | $4.1\times10^{-3}$ | 0.96 | 0.65–1.24 | 95.2% |
| van Trees $\tau = \tau_0 = 5.4\times10^{-3}$ | $1.0\times10^{-3}$ | $3.4\times10^{-3}$ | 1.21 | 0.92–2.05 | 86.1% |

**Where the actual error comes from** [Numerical: 15]. Using the per-direction statistic $e_k$ of README 5.5, to first order $\Delta R\approx\sum_k(Dv_k)\,e_k/\sqrt{\lambda_k}$. This reconstruction reproduces the actual $\Delta R$ to 24% (median relative residual), and splits $\mathbb E\lVert\Delta R\rVert^2$ as follows:

| $k$ | share of the actual $\mathbb E\lVert\Delta R\rVert^2$ | share of $\operatorname{tr}\Sigma_R$ (LAN prediction) |
|---|---|---|
| 1–100 | 0.91 | 0.085 |
| 101–135 | 0.02 | 0.003 |
| 136–400 | 0.04 | 0.029 |
| 401–624 | 0.04 | 0.883 |

Predicted $\operatorname{tr}\Sigma_R = 5.4\times10^{-2}$; actual mean $\lVert\Delta R\rVert_F^2 = 4.5\times10^{-3}$ (12× smaller). Along the principal directions of $\Sigma_R$ the estimate is far from efficient: mean $(q_j\cdot\Delta R)^2/s_j^2$ is 0.01–0.08 for the 100 largest.

**Reading.**
1. **The full LAN width is conservative here, by 4× in standard deviation**, because it is dominated by incoherent directions along which the estimate does not move.
2. **The hard cutoff at $n_{\rm eff}(0.01)$ is not conservative.** For a quarter of the elements the error is mostly in directions 136–400, which the estimate does follow partly ($e_k^2 = 0.07$–$0.32$). For test probabilities (README 5.6) the same cutoff happened to work; for PTM elements it does not. The cutoff that works best, 200, was found by looking at the data.
3. **The soft cutoff (van Trees) is calibrated at $\tau\approx0.01$.** It counts direction $k$ fully when $\lambda_k\gg\tau^{-2}$, and as an error of size $\tau$ along $v_k$ otherwise. The incoherent part gives an a-priori scale: if the estimate misses all of it, the error vector has squared norm $\sum_g s_g = 1.45\times10^{-2}$, spread over $624 - 135 = 489$ weak directions, i.e. $\tau_0 = \sqrt{1.45\times10^{-2}/489} = 5.4\times10^{-3}$ per direction. That is the right order of magnitude but a factor 2 too small here; why is open ([Section 6](#6-caveats-and-open-questions)).

### 3.7 Three numbers per element

For each design and element compute [code in Section 2.8]:

| Number | Meaning | Here |
|---|---|---|
| $\sigma_f^{\rm LAN}$, eq. (4) | the pyGSTi-style width; valid when LAN holds | conservative, 4× |
| $\sigma_f^{\rm vT}(\tau) = \big(\sum_k(\mathbf d_f\cdot v_k)^2/(\lambda_k + \tau^{-2})\big)^{1/2}$ | soft truncation, $\tau$ from the expected incoherent part | calibrated at $\tau\approx2\tau_0$ |
| weak share | how much of $\sigma_f^{\rm LAN}$ rests on unresolved directions | median 0.94 |

When the weak share is small, the two widths agree, LAN holds for that element, and $\pm1.96\sigma_f$ is a calibrated interval. A design target $\delta$ should therefore be checked as $1.96\max_f\sigma_f\le\delta$ under **both** widths, and the weak share says how much to trust the difference.

---

## 4. Scaling with $m$, $N$ and $L_{\max}$

### 4.1 Shots per circuit $m$

[Derived here] $F\propto m$ (README 4.5), so $\sigma_f^{\rm LAN}\propto m^{-1/2}$ exactly, and the shots needed for a target are

$$
m_{\rm req} = m_0\Big(\frac{z\,\sigma_f^{\rm LAN}(m_0)}{\delta}\Big)^2 ,\qquad z = 1.96\ \text{(each element)}\ \text{or}\ 4.14\ \text{(all at once)}.
$$

The van Trees width does not scale this way: as $m$ grows, more directions pass $\lambda_k > \tau^{-2}$ and are counted fully. The weak share falls with $m$, which is the regime where the LAN width becomes calibrated rather than conservative.

### 4.2 Number of circuits $N$

[Derived here]
- **More circuits never hurt (under LAN).** Adding circuits adds their Fisher information: $F_{N+M} = F_N + F_M\succeq F_N$. By the variational formula of Section 3.2, a larger $F$ gives a smaller ratio $(\mathbf d^\top x)^2/x^\top Fx$ for every $x$, so $\mathbf d^\top F_{N+M}^+\mathbf d\le\mathbf d^\top F_N^+\mathbf d$ (both finite, since $\ker F_{N+M}\subseteq\ker F_N$).
- **Above the threshold, only $Nm$ matters (heuristic).** For circuits drawn independently from one distribution, $F_N = \sum_cF_c$ and $F_N/N\to\bar F = \mathbb E[F_c]$ by the law of large numbers. If $\bar F$ has rank 624, then $\sigma_f^2\approx\mathbf d^\top\bar F^+\mathbf d/(Nm)$: circuits and shots are interchangeable. This needs $N$ well above 208, because near the threshold the smallest eigenvalues of $F_N$ are far from those of $N\bar F$ (README 4.7, Rudelson & Vershynin).
- **Below the threshold** ($3N < 624$) the PTM elements are not estimable at first order (Section 2.4). Only the van Trees width is finite, and it is then set by $\tau$, i.e. by the prior, not by the data.

**Measured** [Numerical: 16, prior guess, slopes of $\log\max_f\sigma_f$ at $m = 1000$ over $N = 400$–$800$]:
- van Trees: $\sigma\propto N^{-0.46}$ to $N^{-0.72}$ (closest to $-0.5$ for $L_{\max}\ge14$), as the heuristic says.
- LAN: $\sigma\propto N^{-1.24}$ to $N^{-1.65}$, much faster. The LAN width of the worst element is dominated by the weakest directions, whose eigenvalues grow faster than $N$ while $N$ is only a few times the threshold. So for the conservative width, $Nm$ is **not** the only thing that matters in this range: at $L_{\max} = 14$ the total shots needed for $\delta = 10^{-2}$ fall from $5.5\times10^6$ ($N = 300$) to $1.5\times10^6$ ($N = 400$) and $0.46\times10^6$ ($N = 800$).

### 4.3 Maximum depth $L_{\max}$

[Derived here; toy models. These are expectations to test, not results.]
- **Repeated coherent error (germ-like).** One qubit, $\lvert0\rangle$, $L$ repetitions of a rotation by $\theta$ about $x$, measure $z$: $p_1 = \sin^2(L\theta/2)$. Then $\partial p_1/\partial\theta = \frac L2\sin(L\theta)$ and $p_1(1-p_1) = \frac14\sin^2(L\theta)$, so $F_\theta = mL^2$ **exactly**, for every $\theta$ with $p_1\in(0,1)$. The standard deviation of $\theta$ falls like $1/L$.
- **Random sequences (heuristic).** $\partial p/\partial\theta$ is a sum over the $\approx L/6$ positions of the gate, each term with a sign set by the random gates around it. If those terms behave like independent zero-mean terms, the sum grows like $\sqrt L$ and $F\propto L$. This is the expectation in README 2.2 (action points) that random designs gain less from depth than germ designs.
- **Incoherent errors in random sequences (heuristic).** A stochastic error at any position pulls the outcome distribution towards uniform, so the $L$ contributions to $\partial p/\partial r$ have the same sign and add up: $\partial p/\partial r\propto L$ and $F_r\propto L^2$ while $rL\ll1$. In amplitude coordinates $F_a = 4a^2F_r$ (Section 3.4), also $\propto L^2$.
- **Decoherence limits the gain.** The signal decays roughly like $(1-r)^L$ for a per-gate error rate $r$, which gives an optimum near $L\approx1/(2r)$. With $r\sim10^{-3}$ that is several hundred, far above the $L_{\max}\le23$ that `random_seq_design` supports. So within the supported range, deeper should not hurt here.

**Measured** [Numerical: 16, prior guess, $N = 400$; median per-shot eigenvalue $\lambda_k/m$ in bins of $k$, slope of $\log\lambda$ versus $\log L_{\max}$ over $L_{\max} = 10$–$23$]:

| $k$ | $L_{\max} = 6$ | 10 | 14 | 18 | 23 | slope |
|---|---|---|---|---|---|---|
| 1–50 (coherent, SPAM) | 144 | 192 | 250 | 295 | 360 | 0.74 |
| 51–100 (coherent, SPAM) | 40 | 63 | 94 | 118 | 140 | 0.96 |
| 136–400 (incoherent) | 0.54 | 1.1 | 1.2 | 1.4 | 1.6 | 0.47 |
| 401–624 (incoherent) | $9.9\times10^{-4}$ | $7.9\times10^{-3}$ | $1.6\times10^{-2}$ | $2.3\times10^{-2}$ | $3.5\times10^{-2}$ | 1.77 |

- The strong (coherent and SPAM) directions grow roughly like $L_{\max}$, as the random-walk heuristic says; random sequences do not give the $L^2$ of germs.
- The weakest directions grow like $L_{\max}^{1.8}$, close to the $L^2$ of the incoherent heuristic. The middle bin (0.47) fits neither; it mixes directions whose composition changes with $L_{\max}$.
- Consequence for the PTM widths: $\max_f\sigma_f\propto L_{\max}^{-0.6}$ to $L_{\max}^{-0.8}$ (van Trees) and $L_{\max}^{-0.9}$ to $L_{\max}^{-1.2}$ (LAN), so $m_{\rm req}\propto L_{\max}^{-1.3}$ to $L_{\max}^{-2.4}$.

---

## 5. Numerical tests

The goal is a design $(N, m, L_{\max})$ for which the gauge-fixed PTM elements are within $\delta$ with 95% probability, together with evidence that the prediction is right. The tests go from cheap (no fitting) to expensive (many fits).

### T1. Design scan without fitting (script 16)

For each $(N, L_{\max})$ and three design seeds, at an evaluation point that does not use the truth (the prior guess of README 5.4: target plus different noise, seed 7), compute per element $\sigma^{\rm LAN}$, $\sigma^{\rm vT}(\tau)$, the weak share and estimability, and from them $m_{\rm req}(\delta)$ and the total shots $N\,m_{\rm req}$.

*Results:* see [Section 5.1](#51-results-of-the-design-scan).

### T2. Check the scaling laws on the scan

- $m$: exact by construction for LAN.
- $N$: is $N\,m_{\rm req}$ roughly constant for $N\gg208$ (Section 4.2)?
- $L_{\max}$: does $m_{\rm req}$ fall with depth, and closer to $1/L$ or $1/L^2$ (Section 4.3)?
- Estimability: does every design with $N\ge208$ reach 100%?

### T3. Validation by simulation at the candidate designs (the main confidence test)

The scan predicts; only refitting simulated data tests the prediction for *this* estimator. For each of two or three candidate designs from T1:

1. Simulate $R = 40$ datasets from a truth (start with the current one), fit with the exact pipeline of script 07 (same initial point, optimizer settings, log-likelihood), gauge-fix with $\Pi$.
2. Per element: rmse over datasets, rmse/$\sigma^{\rm LAN}$ and rmse/$\sigma^{\rm vT}$, and coverage of $\pm1.96\sigma$ (as in script 15, part 1).
3. The quantity that matters for the goal: the fraction of (element, dataset) pairs with $\lvert\Delta R\rvert\le\delta$, and the fraction of datasets in which **all** elements are within $\delta$.
4. The fraction of converged runs (assumption A4).

*Precision.* With $R = 40$, an rmse is known to about $\pm1/\sqrt{2(R-1)} = \pm11\%$ (README 6.4), and a coverage near 95% to about $\pm3.4\%$ per element (binomial; but elements are correlated). *Cost.* About 22 s per fit at $N = 400$ (script 07), so about 15 minutes per design; more for larger $N$ and $L_{\max}$.

*Needs:* script 07 with $N$, $m$, $L_{\max}$ and the design seed as arguments, and script 15 reading them. Not written yet.

### T4. Robustness to the truth

The bias of Section 3.5 grows with the incoherent part, and the calibration of $\tau$ was found at one truth.
- Cheap: repeat T1 at several prior guesses (noise seeds; `noise_strength` 0.01, 0.05, 0.2) and check whether the recommended design changes.
- Expensive: repeat T3 at one stronger-noise truth (`noise_strength` 0.2–0.3), where the incoherent part is larger than the shot-noise scale (README 2.1, recommendation 13). There, $\tau_0$ is larger and LAN should hold for more directions.

### T5. With real data

At the gauge-fixed estimate (plus a small random perturbation if the raw estimate is rank deficient; Section 6), report $\sigma^{\rm LAN}$, $\sigma^{\rm vT}(\tau)$ and the weak share. Then run the parametric bootstrap of script 14, which reproduced the actual spread of PTM elements to within a few percent but covered about 92% at nominal 95% (README 6.4). If the bootstrap standard deviation satisfies $1.96\,s\le\delta$ and agrees with $\sigma^{\rm vT}$, the design worked.

### Decision rule

Choose the cheapest $(N, m, L_{\max})$, in total shots $Nm$, such that:
1. all elements are estimable, with a margin above $N = 208$ (conditioning is worst at the threshold, README 5.2);
2. $1.96\max_f\sigma_f^{\rm vT}(\tau)\le\delta$ with $\tau$ from the expected incoherent part;
3. preferably also $1.96\max_f\sigma_f^{\rm LAN}\le\delta$. Then the pyGSTi-style interval certifies $\delta$ on its own, conservatively, whatever $\tau$ is;
4. T3 confirms it: at least 95% of (element, dataset) pairs within $\delta$, or of datasets with all elements within $\delta$ if that is the goal.

### 5.1 Results of the design scan

[Numerical: 16; full tables in `data/16_ptm_design_scan_prior_output.txt` and `..._truth_output.txt`] Medians over 3 design seeds. Designs `seq_len_list = [1, L_max//2 + 1, L_max]`; $N\in\{100,\dots,800\}$, $L_{\max}\in\{6, 10, 14, 18, 23\}$.

**Can the design be evaluated without the truth? Yes.** At the prior guess and at the truth, $\max_f\sigma_f$ agrees within 5% for $N\ge300$ (within 15% at $N = 250$, $L_{\max} = 14$). Across the 3 design seeds, the LAN $\max_f\sigma_f$ varies by a factor 1.04–1.26 at the prior guess for $N\ge300$.

**Predicted errors at $m = 1000$** (prior guess, largest over the 1440 elements):

| $L_{\max}$ | $N$ | $n_{\rm eff}(0.01)$ | weak share (median) | $\max\sigma^{\rm LAN}$ | $\max\sigma^{\rm vT}(0.01)$ |
|---|---|---|---|---|---|
| 14 | 100–200 | 88–110 | 1.00 | $\infty$ (not estimable) | $7.5$–$5.7\times10^{-3}$ |
| 14 | 250 | 114 | 1.00 | $4.6\times10^{-2}$ | $5.1\times10^{-3}$ |
| 14 | 400 | 128 | 0.94 | $9.8\times10^{-3}$ | $3.9\times10^{-3}$ |
| 14 | 800 | 167 | 0.70 | $3.9\times10^{-3}$ | $2.8\times10^{-3}$ |
| 23 | 400 | 131 | 0.91 | $6.2\times10^{-3}$ | $2.8\times10^{-3}$ |
| 23 | 800 | 172 | 0.67 | $2.7\times10^{-3}$ | $2.0\times10^{-3}$ |

**Required shots per circuit for $\delta = 10^{-2}$**, each element at 95% ($z = 1.96$). Cells: LAN | van Trees $\tau = 0.01$ | $\tau = 0.02$. In brackets: total shots $Nm$ in millions.

| $L_{\max}$ | $N = 300$ | $N = 400$ | $N = 600$ | $N = 800$ |
|---|---|---|---|---|
| 6 | $6.4\times10^6$ \| 7943 \| 11220 | $2.5\times10^5$ \| 3548 \| 5012 | $2.9\times10^4$ \| 1585 \| 2239 | $1.2\times10^4$ \| 1413 \| 1995 |
| 10 | $5.6\times10^4$ \| 1413 \| 2239 | $1.1\times10^4$ \| 1413 \| 1995 | 2711 \| 631 \| 891 (1.6 \| 0.4 \| 0.5) | 1175 \| 398 \| 501 (0.9 \| 0.3 \| 0.4) |
| 14 | $1.8\times10^4$ \| 891 \| 1259 | 3679 \| 562 \| 708 (1.5 \| 0.2 \| 0.3) | 1169 \| 316 \| 447 (0.7 \| 0.2 \| 0.3) | 574 \| 251 \| 316 (0.5 \| 0.2 \| 0.3) |
| 18 | 8799 \| 562 \| 794 | 2327 \| 355 \| 501 (0.9 \| 0.1 \| 0.2) | 652 \| 200 \| 282 (0.4 \| 0.1 \| 0.2) | 382 \| 158 \| 224 (0.3 \| 0.1 \| 0.2) |
| 23 | 5546 \| 398 \| 501 | 1463 \| 224 \| 316 (0.6 \| 0.1 \| 0.1) | 453 \| 141 \| 200 (0.3 \| 0.1 \| 0.1) | 269 \| 126 \| 158 (0.2 \| 0.1 \| 0.1) |

For all 1440 elements within $\delta$ at once ($z = 4.14$), multiply the LAN column by 4.5; the van Trees columns grow by 5–6.3, because they do not scale as a pure power of $m$ (see the output file).

**Check against the refits.** At $L_{\max} = 14$, $N = 400$, the 40 refits at $m = 1000$ had a worst-element rmse of $5.2\times10^{-3}$, i.e. a 95% half-width of about $1.0\times10^{-2}$ (Section 3.6). The table brackets that: $m\approx710$–$790$ (van Trees, $\tau = 0.02$; slightly optimistic) and $m\approx3700$ (LAN; conservative). For the *worst* element use $\tau = 0.02$ rather than 0.01: at the reference design the 95th percentile of rmse/$\sigma^{\rm vT}$ is 1.24 for $\tau = 0.01$ and 1.07 for $\tau = 0.02$ (Section 3.6).

**Reading.**
1. **$N\ge250$ is necessary, $N\ge400$ is sensible.** Below 208 nothing is estimable. Just above it the LAN width is huge (worst conditioning, README 5.2), and at $N = 300$ the largest LAN $\sigma$ is still 3.5 ($L_{\max} = 23$) to 50 ($L_{\max} = 6$) times the van Trees one.
2. **Use the deepest supported sequences.** Every column improves monotonically with $L_{\max}$. From 14 to 23 the required $m$ falls by 2.5 (LAN) and 2.5 (van Trees, $\tau = 0.01$).
3. **Then prefer circuits over shots** until LAN and van Trees agree. At $L_{\max} = 23$ the two are within a factor 2 for $N = 800$ (269 versus 126–158), and the weak share falls to 0.67.
4. **A candidate for $\delta = 10^{-2}$ per element:** $L_{\max} = 23$, $N = 600$. The bracket is $m\approx200$ (van Trees, $\tau = 0.02$) to $450$ (LAN). With $m = 450$ the target is met even by the conservative width, at $0.27\times10^6$ shots in total, against $0.4\times10^6$ for the current $L_{\max} = 14$, $N = 400$, $m = 1000$ (which reached $\delta$ only for the van Trees bracket and in the refits). It must be confirmed by T3. Fitting cost grows with $N$ and $L_{\max}$ and is not included.

### 5.2 Results at noise 0.3 (the truth of `rank_vs_number_sequences_2q.ipynb`)

[Numerical: notebook [`../ptm_precision_design_2q.ipynb`](../ptm_precision_design_2q.ipynb), `scripts/ptm_design.py`; scan at the truth, $N\in\{300,\dots,1000\}$, $L_{\max}\in\{10, 14, 18, 23\}$, 3 design seeds.]

**The regime is different.** The incoherent weight per gate is $1-\mu_1 = 1.8$–$2.5\times10^{-2}$, against $5$–$7\times10^{-4}$ at noise 0.05. Empirically $1-\mu_1\approx0.24\,s^2$ for `noise_strength` $s$, from these two points. The bias cap of Section 3.5 is then $7.5\times10^{-2} > \delta = 10^{-2}$, so the design has to resolve the incoherent rates (Section 3.4).

The notebook therefore adds a fourth number: the rate resolution $\sigma(\mu)/\mu$ of the 18 small Choi eigenvalues. These eigenvalues are gauge-invariant, so they need no $\Pi$. A design is accepted when the LAN width reaches $\delta$ **and** $\max\sigma(\mu)/\mu\le0.5$; the 0.5 margin on the "$\ll2$" of Section 3.4 is a judgment call. The notebook's "Conclusions" cell discusses when T3–T5 are still needed.

**Results** ($\delta = 10^{-2}$ per element, $z = 1.96$):
- **Estimability:** 100% of the elements for every $N\ge300$; seed spread 1.01–1.21.
- **Rates:** at the LAN-required $m$, $\max\sigma(\mu)/\mu = 0.33$–$0.37$ in every $(L_{\max}, N)$ cell, so the LAN width alone sets the design. For $\delta\gtrsim1.4\times10^{-2}$ the rate requirement binds instead.
- **LAN versus van Trees:** $m_{\rm LAN}/m_{\rm vT}(2\tau_0) = 1.3$–$2.1$, with $\tau_0 = 3.3\times10^{-2}$; at noise 0.05 the ratio was 6.5. The uncalibrated $\tau$ therefore matters little here.
- **Weak share:** 0.82–0.92 although the rates are resolved. With $\varepsilon = 0.01$ fixed in Kraus coordinates it is not a useful indicator at this noise level (small Kraus amplitudes $0.12$–$0.23$). Use the rate resolution instead.
- **Scaling:**
  - $\sigma^{\rm LAN}\propto N^{-0.91\ldots-1.06}$ and $L_{\max}^{-0.58\ldots-0.85}$; $\sigma^{\rm vT}\propto N^{-0.54\ldots-0.64}$.
  - Total shots keep falling up to $N = 1000$, the edge of the scan: at $L_{\max} = 23$, 2.3M, 1.6M, 1.3M, 1.1M for $N = 400, 600, 800, 1000$.
- **Candidates:**
  - $L_{\max} = 23$, $N = 1000$, $m\approx1100$ (1.1M shots; van Trees $m\approx780$);
  - $L_{\max} = 23$, $N = 600$, $m\approx2600$ (1.6M);
  - for comparison, $L_{\max} = 14$, $N = 400$ needs $m\approx10700$ (4.3M).
  - All 1440 elements at once: about 4.5× more.
- **Done since (2026-11-17), in [`../ptm_design_pipeline_2q.ipynb`](../ptm_design_pipeline_2q.ipynb):**
  - **T4:** the prior guess gives $m_{\rm design}$ 0.74–1.01 times the truth's, and the same best design.
  - **Depth:** the gain stops at $L_{\max}\approx23$–28 at noise 0.3 and continues to 32 at noise 0.05 ([`../depth_check_2q.ipynb`](../depth_check_2q.ipynb)).
  - **T3** (10 refits per design; scripts 17, 18, 25, 26):
    - At noise 0.3 ($L_{\max}=28$, $N=600$, $m=2200$, rates resolved), LAN is nearly calibrated: median rmse/σ = 0.81, coverage 97%, $\Lambda\approx0.88\,n_{\rm id}$, $N_\sigma\approx1.2$.
    - At noise 0.05 ($L_{\max}=32$, $N=400$, $m=150$), van Trees ($2\tau_0$) is calibrated (0.95, 94%).
    - At the noise-0.3 design the optimizer met no convergence criterion even after 60 outer iterations; see the notebook's conclusions.

---

## 6. Caveats and open questions

- **One truth, one noise level.** All calibration numbers (the 4× of LAN, the factor 2 on $\tau_0$, the failure of the cutoff at 135) come from nearly unitary gates with `noise_strength` 0.05. T4 is needed before relying on them elsewhere.
- **Why is $\tau$ a factor 2 above $\tau_0$?** The estimate of $\tau_0$ spreads the missed incoherent part evenly over all 489 weak directions. If the error is concentrated in the less-weak directions (136–400), which have larger $\mathbf d_f\cdot v_k$, the effective $\tau$ is larger. Testable with the reconstruction of Section 3.6 per bin.
- **The gauge fixing is part of the quantity.** $\sigma_f$ refers to $R\circ\Pi$ with this $\Pi$. `perform_gauge` adds a SPAM term with weight 0.1, which changes the numbers slightly (README 6.3).
- **Plug-in at a rank-deficient estimate is fine for PTM elements.** The raw maximum-likelihood estimate has rank 599–620 (README 5.4), so $\mathbf d_f$ could have components in the non-gauge kernel, which $F^+$ would silently drop. For estimates 0, 1 and 2 (ranks 613, 607, 607) no element has a kernel share above $10^{-6}$ (median $2\times10^{-12}$), and the same holds after a $10^{-3}$ random perturbation (rank 624). The reason is (3): the extra kernel directions switch on collapsed Kraus operators, and the PTM gradient along a Kraus operator vanishes when that operator is zero. [Numerical: one-off check during the session, same code as Section 2.8.] Keep the estimability check in the code anyway.
- **Bias cap.** Section 3.5 bounds the effect of collapsing the incoherent part of one gate. It does not include the propagation through $\Pi$ to other gates, nor partial collapse.
- **Depth.** `random_seq_design` supports $L_{\max} < 24$ only, so the scan cannot test the decoherence-limited optimum of Section 4.3, and germ-like designs are not covered.

---

## 7. Files

| File | What it does | Runtime |
|---|---|---|
| [`scripts/ptm_tools.py`](scripts/ptm_tools.py) | shared helpers: gate sets, PTMs, gauge fixing $\Pi$, gradient (2) of the gauge-fixed elements, Fisher eigen-decomposition with kernel basis | — |
| [`scripts/15_ptm_error_decomposition.py`](scripts/15_ptm_error_decomposition.py) | at the reference design: truncated and van Trees predictions versus the 40 estimates; per-direction table of Section 3.3; $\Sigma_R$; metric check; attribution of the actual error; a-priori $\tau_0$. Output in `data/15_ptm_error_decomposition_output.txt` | ~30 s |
| [`scripts/16_ptm_design_scan.py`](scripts/16_ptm_design_scan.py) | design scan T1 over $N$, $L_{\max}$, design seeds, at the prior guess (`prior`) or the truth (`truth`); writes `data/16_ptm_design_scan_<point>.npz`. With `summary`, only prints the tables of Section 5.1 and the slopes of Section 4 from the saved file (`> data/16_ptm_design_scan_<point>_output.txt`) | ~15 min per point; summary seconds |
| [`scripts/ptm_design.py`](scripts/ptm_design.py) | generic version of `ptm_tools.py` (gate set and target as arguments, any number of qubits), plus the design scan of script 16, the rate resolution $\sigma(\mu)/\mu$ of the small Choi eigenvalues (Section 3.4), the summary tables and the design-map figures. Reproduces `ptm_tools.py` to $10^{-13}$ | — |
| [`../ptm_precision_design_2q.ipynb`](../ptm_precision_design_2q.ipynb) | the design maps (tests T1 and T2) at the notebook's truth (`noise_strength` 0.3): regression check against Sections 3.6 and 5.1, scan, figures for the three numbers of Section 3.7 plus the rate resolution, decision plot, scaling slopes, summary tables. Writes `data/ptm_design_scan_noise<s>_<point>.npz` | ~10 min (cached afterwards) |
| [`scripts/17_ptm_refit_runner.py`](scripts/17_ptm_refit_runner.py) | test T3, part 1: script 07 with truth and design as arguments; `--dry-run` prints the design and a time estimate. Writes `data/ptm_refits/` | ~22 s per refit at $N = 400$, $L_{\max} = 14$; more for larger designs |
| [`scripts/18_ptm_refit_check.py`](scripts/18_ptm_refit_check.py) | test T3, part 2: refits of script 17 against the Fisher predictions (rmse/$\sigma$, coverage, fraction within $\delta$, rates at the boundary, converged fraction) | ~1 s per estimate |

Scripts 15 and 16 need the estimates of script 07 (`data/crb_estimates/default_*.npz`). Scripts 17 and 18 are written but have not been run.

**Sources** are those of README Section 8, plus:
- A. W. van der Vaart, *Asymptotic Statistics*, Cambridge University Press (1998), Ch. 5 (asymptotic normality of M-estimators and of the maximum-likelihood estimate) and Ch. 7 (local asymptotic normality). Not checked; cited for the rigorous version of Section 2.5.
