# Validity of the Fisher-information analysis: assumptions, proofs and a plan

Companion to [README.md](README.md), [DERIVATIONS.md](DERIVATIONS.md) and [PTM_PRECISION.md](PTM_PRECISION.md), written 2026-11-16 with Claude Code. It answers one question:

> The study uses the Fisher information (1) to choose an experiment design before the experiment and (2) to attach an error to the estimated Pauli transfer matrix elements after the optimization. Which assumptions does each use rest on? Do they hold for maximum-likelihood estimation with the Riemannian optimization under realistic noise? What is still missing before the method is applied to four-qubit gate set tomography?

**Provenance convention** (as in the README):

- **[Source: …]**: a result from the literature, with full references in [Section 9](#9-sources). **(checked)** means the statement was compared with the source or with a secondary source that quotes it, as detailed in Section 9. Everything else is from memory and must be checked before it goes into a paper.
- **[Derived here]**: a calculation written out in this file.
- **[Numerical: script]**: our own computation, reproducible with the script named.

Results of the other three files are referred to as "README 4.5", "DERIVATIONS §C.7", "PTM_PRECISION 3.4".

---

## Contents

0. [Terms and symbols](#0-terms-and-symbols)
1. [Answers in brief](#1-answers-in-brief)
2. [The chain of claims](#2-the-chain-of-claims)
3. [The assumptions](#3-the-assumptions)
4. [Derivations](#4-derivations)
5. [New numerical evidence: goodness of fit of the saved estimates](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates)
6. [Realistic noise: which assumption each mechanism breaks](#6-realistic-noise-which-assumption-each-mechanism-breaks)
7. [Verdict and plan](#7-verdict-and-plan)
8. [Four qubits](#8-four-qubits)
9. [Sources](#9-sources)

---

## 0. Terms and symbols

| Term | Meaning |
|---|---|
| LAN | local asymptotic normality. A property of a statistical *model*: near the truth and for many shots, the log-likelihood ratio is a quadratic function of the parameters with a Gaussian linear term. Precise definition in [Section 4.2](#42-local-asymptotic-normality-definition-and-proof-at-a-regular-point). It is not the same statement as "the estimate is Gaussian" ([Section 4.3](#43-asymptotic-normality-of-the-estimate-and-of-a-pauli-transfer-matrix-element)) |
| POVM, SPAM | positive operator-valued measure (the measurement); state preparation and measurement |
| Kraus rank $r_K$ | number of Kraus operators per gate in the model (4 in the two-qubit study) |
| Choi eigenvalues $\mu_1\ge\mu_2\ge\dots$ | eigenvalues of the normalized Choi matrix of a gate. $\mu_1\approx1$ is the coherent part. The small ones, $\mu_j$ with $j\ge2$, are the *incoherent rates*. They equal $\lVert K_j\rVert_F^2/d$ for canonical (mutually orthogonal) Kraus operators $K_j$ (PTM_PRECISION 3.3) |
| regular point | a gate set near which $\operatorname{rank}J_T$ is constant and equal to the number of identifiable parameters (624 in the two-qubit study) |
| fold | a point where a parametrization squares a coordinate, here rate $=$ (Kraus amplitude)$^2$. The image has a boundary (rate $\ge0$) although the parameter space has none ([Section 4.4](#44-finite-number-of-shots-the-positivity-fold)) |
| rate resolution $u$ | a small eigenvalue divided by its Cramér–Rao standard deviation, $u = \mu/\sigma(\mu)$ ([Section 4.5](#45-three-numbers-that-measure-validity-without-fitting)) |
| pseudo-truth $\theta^\dagger$ | when the model is wrong: the gate set in the model whose predictions are closest to the device's in Kullback–Leibler divergence ([Section 4.7](#47-wrong-model-pseudo-truth-sandwich-covariance-goodness-of-fit)) |
| deviance $\mathcal D$ | $2[\ell_{\rm sat} - \ell(\theta)]$, twice the gap between the log-likelihood of the data under "probabilities = observed frequencies" and under the gate set $\theta$. In the code it is $2\times$ (cost $-$ cost floor) |
| likelihood-ratio statistic $\Lambda$ | $2[\ell(\hat\theta) - \ell(\theta^*)]$, twice the log-likelihood gained by the estimate over the truth |
| $\chi^2_k$ | chi-squared distribution with $k$ degrees of freedom (mean $k$, standard deviation $\sqrt{2k}$) |
| $O_P(a_m)$, $o_P(1)$ | a random quantity that stays bounded in probability after division by $a_m$; one that converges to zero in probability, as $m\to\infty$ |
| $\Phi$, $\phi$ | cumulative distribution function and density of the standard normal distribution |

| Symbol | Meaning (two-qubit value) |
|---|---|
| $d$, $n_{\rm out}$, $n_g$ | Hilbert-space dimension (4); outcomes per circuit (4); number of gates (6) |
| $\mathcal M$, $D$ | the model manifold (product of Stiefel manifolds, README 4.1); its dimension (815) |
| $n_{\rm gauge}$, $n_{\rm id}$ | number of gauge directions (191); of identifiable directions, $D - n_{\rm gauge}$ (624) |
| $\theta^*$, $\hat\theta$ | true gate set; maximum-likelihood estimate (output of the optimization) |
| $N$, $m$, $L_{\max}$ | number of circuits; shots per circuit; maximum sequence length |
| $p_{ci}(\theta)$, $n_{ci}$, $f_{ci} = n_{ci}/m$ | model probability, count and frequency of outcome $i$ of circuit $c$ |
| $q_{ci}$ | outcome probabilities of the real device when it is *not* in the model (Section 4.7 only) |
| $z$, $\theta(z)$ | tangent coordinates at a reference gate set, $\theta(z) = R_\theta(Tz)$ with $R$ a retraction and $T$ an orthonormal tangent basis (README 4.5) |
| $J_T$, $F = mJ_T^\top\operatorname{diag}(p)^{-1}J_T$, $F_1 = F/m$ | Jacobian of the probabilities in tangent coordinates; Fisher information; Fisher information of one shot of every circuit |
| $\lambda_k$, $v_k$ | eigenvalues (decreasing) and eigenvectors of $F$ |
| $\ell$, $s$ | log-likelihood $\sum_{c,i}n_{ci}\log p_{ci}$; score $\nabla_z\ell$ at the truth |
| $f$, $\mathbf d_f$, $\sigma_f$ | a gauge-fixed Pauli transfer matrix element; its gradient in tangent coordinates; $\sigma_f^2 = \mathbf d_f^\top F^+\mathbf d_f$ (PTM_PRECISION 2.6) |

---

## 1. Answers in brief

### 1.1 Verdict

| Purpose | Usable? | In which form | Still missing |
|---|---|---|---|
| **1. Design before the experiment** | **Yes, now, for two qubits** | A bracket per Pauli transfer matrix element, evaluated at a prior guess: the LAN width $\sigma_f$ (upper end) and the van Trees width (lower end, PTM_PRECISION 3.7), plus the three validity numbers of [Section 4.5](#45-three-numbers-that-measure-validity-without-fitting) | One refit test at a design where the validity numbers say "asymptotic regime" (scripts 17 and 18 exist and have not been run). A realistic prior guess of the device noise |
| **2. Error bars after the fit** | **Partly** | On data that follow the model: Fisher interval as a conservative interval, parametric bootstrap for the realistic width (README 6.5). On real data: only next to a model-violation check, and the standard check is miscalibrated here ([Section 5](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates)) | A goodness-of-fit test calibrated by simulation. A test with a wrong model (Kraus rank too small, drift). A check at collapsed Kraus operators |
| **Four qubits** | **Not yet** | The same formulas. Nothing in the theory depends on the number of qubits | A scalable implementation, the expected-count check with 16 outcomes, and its own refit test ([Section 8](#8-four-qubits)) |

### 1.2 Main learnings

1. **The Fisher information itself needs only two assumptions**: the device is described by one point of the model, and the counts are multinomial. Under them $F = mJ_T^\top\operatorname{diag}(p)^{-1}J_T$ is exact for every number of shots. No normality and no asymptotics are involved ([Section 2](#2-the-chain-of-claims)).
2. **"Vanishing of additional terms" means three different things, with three different answers** ([Section 4.1](#41-the-additional-terms-three-kinds)).
   - Second-derivative and curvature terms in the *expected* Hessian: they cancel **exactly**, at every point of the model and for every $m$. In ambient coordinates the Euclidean second-derivative term and the Lagrange-multiplier term are each nonzero and cancel each other, so neither may be dropped alone.
   - Residual terms in the *observed* Hessian of one dataset: they do **not** vanish. They have mean zero and are smaller than $F$ by a factor of order $1/\sqrt m$ along well-determined directions, and they dominate along weakly determined ones.
   - Curvature terms of the intrinsic Cramér–Rao bounds: they do not appear for scalar quantities such as a Pauli transfer matrix element (README 4.6).
3. **LAN is a property of the model, and it holds here at every regular point as $m\to\infty$** (proof in [Section 4.2](#42-local-asymptotic-normality-definition-and-proof-at-a-regular-point)). The maximum-likelihood estimate is then asymptotically Gaussian with covariance $F^+$ on gauge-invariant quantities ([Section 4.3](#43-asymptotic-normality-of-the-estimate-and-of-a-pauli-transfer-matrix-element)). The true gate set of the study is a regular point. So the question is never whether LAN holds but whether $m$ is large enough.
4. **At the reference design it is not.** With $N = 400$, $m = 1000$, the fit gains as much likelihood over the truth as a regular model with **about 241 parameters** would, not 624 ([Section 5](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates), new). This agrees with the per-direction test of README 5.5.
5. **One mechanism explains it: the fold.** The model writes each incoherent rate as a squared Kraus amplitude. Along such a direction the asymptotic statements hold if and only if the rate is resolved, $\mu\gtrsim2$ to $3$ times its Cramér–Rao standard deviation $\sigma(\mu)$ ([Section 4.4](#44-finite-number-of-shots-the-positivity-fold)). In the best case this means that the data contain at least 4 to 9 error events attributable to that rate.
6. **In the one-parameter fold model the Fisher interval is never too narrow**: its coverage is at least 95% and the mean squared error is at most the Cramér–Rao value, for every true rate ([Section 4.4](#44-finite-number-of-shots-the-positivity-fold)). This explains why the intervals of README 6.3 were valid but loose. It is not a theorem for a Pauli transfer matrix element of the full model ([Section 4.8](#48-what-remains-true-when-asymptotic-normality-fails)).
7. **The standard goodness-of-fit test fails in the same regime.** For data simulated from the model itself, pyGSTi's model-violation measure comes out as $N_\sigma\approx+13$ ([Section 5](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates), new). On real data it would report a model violation that is not there. The reference distribution has to come from simulation.
8. **29% of the outcomes of the reference design have an expected count below 5**, and 6% are never observed in a given dataset ([Section 5](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates), new). Gaussian approximations of the counts are poor for them. With 16 outcomes per circuit this gets worse for four qubits.
9. **A wrong model changes what the error bar refers to more than its width.** The estimate converges to the pseudo-truth. To first order in the relative misfit the Fisher width is still right for the well-determined directions. The weakly determined directions absorb the model error whatever the number of shots ([Section 4.7](#47-wrong-model-pseudo-truth-sandwich-covariance-goodness-of-fit)).
10. **All simulations so far used a truth with exactly the Kraus rank of the model**, so the model was correct by construction (script 07: `rank_kraus=kraus_rank`). Realistic decoherence has full Kraus rank. This case has not been tested.
11. **The optimizer tolerance of 1 nat allows a shift of up to $1.4\,\sigma_f$ in the worst case**; a tolerance of 0.02 nat bounds it by $0.2\,\sigma_f$ ([Section 4.6](#46-optimizer-tolerance)). The converged and default runs agree (README 5.5), so the worst case did not occur, but the guarantee needs the tighter tolerance or the certificate given there.

### 1.3 What to do, in one paragraph

Use the analysis now for two-qubit design, with the bracket of widths and the validity numbers. Run the refit test once at a design the validity numbers approve, to confirm that the LAN width is then calibrated rather than conservative. For error bars on real data, add a bootstrap-calibrated goodness-of-fit test and test one deliberately wrong model. Only then scale to four qubits, where the first step is a matrix-free implementation and the expected-count check. The ordered list is in [Section 7.3](#73-what-still-has-to-be-done).

---

## 2. The chain of claims

An error bar on a Pauli transfer matrix element is the end of a chain of six claims. Each needs the assumptions of the previous ones plus new ones ([Section 3](#3-the-assumptions) defines the labels).

| # | Claim | Status | Needs |
|---|---|---|---|
| C1 | $F = mJ_T^\top\operatorname{diag}(p)^{-1}J_T$ is the Fisher information of the experiment at $\theta$ | **exact** for every $m$ (DERIVATIONS §C.7) | A1, A2, $p_{ci} > 0$ |
| C2 | $F$ equals the expected Riemannian Hessian of $-\ell$ | **exact** ([Section 4.1](#41-the-additional-terms-three-kinds)) | A1, A2 |
| C3 | The model is LAN at $\theta^*$ | **limit theorem**, $m\to\infty$ ([Section 4.2](#42-local-asymptotic-normality-definition-and-proof-at-a-regular-point)) | + R1, R2 |
| C4 | $\sqrt m\,(\hat\theta - \theta^*)$ is asymptotically Gaussian with covariance $F_1^+$, modulo gauge | **limit theorem** ([Section 4.3](#43-asymptotic-normality-of-the-estimate-and-of-a-pauli-transfer-matrix-element)) | + E1, E2 |
| C5 | $f(\hat\theta) \pm 1.96\,\sigma_f$ is a 95% interval for a gauge-fixed element $f$ | **limit theorem** (delta method) | + G1 |
| C6 | The same with $F$ and $\mathbf d_f$ evaluated at the estimate, or at a prior guess for design | **approximation** | + P1 |
| C7 | C3–C6 are accurate at the actual $m$ | **approximation**, measured by [Section 4.5](#45-three-numbers-that-measure-validity-without-fitting) | + R3 |

What does not need the chain beyond C1:

- **The spectrum of $F$, the rank, and comparisons between designs** (README 4.7, 6.1). They describe the information in the data, whatever the estimator.
- **The Cramér–Rao bound for unbiased estimators** and **the van Trees bound for all estimators** (DERIVATIONS §D, §E). Both are exact for every $m$.

What needs the whole chain: reading $\sigma_f$ as the standard deviation of *your* estimate.

---

## 3. The assumptions

| Label | Assumption | Holds in the simulations of this study? | In a real experiment? | Test | Consequence of failure |
|---|---|---|---|---|---|
| **A1** | The device is one point $\theta^*$ of the model: time-independent gates that do not depend on the circuit they are in, each a completely positive trace-preserving map of Kraus rank at most $r_K$ on a $d$-dimensional space | Yes, by construction: the truth is the target followed by a random channel of Kraus rank 4, the rank of the model (script 07) | Not exactly ([Section 6](#6-realistic-noise-which-assumption-each-mechanism-breaks)) | goodness of fit, calibrated by simulation (Sections 4.7, 5) | the estimate targets the pseudo-truth; weak directions absorb the misfit (4.7) |
| **A2** | Counts are multinomial: shots of a circuit are independent and identically distributed, circuits are independent | Yes | Drift violates it | dispersion of the residuals (4.7) | widths scale by the square root of the dispersion factor |
| **R1** | Every $p_{ci}(\theta^*) > 0$. For the Gaussian statements also: expected counts $mp_{ci}$ not small | $p > 0$: yes ($\min p = 1.5\times10^{-4}$). Counts: no, 470 of 1600 outcomes have $mp_{ci} < 5$ [Numerical: 19] | Same or worse | count them; merged-outcome width (4.5) | skewed errors for quantities that rely on rare outcomes |
| **R2** | $\theta^*$ is a regular point: $\operatorname{rank}J_T = n_{\rm id}$ in a neighbourhood. Equivalent here to: every gate has Kraus rank exactly $r_K$ and the state and effects have their full model rank (DERIVATIONS §A.7) | Yes (rank 624), but the truth is close to the singular set: its small Kraus operators have norm 0.02–0.04 | Yes in the same sense, and equally close or closer | rank at the evaluation point (README 4.3) | LAN and C4 are false at that point; boundary theory applies (4.4) |
| **R3** | Finite-$m$ version of R2: every direction that carries weight in $\mathbf d_f$ is resolved | No at noise strength 0.05; predicted yes for the candidate designs at noise strength 0.3 (PTM_PRECISION 5.2), not confirmed by refits | Depends on the incoherent infidelity and the design | rate resolution $u\ge2$, nonlinearity ratio $\kappa\le0.1$ (4.5) | interval conservative in all tests so far; realistic width only from the bootstrap |
| **E1** | The optimizer returns the global maximum of the likelihood on $\mathcal M$, and no gate set outside the gauge orbit of $\theta^*$ predicts the same probabilities | No outlier among 60 runs at the reference design: deviance $1007\pm44$ [Numerical: 19]. Failures at small $N$ (README 5.1). Global identifiability is not proven; the rank gives only the local statement | Same | several initial points; compare deviances | a wrong basin: no error bar applies |
| **E2** | The optimization error is small compared with $\sigma_f$ | Default and converged runs agree (README 5.5) | Same | certificate of Section 4.6 | shift of up to $\sigma_f\sqrt{2\epsilon}$ for a tolerance of $\epsilon$ nat |
| **G1** | The gauge fixing $\Pi$ is differentiable at $\theta^*$: the $15\times15$ Hessian $H_{hh}$ of the gauge cost is invertible (PTM_PRECISION 2.2) | Yes | Depends only on the target gate set | smallest eigenvalue of $H_{hh}$ | $\mathbf d_f$ undefined |
| **P1** | The point where $F$ and $\mathbf d_f$ are evaluated is regular and close to the truth | Truth and prior guess: yes. Raw estimate: rank 599–620 (README 5.4) | The estimate is all there is | evaluate at the estimate and at the estimate plus a small perturbation; compare | variance of collapsed directions silently dropped (4.4) |

---

## 4. Derivations

### 4.1 The "additional terms": three kinds

Three different sets of terms are called "additional" in the literature. They have different fates.

#### (a) Second-derivative and curvature terms in the expected Hessian: exact cancellation

[Derived here; extends DERIVATIONS §C.7.]

**In a chart.** Let $z\mapsto\theta(z)\in\mathcal M$ be *any* smooth local parametrization of the model (for example $\theta(z) = R_\theta(Tz)$), and write $p_{ci}(z)$ for $p_{ci}(\theta(z))$ and $\partial_k$ for $\partial/\partial z_k$. Differentiating $\ell(z) = \sum_{c,i}n_{ci}\log p_{ci}(z)$ twice:

$$
\partial_k\ell = \sum_{c,i}n_{ci}\frac{\partial_kp_{ci}}{p_{ci}},\qquad
\partial_k\partial_l\ell = \sum_{c,i}n_{ci}\Big[\frac{\partial_k\partial_lp_{ci}}{p_{ci}} - \frac{\partial_kp_{ci}\,\partial_lp_{ci}}{p_{ci}^2}\Big].
$$

This holds for every $z$ and every dataset. Now write $n_{ci} = mp_{ci} + (n_{ci} - mp_{ci})$ and collect:

$$
-\partial_k\partial_l\ell(z) \;=\; \underbrace{m\sum_{c,i}\frac{\partial_kp_{ci}\,\partial_lp_{ci}}{p_{ci}}}_{F_{kl}(z)}
\;-\; \underbrace{m\sum_c\partial_k\partial_l\Big(\sum_ip_{ci}\Big)}_{\text{(i)}}
\;+\; \underbrace{\sum_{c,i}\big(n_{ci} - mp_{ci}\big)\Big[\frac{\partial_kp_{ci}\,\partial_lp_{ci}}{p_{ci}^2} - \frac{\partial_k\partial_lp_{ci}}{p_{ci}}\Big]}_{\text{(ii)}\;=\;E_{kl}(z;n)} . \tag{1}
$$

- **Term (i) is zero identically.** $\theta(z)\in\mathcal M$ for all $z$, and every gate set on $\mathcal M$ gives $\sum_ip_{ci} = 1$. So $\sum_ip_{ci}(z)$ is the constant function 1 and all its derivatives vanish. This is the only place where second derivatives of the chart (the second-order part of the retraction, hence the curvature of $\mathcal M$) could have entered with a factor $m$. No expectation and no limit is taken.
- **Term (ii) is linear in the residuals $n_{ci} - mp_{ci}(z)$.** If the data come from $\theta(z)$, then $\mathbb E[n_{ci}] = mp_{ci}(z)$ and $\mathbb E[E] = 0$.

Hence, at the true gate set, $\mathbb E[-\nabla_z^2\ell] = F$ exactly. This is claim C2.

**In ambient coordinates (the form of README 4.4).** The Riemannian Hessian in the basis $T$ is $T^\top\big[\nabla^2\bar\ell - \sum_j\mu_j\nabla^2h_j\big]T$ with multipliers $\mu = (Dh^\top)^+\nabla\bar\ell$ [Source: Boumal 2023, eq. (7.81) (checked, README 4.4)]. Here $\bar\ell$ and $\bar p_{ci}$ denote the same formulas evaluated at any point $\theta$ of the ambient space ($\mathbb R^{928}$ for two qubits), on or off the model. Two facts make this form treacherous:

1. Off the model, $\sum_i\bar p_{ci}\neq1$. For example scaling the state factor $B\to(1+t)B$ gives $\sum_i\bar p_{ci} = (1+t)^2$. So $\mathbb E[\nabla^2\bar\ell]$ contains the nonzero term $m\sum_c\nabla^2\big(\sum_i\bar p_{ci}\big)$.
2. The expected Euclidean gradient is not zero either: $\mathbb E[\nabla\bar\ell] = \sum_{c,i}mp_{ci}\nabla\bar p_{ci}/p_{ci} = m\sum_c\nabla\big(\sum_i\bar p_{ci}\big)$. Its tangential part vanishes and its normal part does not. So the multipliers $\mu$ are nonzero in expectation, of order $m$.

Both terms together are $m$ times the Riemannian Hessian of the function $\theta\mapsto\sum_{c,i}p_{ci}(\theta)$. The proof uses that $\mu$ is linear in the gradient: $\mu(\bar\ell) = \sum_{c,i}(n_{ci}/p_{ci})\,\mu(\bar p_{ci})$, so

$$
\operatorname{Hess}\ell = \sum_{c,i}\frac{n_{ci}}{p_{ci}}\operatorname{Hess}p_{ci} - J_T^\top\operatorname{diag}\big(n/p^2\big)J_T,
\qquad
\operatorname{Hess}p_{ci} := T^\top\Big[\nabla^2\bar p_{ci} - \sum_j\mu_j(\bar p_{ci})\nabla^2h_j\Big]T ,
$$

and taking the expectation gives $\mathbb E[\operatorname{Hess}\ell] = m\sum_c\operatorname{Hess}\big(\sum_ip_{ci}\big) - F$. The Riemannian Hessian of a function on $\mathcal M$ depends only on its values on $\mathcal M$, and $\sum_ip_{ci}$ is constant there, so the first term is zero.

**Consequence for code.** If the expected Hessian is assembled from ambient derivatives, the Euclidean second-derivative term and the multiplier term must both be kept: each is of order $m$, and only their sum vanishes. Using $F = mJ_T^\top\operatorname{diag}(p)^{-1}J_T$ avoids the issue.

#### (b) Residual terms in the observed Hessian: they do not vanish

[Derived here.] For one dataset, (1) gives $-\nabla_z^2\ell(z) = F(z) + E(z;n)$ with $E\neq0$.

- **Size.** $n_{ci} - mp_{ci}$ has standard deviation $\sqrt{mp_{ci}(1-p_{ci})}$ (DERIVATIONS §C.6). So $E$ is of order $\sqrt m$ and $F$ of order $m$: along a direction $v$ with $v^\top F_1v$ of order 1, the relative size of $E$ is of order $1/\sqrt m$.
- **Weak directions.** Along a direction with small $v^\top F_1v$, the part of $E$ with second derivatives, $-\sum_{c,i}(n_{ci} - mp_{ci})\,v^\top\nabla^2p_{ci}\,v/p_{ci}$, does not become small with it. It dominates as soon as $v^\top F_1v\lesssim\lVert\operatorname{diag}(p)^{-1/2}\,v^\top\nabla^2p\,v\rVert/\sqrt m$. That is the condition $\kappa(v)\gtrsim\tfrac12$ of [Section 4.5](#45-three-numbers-that-measure-validity-without-fitting). It is the reason why the Hessian at noisy data has curvature of either sign along such directions (README 4.4, table).
- **Chart dependence.** At a point where $\nabla_z\ell\neq0$, the matrix $\nabla_z^2\ell$ depends on the chart. At the estimate, where $\nabla_z\ell = 0$, it does not: under a change of chart $z = \psi(w)$ the extra term is $\sum_a(\partial_a\ell)\,\nabla^2\psi_a = 0$. So the observed information at the estimate is well defined, and it equals the Riemannian Hessian in the basis $T$ for every retraction [Source: Boumal 2023, Props. 5.44–5.45 (checked, README Section 8)].
- **Which one to use.** pyGSTi uses the observed Hessian at the estimate (README 6.1). We use $F$ at the estimate. They differ by $E$, which is negligible exactly along the directions for which the asymptotic statements are valid anyway.

#### (c) Curvature terms of the intrinsic Cramér–Rao bounds: absent for scalar quantities

README 4.6 and DERIVATIONS §D.3–D.4 show that these terms belong to bounds on the covariance of the estimated *point*, measured with the logarithmic map, and that the bound for a real-valued gauge-invariant quantity has none. One addition [Derived here]: in the limit theorems of Sections 4.2 and 4.3 curvature cannot appear either. They are statements about $\sqrt m\,(\hat y - y^*)$ in a chart, and two charts differ at second order, which changes $\sqrt m\,(\hat y - y^*)$ by $O_P(1/\sqrt m)$. The curvature corrections are therefore of the same order as every other finite-$m$ effect that the first-order theory ignores. DERIVATIONS §D.5 estimated their relative size as (curvature) $\times$ (variance) with curvature $\approx3/(\sigma_3^2 + \sigma_4^2)$, where $\sigma_r$ are the singular values of the stacked Kraus operators. Requiring this to be small is the requirement that the Cramér–Rao standard deviation of a small Kraus amplitude is small compared with the amplitude: the same condition as in [Section 4.4](#44-finite-number-of-shots-the-positivity-fold).

### 4.2 Local asymptotic normality: definition and proof at a regular point

**The experiment as independent, identically distributed observations.** Fix the design. Call one *round* one shot of each of the $N$ circuits. Its outcome is $x = (x_1,\dots,x_N)$ with $x_c\in\{1,\dots,n_{\rm out}\}$, and its probability is $P_\theta(x) = \prod_cp_{c,x_c}(\theta)$. The data are $m$ independent rounds. The Fisher information of one round is $F_1 = J_T^\top\operatorname{diag}(p)^{-1}J_T$ (DERIVATIONS §C.7 with $m = 1$). The limit considered is $m\to\infty$ with the design and the true gate set fixed.

**Identifiable coordinates.** $F_1$ is singular along the gauge directions. Assume R2: $\operatorname{rank}J_T = n_{\rm id}$ in a neighbourhood of $\theta^*$. By the rank theorem [Source: Lee 2012, Thm. 4.12] there are local coordinates $(y, g)\in\mathbb R^{n_{\rm id}}\times\mathbb R^{n_{\rm gauge}}$ on $\mathcal M$ around $\theta^*$ such that the probabilities depend on $y$ only and the map $y\mapsto p$ has an injective differential. The sets $\{y = \text{const}\}$ are the gate sets that predict the same probabilities; they contain the gauge orbits and have the same dimension (README 4.3). In the coordinates $y$ the experiment is an ordinary statistical model with $n_{\rm id}$ free parameters and invertible Fisher information $F_{1,y}$. Write $y^*$ for the coordinates of the truth.

**Definition** [Source: van der Vaart 1998, Def. 7.14 (number confirmed through a secondary source; wording from memory)]. The model is *locally asymptotically normal* at $y^*$ if there are random vectors $\Delta_m$ that converge in distribution to $\mathcal N(0, F_{1,y})$ such that, for every $h\in\mathbb R^{n_{\rm id}}$ and every sequence $h_m\to h$,

$$
\ell\big(y^* + h_m/\sqrt m\big) - \ell(y^*) \;=\; h^\top\Delta_m - \tfrac12h^\top F_{1,y}h + o_P(1). \tag{2}
$$

In words: on the scale $1/\sqrt m$ around the truth, the log-likelihood is a random linear term plus a fixed quadratic term. That is the log-likelihood of observing one Gaussian vector with mean $h$ and covariance $F_{1,y}^{-1}$. It says nothing yet about any estimator.

**Proposition.** Under A1, A2, R1 ($p_{ci}(\theta^*) > 0$) and R2, the model is LAN at $y^*$, with $\Delta_m = s/\sqrt m$ the rescaled score. [Derived here. For general smooth models this is van der Vaart 1998, Thm. 7.2 (number confirmed through a secondary source). For our finite sample space the proof is elementary.]

**Proof.** Write $p_{ci}$ for $p_{ci}(y^*)$.

1. *Taylor expansion of the probabilities.* The $p_{ci}$ are polynomials in $\theta$ composed with a smooth chart, so for small $t\in\mathbb R^{n_{\rm id}}$

   $$p_{ci}(y^* + t) = p_{ci} + a_{ci}^\top t + \tfrac12t^\top B_{ci}t + O(\lVert t\rVert^3),\qquad a_{ci} = \nabla_yp_{ci},\quad B_{ci} = \nabla_y^2p_{ci}.$$

   Because $\sum_ip_{ci}(y^* + t) = 1$ for all $t$, comparing orders gives $\sum_ia_{ci} = 0$ and $\sum_iB_{ci} = 0$ for every circuit $c$.
2. *Expansion of the logarithm.* Put $t = h_m/\sqrt m$ and $x_{ci} = p_{ci}(y^* + t)/p_{ci} - 1$. By R1, $x_{ci} = O(1/\sqrt m)$, and $\log(1+x) = x - x^2/2 + O(\lvert x\rvert^3)$ gives

   $$\log\frac{p_{ci}(y^*+t)}{p_{ci}} = \frac{a_{ci}^\top h_m}{\sqrt m\,p_{ci}} + \frac1m\Big[\frac{h_m^\top B_{ci}h_m}{2p_{ci}} - \frac{(a_{ci}^\top h_m)^2}{2p_{ci}^2}\Big] + O(m^{-3/2}).$$

3. *Split the counts.* Write $n_{ci} = mp_{ci} + \sqrt m\,\epsilon_{ci}$ with $\epsilon_{ci} = (n_{ci} - mp_{ci})/\sqrt m$. It has mean 0 and variance $p_{ci}(1-p_{ci})$ for every $m$ (DERIVATIONS §C.6), so $\epsilon_{ci} = O_P(1)$ by Chebyshev's inequality.
4. *Multiply and sum over $c$ and $i$.* The log-likelihood difference is $\sum_{c,i}n_{ci}\log[p_{ci}(y^*+t)/p_{ci}]$. Its terms are:

   | Term | Value | Reason |
   |---|---|---|
   | $\sqrt m\sum_{c,i}a_{ci}^\top h_m$ | $0$ | $\sum_ia_{ci} = 0$ |
   | $\sum_{c,i}\epsilon_{ci}\,a_{ci}^\top h_m/p_{ci}$ | $h_m^\top\Delta_m$ | definition $\Delta_m = \sum_{c,i}\epsilon_{ci}\,a_{ci}/p_{ci}$ |
   | $\tfrac12\sum_{c,i}h_m^\top B_{ci}h_m$ | $0$ | $\sum_iB_{ci} = 0$ |
   | $-\tfrac12\sum_{c,i}(a_{ci}^\top h_m)^2/p_{ci}$ | $-\tfrac12h_m^\top F_{1,y}h_m$ | definition of $F_{1,y}$ |
   | $\frac1{\sqrt m}\sum_{c,i}\epsilon_{ci}\,[\dots]$ | $O_P(m^{-1/2})$ | $\epsilon_{ci} = O_P(1)$ |
   | $\sum_{c,i}n_{ci}\,O(m^{-3/2})$ | $O(m^{-1/2})$ | $n_{ci}\le m$ |

5. *The linear term.* $\Delta_m = s/\sqrt m$, because $s = \sum_{c,i}n_{ci}a_{ci}/p_{ci}$ (DERIVATIONS §C.4) and $\sum_{c,i}mp_{ci}\,a_{ci}/p_{ci} = 0$. It is $1/\sqrt m$ times a sum of $m$ independent, identically distributed, mean-zero vectors, one per round, with covariance $F_{1,y}$. By the multivariate central limit theorem it converges in distribution to $\mathcal N(0, F_{1,y})$. (Standard; van der Vaart 1998, Ch. 2.)
6. Since $h_m\to h$ and $\Delta_m = O_P(1)$, replacing $h_m$ by $h$ changes the result by $o_P(1)$. This gives (2). ∎

**What the proof shows about finite $m$.** The neglected terms are of relative order $1/\sqrt m$, with coefficients $B_{ci}/p_{ci}$, $(a_{ci}/p_{ci})^2$ and third derivatives. They are large in two situations: small probabilities (R1), and directions along which the second derivatives $B_{ci}$ are not small compared with the first derivatives $a_{ci}$ (R3). Both occur here.

### 4.3 Asymptotic normality of the estimate and of a Pauli transfer matrix element

LAN concerns the likelihood. A statement about the estimate needs more.

**Theorem** [Source: van der Vaart 1998, Thm. 5.39 (number and content confirmed through secondary sources that paraphrase it; not checked against the book)]. Let the model be differentiable in quadratic mean at an interior point $y^*$ of the parameter set. Suppose there is a function $\dot\ell(x)$ with finite second moment such that $\lvert\log P_{y_1}(x) - \log P_{y_2}(x)\rvert\le\dot\ell(x)\lVert y_1 - y_2\rVert$ for all $y_1, y_2$ in a neighbourhood of $y^*$. If the Fisher information at $y^*$ is nonsingular and the maximum-likelihood estimate $\hat y_m$ is consistent (converges in probability to $y^*$), then

$$
\sqrt m\,(\hat y_m - y^*) = F_{1,y}^{-1}\,\Delta_m + o_P(1),\qquad\text{so}\qquad \sqrt m\,(\hat y_m - y^*)\ \to\ \mathcal N\big(0, F_{1,y}^{-1}\big)\ \text{in distribution.}
$$

**Checking the hypotheses for our model** [Derived here].

| Hypothesis | Holds because | Uses |
|---|---|---|
| interior point | $y^*$ lies inside the chart of Section 4.2 | R2 |
| differentiable in quadratic mean, i.e. $\sum_x\big[\sqrt{P_{y^*+h}(x)} - \sqrt{P_{y^*}(x)} - \tfrac12h^\top\nabla\log P_{y^*}(x)\sqrt{P_{y^*}(x)}\big]^2 = o(\lVert h\rVert^2)$ | the sum has finitely many terms ($n_{\rm out}^N$ outcomes of a round). Each $\sqrt{P_y(x)}$ is differentiable at $y^*$ with gradient $\tfrac12\nabla P/\sqrt P = \tfrac12(\nabla\log P)\sqrt P$, so each bracket is $o(\lVert h\rVert)$ | R1 |
| Lipschitz bound | $\log P_y(x) = \sum_c\log p_{c,x_c}(y)$ has a gradient bounded by $\sum_c\max_i\lVert\nabla_yp_{ci}\rVert/p_{ci}$, which is finite on a closed ball around $y^*$ on which all $p_{ci}\ge\tfrac12\min p_{ci}(y^*)$. The mean value theorem then gives the bound with a constant $\dot\ell$ | R1 |
| nonsingular Fisher information | $F_{1,y} = (\partial p/\partial y)^\top\operatorname{diag}(p)^{-1}(\partial p/\partial y)$ with $\partial p/\partial y$ injective | R1, R2 |
| consistency | **not proven here.** It needs E1: the estimate is the global maximum, and the truth is the only maximizer of the expected log-likelihood up to gauge | E1 |

**From parameters to a Pauli transfer matrix element.** Let $f$ be gauge invariant and differentiable at $\theta^*$, for example a gauge-fixed element $R_{g,ij}\circ\Pi$ under G1. The delta method [Source: van der Vaart 1998, Thm. 3.1] gives $\sqrt m\,(f(\hat\theta) - f(\theta^*))\to\mathcal N\big(0, \nabla_yf^\top F_{1,y}^{-1}\nabla_yf\big)$. This variance equals $\mathbf d_f^\top F_1^+\mathbf d_f$ computed in the tangent coordinates $z$: in the coordinates $(y, g)$ the Fisher information is block diagonal with blocks $F_{1,y}$ and $0$, the gradient of $f$ is $(\nabla_yf, 0)$, and the quantity $\mathbf d^\top F^+\mathbf d$ is the same in all coordinates (PTM_PRECISION 3.2, item 1). Hence

$$
f(\hat\theta) - f(\theta^*)\ \approx\ \mathcal N\big(0, \sigma_f^2\big),\qquad \sigma_f^2 = \mathbf d_f^\top F^+\mathbf d_f\qquad(m\to\infty),
$$

which is claim C5 and eq. (4) of PTM_PRECISION.

**Status.** This is a theorem for $m\to\infty$ at a fixed regular truth. It says nothing about how large $m$ must be, and the convergence is not uniform: for every $m$ there are regular truths, those closer to the singular set than the statistical error, for which the approximation is poor. Sections 4.4 and 4.5 quantify this.

### 4.4 Finite number of shots: the positivity fold

**Where the singular set comes from.** A gate enters the probabilities through $\sum_rK_rXK_r^\dagger$, which is quadratic in each Kraus operator. In the notation of PTM_PRECISION 3.4, a small Kraus operator of amplitude $a$ contributes a rate $r = a^2$, and near the truth the probabilities are linear in the rate. The map $a\mapsto a^2$ is a fold: the amplitude ranges over all real numbers and the rate over $r\ge0$ only. The model manifold $\mathcal M$ has no boundary, but its image in the space of channels does, namely the boundary of the set of completely positive maps of Kraus rank at most $r_K$. The points with $a = 0$ are the singular points of R2.

**One-parameter model** [Derived here]. One rate $r\ge0$ with Cramér–Rao standard deviation $\sigma_r$, in the Gaussian limit: the data are summarized by $X\sim\mathcal N(r, \sigma_r^2)$. The likelihood in the amplitude, $\exp[-(X - a^2)^2/2\sigma_r^2]$, is maximal at $a^2 = \max(X, 0)$. So the maximum-likelihood estimate of the rate is

$$\hat r = \max(X, 0).$$

Let $u = r/\sigma_r$ be the *rate resolution* and $t = (X - r)/\sigma_r\sim\mathcal N(0,1)$, so that $\hat r = r + \sigma_rt$ for $t > -u$ and $\hat r = 0$ otherwise.

1. *Probability of a collapsed estimate.* $P(\hat r = 0) = P(t\le-u) = \Phi(-u)$.
2. *Bias.* $\mathbb E[\hat r] = \int_{-u}^\infty(r + \sigma_rt)\phi(t)\,dt = r\,\Phi(u) + \sigma_r\phi(u)$, using $\int_{-u}^\infty t\,\phi(t)\,dt = \phi(u)$. So

   $$\mathbb E[\hat r] - r = \sigma_r\big[\phi(u) - u\,\Phi(-u)\big].$$

3. *Mean squared error.* $\mathbb E[(\hat r - r)^2] = \sigma_r^2\int_{-u}^\infty t^2\phi(t)\,dt + r^2\,\Phi(-u)$. Integrating by parts with $\phi' = -t\phi$: $\int_{-u}^\infty t^2\phi\,dt = [-t\phi]_{-u}^\infty + \int_{-u}^\infty\phi\,dt = -u\,\phi(u) + \Phi(u)$. So

   $$\frac{\mathbb E[(\hat r - r)^2]}{\sigma_r^2} = M(u) := \Phi(u) - u\,\phi(u) + u^2\,\Phi(-u).$$

   Its derivative is $M'(u) = \phi - \phi + u^2\phi + 2u\,\Phi(-u) - u^2\phi = 2u\,\Phi(-u)\ge0$. So $M$ increases from $M(0) = \tfrac12$ to $M(\infty) = 1$: **the mean squared error never exceeds the Cramér–Rao value.**
4. *Coverage of the interval $\hat r\pm1.96\,\sigma_r$.* If $t > -u$ the interval contains $r$ when $\lvert t\rvert\le1.96$. If $t\le-u$ the estimate is 0 and the interval contains $r$ when $r\le1.96\,\sigma_r$, i.e. $u\le1.96$. Therefore

   $$\text{coverage} = \begin{cases}P(-u < t\le1.96) + P(t\le-u) = \Phi(1.96) = 97.5\% & u\le1.96,\\ P(\lvert t\rvert\le1.96) = 95\% & u > 1.96.\end{cases}$$

   **The interval is never too narrow.**

| $u = r/\sigma_r$ | $P(\hat r = 0)$ | bias$/\sigma_r$ | mean squared error$/\sigma_r^2$ | coverage of $\pm1.96\sigma_r$ |
|---|---|---|---|---|
| 0 | 0.50 | 0.399 | 0.50 | 97.5% |
| 0.5 | 0.31 | 0.198 | 0.59 | 97.5% |
| 1 | 0.16 | 0.083 | 0.76 | 97.5% |
| 2 | 0.023 | 0.0085 | 0.96 | 95.0% |
| 3 | 0.0013 | 0.0004 | 0.998 | 95.0% |

**Reading.**

- The row $u = 0$ is the toy model of README 5.5. For $u\ge2$ the estimate is Gaussian for practical purposes: 2% of collapsed estimates, 4% deficit in mean squared error. For $u\le1$ it is not. **Criterion: a rate is resolved when $u\ge2$, better 3.** The notebook's threshold $\sigma(\mu)/\mu\le0.5$ (PTM_PRECISION 5.2) is $u\ge2$.
- In Kraus coordinates the Fisher information of the amplitude is $F_a = 4a^2/\sigma_r^2$ (PTM_PRECISION 3.4), so $u = \tfrac12\,a\sqrt{F_a}$: the amplitude must be at least 4 to 6 times its own Cramér–Rao standard deviation.
- **How many shots that is** [Derived here]. Take a two-outcome circuit with $p = p_0 + \beta r$, so $\sigma_r = \sqrt{p(1-p)}/(\lvert\beta\rvert\sqrt m)$ (DERIVATIONS §C.8).
  - If the outcome occurs only because of the error ($p_0 = 0$, $p = \beta r\ll1$): $u = \sqrt{m\beta r}$, the square root of the expected number of error events. Resolution needs **4 to 9 observed error events**, and the shots needed grow like $1/r$.
  - If the error changes a probability that is already of order one: $u = \lvert\beta\rvert r\sqrt m/\sqrt{p_0(1-p_0)}$, and the shots needed grow like $1/r^2$.
- **Several small eigenvalues.** A gate has $r_K - 1$ small Choi eigenvalues. In the Gaussian limit the constrained estimate clips the negative eigenvalues of a noisy Hermitian block [Source: Scholten & Blume-Kohout 2018, for quantum states]. The criterion then applies to the smallest eigenvalue, and it gets stricter when small eigenvalues are close to each other, because a single eigenvalue is not a differentiable function at a degeneracy. `small_choi_gradient` in `scripts/ptm_design.py` raises an error in that case.
- **Plug-in at a collapsed estimate (P1).** At $\hat a = 0$ the amplitude direction has $F_a = 0$ and the gradient of every Pauli transfer matrix element along it is also 0 (PTM_PRECISION, eq. (3)). The pseudo-inverse then drops the direction, and its contribution to $\sigma_f^2$ is counted as zero. The correct contribution is the limit $\lim_{a\to0}(2\gamma a)^2/(4a^2/\sigma_r^2) = \gamma^2\sigma_r^2$, which a small nonzero amplitude reproduces. This is why the evaluation point should be the estimate plus a small perturbation (README 2.1, recommendation 2). The perturbation restores one direction per collapsed Kraus operator, the direction of the perturbation; whether that is enough has not been tested ([Section 7.3](#73-what-still-has-to-be-done), item 6).

**General theory** [Source: Chernoff 1954; Self & Liang 1987 (abstract checked); Geyer 1994 (abstract checked)]. When the truth is on the boundary of the parameter set, the limiting distribution of the maximum-likelihood estimate is that of the projection of a Gaussian vector onto the tangent cone of the parameter set, in the metric of the Fisher information, and likelihood-ratio statistics become mixtures of chi-squared distributions. The model above is the one-dimensional case with cone $[0,\infty)$. When the true rate is exactly 0, the likelihood-ratio statistic is $\max(X,0)^2/\sigma_r^2$: it is 0 with probability $\tfrac12$ and $\chi^2_1$-distributed otherwise, instead of $\chi^2_1$ always.

### 4.5 Three numbers that measure validity without fitting

All three are computed from $F$ and derivatives of $p$ at the evaluation point, like $\sigma_f$ itself.

**(a) Rate resolution** $u_j = \mu_j/\sigma(\mu_j)$, with $\sigma(\mu_j)^2 = \mathbf d_{\mu_j}^\top F^+\mathbf d_{\mu_j}$, for every small eigenvalue: of the Choi matrix of each gate (implemented: `small_choi_gradient`, `rate_sigma1` in `scripts/ptm_design.py`), of the state $\rho$ and of each POVM effect (not implemented). All are gauge invariant, because the Hilbert-space gauge acts on them by unitary conjugation. *Pass:* $\min_ju_j\ge2$.

**(b) Nonlinearity ratio** [Derived here; in the spirit of the relative curvature measures of Bates & Watts 1980, whose definitions were not checked]. For a tangent direction $v$ consider the curve $\theta_v(t) = R_\theta(t\,Tv)$ and the whitened change of the predictions along it,

$$w(t) = \sqrt m\,\operatorname{diag}(p)^{-1/2}\big[p(\theta_v(t)) - p\big] = t\,\dot w + \tfrac12t^2\,\ddot w + O(t^3),\qquad \dot w = \sqrt m\,Wv,\quad \lVert\dot w\rVert^2 = v^\top Fv .$$

The step $t_1 = 1/\sqrt{v^\top Fv}$ is one Cramér–Rao standard deviation along $v$: it moves the predictions by exactly one unit of shot noise, $\lVert t_1\dot w\rVert = 1$. Define the ratio of the second-order to the first-order change at that step:

$$
\kappa(v) = \frac{\lVert\tfrac12t_1^2\,\ddot w\rVert}{\lVert t_1\dot w\rVert} = \frac{\lVert\ddot w\rVert}{2\,v^\top Fv} = \frac{1}{\sqrt m}\cdot\frac{\big\lVert\operatorname{diag}(p)^{-1/2}\,\tfrac{d^2}{dt^2}p(\theta_v(t))\big\rvert_{t=0}\big\rVert}{2\,v^\top F_1v}.
$$

- $\kappa\propto1/\sqrt m$, so the shots needed for $\kappa\le\kappa_{\max}$ are $m\ge(\kappa(v)\rvert_{m=1}/\kappa_{\max})^2$.
- *Check on the fold.* For $p = (p_0 + \beta a^2,\ 1 - p_0 - \beta a^2)$ and $v$ the amplitude direction: $\frac{d^2p}{da^2} = (2\beta, -2\beta)$, $\lVert\operatorname{diag}(p)^{-1/2}(2\beta, -2\beta)\rVert = 2\lvert\beta\rvert/\sqrt{p_0(1-p_0)}$ and $v^\top F_1v = 4\beta^2a^2/(p_0(1-p_0))$. So $\kappa = \sqrt{p_0(1-p_0)}/(4\lvert\beta\rvert a^2\sqrt m) = \sigma_r/(4r) = 1/(4u)$. The criterion $u\ge2$ is $\kappa\le1/8$.
- *Per quantity.* For a quantity $f$ take $v_f = F^+\mathbf d_f$. Under the Gaussian law of Section 4.3, $\operatorname{Cov}(\hat z, f(\hat\theta)) = F^+\mathbf d_f$, so a one-standard-deviation fluctuation of $f$ comes with the mean displacement $t_1v_f$ of the gate set, $t_1 = 1/\sigma_f$. $\kappa(v_f)$ says whether the model is linear over that displacement. *Pass:* $\kappa(v_f)\le0.1$.
- It also catches weak directions that are not folds, for example near the identifiability threshold $N\approx n_{\rm id}/(n_{\rm out}-1)$.
- $\kappa$ is defined with the retraction in Kraus coordinates. That is intended: the positivity constraint is built into these coordinates. A version that needs no second derivatives is $\lVert w(t_1) - t_1\dot w\rVert$, from two evaluations of the probabilities.

**(c) Rare-outcome share.** Merge, within each circuit, all outcomes with $mp_{ci} < 5$ into one bin and recompute the width, $\sigma_f^{\rm merged}$. Merging a set $S$ of outcomes replaces their contribution to $F$, $\sum_{i\in S}J_{ci}^\top J_{ci}/p_{ci}$, by $(\sum_{i\in S}J_{ci})^\top(\sum_{i\in S}J_{ci})/\sum_{i\in S}p_{ci}$, where $J_{ci}$ is row $(c,i)$ of $J_T$. The second is never larger: for any vector $x$, with $\alpha_i = J_{ci}x$, the Cauchy–Schwarz inequality gives $(\sum_i\alpha_i)^2 = (\sum_i\frac{\alpha_i}{\sqrt{p_i}}\sqrt{p_i})^2\le(\sum_i\alpha_i^2/p_i)(\sum_ip_i)$. So $\sigma_f^{\rm merged}\ge\sigma_f$ [Derived here]. *Pass:* $\sigma_f^{\rm merged}/\sigma_f$ close to 1. Otherwise the information about $f$ sits in outcomes whose counts are far from Gaussian.

**A fourth number, which needs refits** [Derived here]. In a regular model, $\Lambda = 2[\ell(\hat\theta) - \ell(\theta^*)]$ is asymptotically $\chi^2_{n_{\rm id}}$ (Wilks; Section 4.7), with mean $n_{\rm id}$. The measured mean of $\Lambda$ is therefore an *effective number of fitted parameters*, and its ratio to $n_{\rm id}$ a one-number summary of how far the experiment is from the asymptotic regime. In a parametric bootstrap it is available without the truth: for each replicate, $2[\ell_b(\hat\theta_b) - \ell_b(\hat\theta)]$, where $\ell_b$ is the log-likelihood of the replicate's data, $\hat\theta_b$ its refit and $\hat\theta$ the gate set the replicate was simulated from.

### 4.6 Optimizer tolerance

[Derived here.] Let $\tilde z$ be the point returned by the optimizer, $\hat z$ the exact maximum, and $g = \nabla_z\ell(\tilde z)$ the Riemannian gradient of the log-likelihood at $\tilde z$ in the basis $T$. Near the maximum, $\ell(z)\approx\ell(\tilde z) + g^\top(z - \tilde z) - \tfrac12(z - \tilde z)^\top F(z - \tilde z)$, with the expected information in place of the observed one (Section 4.1 (b)). The likelihood is gauge invariant, so $g$ has no gauge component and $g\in\operatorname{range}F$.

1. Maximizing the quadratic model gives $\hat z - \tilde z = F^+g$ up to gauge.
2. The remaining gain is $\epsilon = \ell(\hat z) - \ell(\tilde z) = \tfrac12g^\top F^+g$, in nats.
3. A gauge-invariant quantity $f$ changes by $\mathbf d_f^\top F^+g$. By the Cauchy–Schwarz inequality in the inner product $\langle a, b\rangle = a^\top F^+b$,

   $$\big\lvert f(\hat\theta) - f(\tilde\theta)\big\rvert \approx \big\lvert\mathbf d_f^\top F^+g\big\rvert\ \le\ \sqrt{\mathbf d_f^\top F^+\mathbf d_f}\,\sqrt{g^\top F^+g} = \sigma_f\sqrt{2\epsilon}.$$

| remaining gain $\epsilon$ (nat) | worst-case shift of any $f$ |
|---|---|
| 1 (`absolute_tolerance=1`, script 07 default) | $1.41\,\sigma_f$ |
| 0.02 | $0.2\,\sigma_f$ |
| 0.005 | $0.1\,\sigma_f$ |
| $10^{-4}$ (converged runs) | $0.014\,\sigma_f$ |

- `absolute_tolerance` in `trust_region.py` is an extrapolated *estimate* of the remaining decrease, not a bound.
- **Certificate.** After the fit, compute $\mathbf d_f^\top F^+g$ for the quantities of interest, with $g$ the Riemannian gradient at the returned point. It is the first-order distance of $f$ from its value at the exact maximum, and it should be well below $\sigma_f$.
- The statistical error is of order $\sigma_f$ regardless, and the default and converged runs gave the same statistics (README 5.5). So the worst case did not occur at the reference design.

### 4.7 Wrong model: pseudo-truth, sandwich covariance, goodness of fit

[Derived here, except where a source is given.] Suppose A1 fails: the device produces multinomial counts with probabilities $q_{ci}$, and no gate set in the model has $p(\theta) = q$.

**What the estimate converges to.** $\mathbb E[\ell(\theta)] = m\sum_{c,i}q_{ci}\log p_{ci}(\theta) = -m\sum_c\mathrm{KL}\big(q_c\,\Vert\,p_c(\theta)\big) + \text{const}$, where $\mathrm{KL}(q\Vert p) = \sum_iq_i\log(q_i/p_i)$ is the Kullback–Leibler divergence. Its maximizer $\theta^\dagger$ is the *pseudo-truth*. The maximum-likelihood estimate converges to it [Source: White 1982, Thm. 2.2 (number confirmed through a secondary source)]. Define the relative misfit $\eta_{ci} = (q_{ci} - p_{ci}(\theta^\dagger))/p_{ci}(\theta^\dagger)$. The first-order condition for $\theta^\dagger$ is $\sum_{c,i}q_{ci}\,\partial_kp_{ci}/p_{ci} = \sum_{c,i}\eta_{ci}\,\partial_kp_{ci} = 0$ (using $\sum_i\partial_kp_{ci} = 0$), i.e. $J_T^\top\eta = 0$.

**Its covariance.** For an estimator defined by a vanishing score, $\sqrt m\,(\hat y - y^\dagger)\to\mathcal N\big(0,\ m\,A^{-1}BA^{-1}\big)$ with $A = -\mathbb E[\nabla^2\ell(\theta^\dagger)]$ and $B = \operatorname{Cov}(\nabla\ell(\theta^\dagger))$, both taken under the true distribution of the data [Source: van der Vaart 1998, Thm. 5.21 (checked through lecture notes that quote it with its number); White 1982, Thm. 3.2 (number confirmed through a secondary source)]. Compute both from eq. (1) and DERIVATIONS §C.6 with $\mathbb E[n_{ci}] = mq_{ci}$, $\operatorname{Cov}(n_{ci}, n_{cj}) = m(q_{ci}\delta_{ij} - q_{ci}q_{cj})$ and $q_{ci} = p_{ci}(1 + \eta_{ci})$:

$$
\begin{aligned}
A_{kl} &= m\sum_{c,i}q_{ci}\Big[\frac{\partial_kp_{ci}\partial_lp_{ci}}{p_{ci}^2} - \frac{\partial_k\partial_lp_{ci}}{p_{ci}}\Big]
= F_{kl} + m\sum_{c,i}\eta_{ci}\frac{\partial_kp_{ci}\partial_lp_{ci}}{p_{ci}} - m\sum_{c,i}\eta_{ci}\,\partial_k\partial_lp_{ci},\\
B_{kl} &= m\sum_c\Big[\sum_i\frac{\partial_kp_{ci}\partial_lp_{ci}}{p_{ci}^2}q_{ci} - \Big(\sum_i\frac{\partial_kp_{ci}}{p_{ci}}q_{ci}\Big)\Big(\sum_j\frac{\partial_lp_{cj}}{p_{cj}}q_{cj}\Big)\Big]
= F_{kl} + m\sum_{c,i}\eta_{ci}\frac{\partial_kp_{ci}\partial_lp_{ci}}{p_{ci}} - m\sum_c\Big(\sum_i\eta_{ci}\partial_kp_{ci}\Big)\Big(\sum_j\eta_{cj}\partial_lp_{cj}\Big).
\end{aligned}
$$

Here $F$ is evaluated at $\theta^\dagger$, and $\sum_i\partial_k\partial_lp_{ci} = 0$ and $\sum_i\partial_kp_{ci} = 0$ were used. For $\eta = 0$ both reduce to $F$.

**Three consequences.**

1. **The width is right to first order, the target is not.** The common correction satisfies $-\eta_{\max}F\preceq mJ_T^\top\operatorname{diag}(\eta/p)J_T\preceq\eta_{\max}F$ with $\eta_{\max} = \max\lvert\eta_{ci}\rvert$, and the last term of $B$ is of second order in $\eta$. So along directions where the second-derivative term of $A$ is negligible, $A^{-1}BA^{-1} = F^{-1}(1 + O(\eta_{\max}))$: a 5% relative misfit changes the variance by about 5%. But the interval is centred on the pseudo-truth. The distance between $\theta^\dagger$ and whatever one would call the true gates is a systematic error that no Fisher quantity contains.
2. **Weak directions absorb the misfit.** The term $m\sum_{c,i}\eta_{ci}\,\partial_k\partial_lp_{ci}$ in $A$ is not bounded by $F$. Along a direction $v$ it competes with $v^\top Fv$ when $v^\top F_1v\lesssim\eta_{\max}\lVert v^\top\nabla^2p\,v\rVert$. Both sides scale with $m$, so more shots do not help. These are the directions with a large nonlinearity ratio, i.e. the unresolved incoherent rates. In rate language: a misfit of relative size $\eta$ in a probability with sensitivity $\beta$ shifts the fitted rate by about $\eta p/\beta$, which is not small compared with a rate of $10^{-3}$ even for $\eta$ of a few percent.
3. **Overdispersion scales the width.** If the counts have covariance $\omega\,m(p_{ci}\delta_{ij} - p_{ci}p_{cj})$ with a dispersion factor $\omega > 1$ and the mean is right, then $A = F$, $B = \omega F$ and the covariance is $\omega F^+$ [Source: McCullagh & Nelder 1989, quasi-likelihood]. *Slow drift produces this.* If a two-outcome probability $p$ is constant during the $m$ shots of a circuit but varies between repetitions of the experiment with mean $\bar p$ and variance $\operatorname{Var}(p)$, the law of total variance gives

   $$\operatorname{Var}(n) = \mathbb E[mp(1-p)] + \operatorname{Var}(mp) = m\bar p(1-\bar p) + m(m-1)\operatorname{Var}(p),\qquad \omega = 1 + (m-1)\frac{\operatorname{Var}(p)}{\bar p(1-\bar p)}.$$

   For $m = 1000$, a drift of the probability by 0.01 (standard deviation) at $\bar p(1-\bar p) = 0.2$ gives $\omega = 1.5$. The factor grows with $m$.

**Detecting a wrong model.** The deviance is

$$\mathcal D(\theta) = 2\big[\ell_{\rm sat} - \ell(\theta)\big] = 2m\sum_c\mathrm{KL}\big(f_c\,\Vert\,p_c(\theta)\big),\qquad \ell_{\rm sat} = \sum_{c,i}n_{ci}\log f_{ci}.$$

In the code, $\mathcal D(\hat\theta) = 2\times$ (final cost $-$ `compute_cost_floor`), see `trust_region.py`. If the model is correct and regular, Wilks' theorem gives $\mathcal D(\hat\theta)\to\chi^2_k$ with $k = (n_{\rm out}-1)N - n_{\rm id}$, the number of independent data minus the number of fitted parameters [Source: Wilks 1938; applied to gate set tomography in Nielsen et al. 2021, Sec. 6.1 (checked), with $N_\sigma = (\mathcal D - k)/\sqrt{2k}$]. Wilks' theorem needs the same conditions as Section 4.3, and [Section 5](#5-new-numerical-evidence-goodness-of-fit-of-the-saved-estimates) shows that it fails here.

### 4.8 What remains true when asymptotic normality fails

| Statement | Valid for | Reference |
|---|---|---|
| $F$ is the Fisher information; its spectrum and rank compare designs | every $m$, every estimator | C1; README 4.7, 6.1 |
| $\operatorname{Var}(\hat f)\ge\mathbf d_f^\top F^+\mathbf d_f$ | every $m$, **unbiased** estimators only. Ours is biased (README 5.5) | DERIVATIONS §D.2 |
| van Trees bound | every $m$, every estimator, error averaged over a prior | DERIVATIONS §E |
| No estimator beats $\mathbf d_f^\top F^+\mathbf d_f$ uniformly over a neighbourhood of radius $c/\sqrt m$ of the truth, as $m\to\infty$ and then $c\to\infty$ (local asymptotic minimax theorem) | $m\to\infty$, every estimator | [Source: van der Vaart 1998, Thm. 8.11 (number confirmed through a secondary source; form for functions of the parameter checked against Duchi 2021, slides 15)] |

The last line does not contradict README 5.6, where the error of the estimate was 0.39 times the full bound. The theorem needs the neighbourhood of radius $c/\sqrt{\lambda_k}$ along $v_k$ to lie where the expansion (2) holds. For the weak directions that radius exceeds the size of the manifold.

**Why the Fisher width tends to be conservative: a partial result** [Derived here]. In the Gaussian limit the unconstrained estimate is $X\sim\mathcal N(h, F_1^{-1})$ and the constrained one is the point of the constraint set $C$ closest to $X$ in the norm $\lVert x\rVert_F^2 = x^\top F_1x$ (Section 4.4, "General theory"). If $C$ is convex and closed and the truth $h$ is in $C$, then for every realization

$$\lVert P_C(X) - h\rVert_F\ \le\ \lVert X - h\rVert_F .$$

*Proof.* Let $\langle\cdot,\cdot\rangle$ be the inner product of the norm and $P = P_C(X)$. For $y\in C$ and $t\in(0,1]$ the point $P + t(y - P)$ is in $C$, so $\lVert X - P - t(y - P)\rVert^2\ge\lVert X - P\rVert^2$. Expanding gives $-2t\langle X - P, y - P\rangle + t^2\lVert y - P\rVert^2\ge0$; dividing by $t$ and letting $t\to0$ gives $\langle X - P, y - P\rangle\le0$. With $y = h$: $\lVert X - h\rVert^2 = \lVert X - P\rVert^2 + \lVert P - h\rVert^2 + 2\langle X - P, P - h\rangle\ge\lVert P - h\rVert^2$. ∎

So a convex constraint never increases the total error measured in units of shot noise. The limits of this result:

- It bounds the total, not each quantity. For one Pauli transfer matrix element the constrained error can exceed the unconstrained one. That it did not in README 6.3 and PTM_PRECISION 3.6 is an empirical finding.
- The set of completely positive trace-preserving maps is convex. The set of those with Kraus rank at most $r_K < d^2$ is not.

**Known ways in which the Fisher interval can be too narrow:** a wrong model (4.7); a local maximum (E1); a loose optimizer tolerance (4.6); evaluation at a collapsed estimate without perturbation (4.4); a design below or near the identifiability threshold, where $F^+$ silently drops non-estimable components (PTM_PRECISION 2.4).

---

## 5. New numerical evidence: goodness of fit of the saved estimates

[Numerical: 19; output in `data/19_goodness_of_fit_check_output.txt`. No new fits: the script evaluates the log-likelihood at the 40 default and 20 converged estimates of script 07 ($N = 400$, $m = 1000$, noise strength 0.05) and at the truth.]

| Quantity | Wilks prediction (mean, standard deviation) | default (40) | converged (20) |
|---|---|---|---|
| deviance at the truth, $\mathcal D(\theta^*)$ | $\chi^2_{1200}$: 1200, 49 | 1248, 49 | 1245, 46 |
| deviance at the estimate, $\mathcal D(\hat\theta)$ | $\chi^2_{576}$: 576, 34 | **1007**, 44 | **1003**, 41 |
| $N_\sigma = (\mathcal D(\hat\theta) - 576)/\sqrt{2\cdot576}$ | of order 1 | **12.7** (9.5 to 14.9) | **12.6** (9.5 to 14.8) |
| likelihood-ratio statistic $\Lambda = \mathcal D(\theta^*) - \mathcal D(\hat\theta)$ | $\chi^2_{624}$: 624, 35 | **241**, 15 | **242**, 17 |

Also: the smallest true outcome probability is $1.5\times10^{-4}$; 470 of the 1600 outcomes have an expected count below 5 and 90 below 1; on average 97 outcomes per dataset are never observed.

**Reading.**

1. **Effective number of fitted parameters: about 241 of 624** (Section 4.5). Fully converged runs give the same number, so this is not early stopping. It is the global counterpart of the per-direction table of README 5.5.
2. **The Wilks reference distribution for the goodness of fit is wrong here.** The data were simulated from the model, yet $N_\sigma\approx13$ in every one of the 60 fits. The deviance is about $1200 + 48 - 241 = 1007$, not $1200 - 624 = 576$: the fit removes 241 units of deviance, not 624.
3. **Even without any fit the chi-squared approximation is off by 4%**: the deviance at the truth has mean 1248 instead of 1200 (the standard error of the mean over 40 datasets is 8). This is the effect of the rare outcomes (R1).
4. **Consequence for real data.** A model-violation test must take its reference distribution from simulation: the deviances of parametric-bootstrap refits (README 6.4), which are needed for the error bars anyway.
5. **No outliers.** All 60 deviances lie within 3 standard deviations of their mean, so no run ended in a different basin (E1).

**Caveat.** One truth, one design, one noise level. At a design with resolved rates the prediction is that $\Lambda$ approaches 624 and $N_\sigma$ approaches 0. That is untested.

---

## 6. Realistic noise: which assumption each mechanism breaks

The entries are consequences of Sections 4.4 and 4.7. Device numbers are not assumed; where an order of magnitude matters it is stated as a condition.

| Mechanism | Inside the model? | Assumption at risk | Consequence | Action |
|---|---|---|---|---|
| Coherent errors: over-rotation, axis tilt, static crosstalk that is part of the $n$-qubit unitary | Yes. They move the dominant Kraus operator | none | Strong, regular directions. The asymptotic regime is reached with moderate $m$: mean $e_k^2\approx0.85$–$0.9$ for $k\le100$ (README 5.5) | none |
| Markovian decoherence: relaxation, dephasing, depolarization | Inside the completely positive trace-preserving maps, but generically of full Kraus rank $d^2$, with a few Choi eigenvalues of the order of the infidelity and a tail far below | **A1** if $r_K$ is smaller than the true rank; **R3** | Truncation bias of the order of the discarded Choi weight (compare the cap of PTM_PRECISION 3.5). Unresolved rates collapse (4.4) and absorb misfit (4.7) | choose $r_K$ by comparing deviances across ranks, calibrated by simulation; design for $u\ge2$ if $\delta$ is at or below the incoherent infidelity |
| Readout error | Yes (general POVM) | R3 for small eigenvalues of the effects | as for rates | include effects in the rate resolution |
| State-preparation error | Yes (mixed state) | R3 for small eigenvalues of $\rho$ | as for rates | include $\rho$ in the rate resolution |
| Leakage out of the qubit space | No ($d$ is fixed) | **A1** | pseudo-truth; shows up as misfit that grows with depth | goodness of fit versus depth |
| Slow drift between circuits or repetitions | No | **A2** | dispersion factor $\omega = 1 + (m-1)\operatorname{Var}(p)/(\bar p(1-\bar p))$, growing with $m$ | interleave the circuits in time; estimate $\omega$ from the deviance |
| Gates that depend on their neighbours in the circuit, non-Markovian noise | No | **A1** | pseudo-truth | goodness of fit |
| Few shots on rare outcomes | — | **R1** | non-Gaussian counts; chi-squared references off | rare-outcome share (4.5) |

**Two regimes for the incoherent part** (PTM_PRECISION 3.5, 5.2).

- *Target error $\delta$ well above the incoherent infidelity per gate.* To first order, losing the whole incoherent part of a gate moves each of its Pauli transfer matrix elements by at most 3 times that infidelity (PTM_PRECISION 3.5, with the caveats stated there). The design only has to resolve the coherent part and SPAM. The LAN width is conservative and the van Trees width is closer to reality.
- *$\delta$ at or below the incoherent infidelity.* The rates must be resolved, $u\ge2$. Then, and only then, is the LAN width expected to be calibrated. By Section 4.4 the cost grows like $1/\mu$ to $1/\mu^2$ as the rates $\mu$ get smaller.

---

## 7. Verdict and plan

### 7.1 Purpose 1: design before the experiment

**What is justified.**

- $F$ at a prior guess is exact for that guess (C1), and it changed little between the prior guess and the truth: $\max_f\sigma_f$ within 5% for $N\ge300$ (PTM_PRECISION 5.1).
- $\sigma_f$ is the exact large-$m$ standard deviation of your estimate (Section 4.3). At finite $m$ it has been conservative in every test, for a reason that is proven in the fold model (Section 4.4) and partly in general (Section 4.8).
- The van Trees width is a rigorous lower bound on the prior-averaged error of any estimator (DERIVATIONS §E), given the prior width $\tau$.

**Procedure.**

1. Choose a prior guess that resembles the device: coherent errors, incoherent infidelity per gate, SPAM errors. Repeat the scan for two or three guesses that bracket the incoherent infidelity, because the weak directions depend on it.
2. For each candidate $(N, m, L_{\max}$, gate set $)$ compute per element $\sigma_f$, $\sigma_f^{\rm vT}(\tau)$ and the estimability check (PTM_PRECISION 2.8).
3. Compute the validity numbers of Section 4.5: $\min_ju_j$, $\kappa(v_f)$ and the rare-outcome share.
4. Decide:
   - if $u\ge2$, $\kappa\le0.1$ and the rare-outcome share is small: the design is in the asymptotic regime; require $1.96\max_f\sigma_f\le\delta$;
   - otherwise, if $\delta$ is well above 3 times the incoherent infidelity: require $1.96\max_f\sigma_f^{\rm vT}\le\delta$ and treat $\sigma_f$ as the conservative end;
   - otherwise change the design (deeper sequences first, then more circuits, PTM_PRECISION 5.1) until the first case applies.
5. Keep $N$ at about 2 times the threshold $n_{\rm id}/(n_{\rm out}-1)$ or more. Just above the threshold the weakest directions are very badly conditioned (README 5.2), and PTM_PRECISION 5.1 found $N\ge400$ sensible for a threshold of 208.
6. Confirm the chosen design once by refitting simulated data (scripts 17 and 18).

**What the Fisher information cannot tell you:** whether the optimizer finds the right basin (E1), and anything about a wrong model (A1).

### 7.2 Purpose 2: error bars after the fit

1. **Optimization.** Use several initial points and compare deviances (E1). Use a tolerance of at most 0.02 nat, or compute the certificate of Section 4.6 (E2).
2. **Evaluation point.** Gauge-fix the estimate; evaluate $F$ and $\mathbf d_f$ at the estimate plus a small perturbation (P1). Keep the estimability check.
3. **Report three things per quantity:** the Fisher interval $f\pm1.96\,\sigma_f$, labelled as conservative under the asymptotic assumption; the rate resolution and $\kappa(v_f)$ that say whether "conservative" or "calibrated" applies; and, for a realistic width, the parametric-bootstrap standard deviation (README 6.4).
4. **Model violation.** From the same bootstrap refits, collect the deviance of each replicate. Compare the deviance of the real data with that distribution, not with $\chi^2_k$ (Section 5).
   - Inside the distribution: no detectable violation; the statistical error bars stand.
   - Above it: the error bars are a lower limit on the uncertainty. If the excess is uniform over circuits and depths, scale the widths by $\sqrt{\hat\omega}$ with $\hat\omega$ the ratio of the real deviance to the mean bootstrap deviance (overdispersion, Section 4.7; this estimate of $\omega$ is a heuristic and has not been tested). If it grows with depth or sits on particular circuits, the model is wrong in a structured way: raise $r_K$ or question A1.
5. **Bias near the boundary.** For quantities that are themselves near a boundary (small rates, infidelities) report a one-sided limit (README 6.5). Do not use the bias-corrected bootstrap interval there; the bootstrap is known to be inconsistent when a parameter is on the boundary [Source: Andrews 2000 (abstract checked)], and README 6.4 found exactly this.

### 7.3 What still has to be done

Ordered by how much each item decides. Runtimes are those of the existing scripts.

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | Run scripts 17 and 18 at one design that passes the validity numbers (for example noise strength 0.3, $L_{\max} = 23$, $N = 1000$, $m\approx1100$, PTM_PRECISION 5.2), 40 refits. Record also the mean of $\Lambda$ and of the deviance | Tests the central prediction: resolved rates $\Rightarrow$ $\sigma_f$ calibrated, $\Lambda\approx624$, $N_\sigma\approx0$. Without it "calibrated" is a conjecture | about an hour of fits |
| 2 | Add to `scripts/ptm_design.py`: rate resolution for $\rho$ and the POVM effects, the nonlinearity ratio $\kappa(v_f)$, the count of outcomes with $mp < 5$ and the merged-outcome width | The validity numbers of Section 4.5; needed for every design decision | small |
| 3 | Add the deviance of each replicate to the bootstrap (script 14 does not save the replicate data, so it cannot be recomputed) and build the calibrated goodness-of-fit test | Needed before any error bar on real data (Section 5) | small, plus reruns of the bootstrap |
| 4 | Simulate a truth outside the model and refit: (a) Kraus rank above $r_K$, with a realistic decoherence spectrum; (b) slow drift. Compare the error of the Pauli transfer matrix elements with $\sigma_f$, and the deviance with its bootstrap reference | A1 and A2 have never been violated in this study (learning 10). Tells how large the systematic error is and whether the test sees it | one new truth generator; fits as in item 1 |
| 5 | Decide on a realistic noise model for the device, as the prior guess for design and the truth for item 4 | The answer to "do the assumptions hold under realistic noise" depends on the size of the incoherent infidelity relative to $\delta$ (Section 6) | input from the hardware side |
| 6 | Compare $\sigma_f$ at fully converged (collapsed) estimates with and without perturbation, and with the oracle value | P1; the perturbation restores only one direction per collapsed Kraus operator (Section 4.4) | minutes, no fits |
| 7 | Four qubits: Section 8 | | |
| 8 | Verify the citations of Section 9 that are not marked checked | | reading |

---

## 8. Four qubits

**Nothing in Sections 2–4 depends on the number of qubits.** What changes is the size of every object and how far the experiment is from the asymptotic regime.

**Parameter counts** [Derived here, by the counting of README 4.1; the two-qubit case gives $576 + 15 + 48 - 15 = 624$]. With Kraus rank $r_K$, state rank $r_\rho$, effect rank $r_E$:

$$
n_{\rm id} = \underbrace{n_g\big(2d^2r_K - d^2 - r_K^2\big)}_{\text{gates}} + \underbrace{\big(2dr_\rho - r_\rho^2 - 1\big)}_{\text{state}} + \underbrace{\big(2n_{\rm out}dr_E - d^2 - n_{\rm out}r_E^2\big)}_{\text{POVM}} - \underbrace{(d^2 - 1)}_{\text{Hilbert-space gauge}} .
$$

Each bracket is the dimension of a Stiefel factor minus its mixing or purification gauge. For $d = n_{\rm out} = 16$ and, as an example, $n_g = 6$ gates:

| $r_K$ | $r_\rho$, $r_E$ | $n_{\rm id}$ | minimum $N = \lceil n_{\rm id}/15\rceil$ |
|---|---|---|---|
| 1 | 1, 1 | 1 545 | 103 |
| 4 | 4, 4 | 12 048 | 804 |
| 4 | 16, 16 | 14 496 | 967 |
| 16 | 4, 4 | 47 472 | 3 165 |
| 256 (all channels) | 4, 4 | 393 072 | 26 205 |

These counts assume that the generic rank of $J_T$ reaches $n_{\rm id}$ for the four-qubit design. That has to be checked numerically, as script 01 did for two qubits.

**Computation** (estimates from operation counts, not measured). For $r_K = r_\rho = r_E = 4$ the tangent space has dimension $D = 10\,752 + 1\,792 + 127 = 12\,671$.

- $W$ has $16N$ rows and $D$ columns: 3.2 GB in double precision at $N = 2000$. Building it takes $D$ forward-mode derivatives of a simulation of $N$ circuits with $16\times16$ matrices. Its singular value decomposition costs of order $16N\cdot D^2\approx5\times10^{12}$ operations times a small constant: minutes to tens of minutes on a workstation.
- For a handful of quantities no decomposition is needed. Solve $(F + \tau^{-2}I)\,x = \mathbf d_f$ by conjugate gradients, using only products of $J_T$ and $J_T^\top$ with vectors (`jax.jvp`, `jax.vjp`). The condition number is at most $1 + \lambda_1\tau^2$. Then $(\sigma_f^{\rm vT})^2 = \mathbf d_f^\top x$, and the LAN width is the limit of large $\tau$.
- The tangent basis should be built per Stiefel factor instead of from the singular value decomposition of the full constraint Jacobian $Dh$.
- There are $6\times255\times256 = 391\,680$ non-constant Pauli transfer matrix elements. Choose derived quantities (README 6.5, item 2) rather than all elements.

**Statistics.**

- **Rare outcomes get worse.** With 16 outcomes the average expected count is $m/16$, 62 at $m = 1000$, and for circuits with concentrated output distributions most outcomes are far below 5. At two qubits 29% of the outcomes were already below 5 (Section 5). Check R1 first.
- **The share of fold directions is about the same.** Per gate at $r_K = 4$, $255$ of the $1776$ identifiable directions are coherent (14%) and the other 86% are incoherent, against $15$ of $96$ (16%) and 84% at two qubits. Whether the incoherent ones are resolved depends on the incoherent infidelity of a four-qubit layer, which includes the idling errors of all four qubits.
- **A1 is the main risk.** $r_K = 4$ out of a possible 256 is a strong restriction, and a four-qubit layer has at least as many independent decoherence channels as it has qubits. Item 4 of Section 7.3 should be done with a four-qubit-like spectrum before trusting any error bar.
- **Static crosstalk is inside the model** when every gate is a full four-qubit channel. That is an advantage over models built from one- and two-qubit gates.

**Order of work.**

1. Items 1–4 of Section 7.3 at two qubits.
2. Make the Fisher computation matrix-free and time it on one four-qubit design; verify the rank and the threshold.
3. Expected-count check and validity numbers at a four-qubit prior guess.
4. One refit test at four qubits, with a truth of the model's own rank and one of higher rank.

### 8.1 Implementation and first four-qubit numbers

[Numerical: 20. Code in [`scripts/fisher_matrix_free.py`](scripts/fisher_matrix_free.py), checks in [`scripts/20_matrix_free_check.py`](scripts/20_matrix_free_check.py), outputs in `data/20_matrix_free_*_output.txt`. No fits were run.]

**The actual four-qubit gate set has 9 gates** (`create_4q_gst_config`: four $R_x(\pi/2)$, four $R_y(\pi/2)$, one CZ–CZ), not the 6 of the table above. With all ranks equal to 4: tangent dimension $18\,047$, $n_{\rm id} = 17\,376$, minimum $N = 1159$. The saved run has $N = 10\,000$, so the whitened Jacobian would have $160\,000\times18\,047$ entries (23 GB). The implementation never builds it.

**How it works** [Derived here].

- *Fisher product.* $F_1v = P\,J^\top\operatorname{diag}(1/p)\,J\,Pv$ on ambient vectors, with $J$ applied by `jax.jvp`, $J^\top$ by `jax.vjp`, and the tangent projector $P$ applied per Stiefel factor (`mGST.riemannian.project_onto_tangent_space`). In a tangent basis $T$ this is $J_T^\top\operatorname{diag}(1/p)J_T$, because $P = TT^\top$.
- *Widths from one Lanczos run per quantity.* Lanczos on $F_1$ started at $\mathbf d_f$ (`mGST.stochastic_lanczos_quadrature.lanczos_from_vector`) gives nodes $\theta_i$ and weights $w_i$ with $\mathbf d_f^\top g(F_1)\,\mathbf d_f\approx\lVert\mathbf d_f\rVert^2\sum_iw_ig(\theta_i)$ for any function $g$. With $g(x) = 1/(mx + \tau^{-2})$ this is the squared van Trees width for every $m$ and $\tau$ at once; with $g(x) = 1/(mx)$ the squared LAN width.
- *Error control.* For $g(x) = 1/(mx + \tau^{-2})$ the Gauss rule is a lower bound and the Gauss–Radau rule with a node at 0 an upper bound [Source: Golub & Meurant 2010, Ch. 6; from memory]. The van Trees width is returned as an interval. For the LAN width only the lower bound exists.
- *Gauge fixing without Pauli transfer matrices.* $\sum_g\lVert R_g - R_g^{\rm target}\rVert_F^2$ equals the same distance between superoperators, and $\langle S, S'\rangle = \sum_{r,s}\lvert\operatorname{tr}(K_r^\dagger K_s')\rvert^2$ for $S = \sum_rK_r\otimes\bar K_r$. The gradient of a gauge-fixed quantity (PTM_PRECISION eq. 2) applies $H_{h\theta}$ as one reverse-mode gradient.
- *Nonlinearity ratio.* $c''(0)$ along the polar retraction is $v^\top(\nabla^2p)\,v + J\,a$ with $a = -XV^\dagger V$ per factor, the second-order term of $(X + tV)(I + t^2V^\dagger V)^{-1/2}$. Both terms are exact (nested `jax.jvp`). The direction is $v = (mF_1 + \tau^{-2}P)^{-1}\mathbf d_f$ from conjugate gradients, i.e. $F^+\mathbf d_f$ of Section 4.5 regularized by the prior width.

**Validation at two qubits against the dense `ptm_design.py`** (reference design; `validate` mode, about 4 minutes).

| Check | Result |
|---|---|
| projector, Fisher product, grouped contraction, gauge cost, gauge-fixed gradients, rate gradients | agree to $10^{-14}$ or better |
| van Trees width, $\tau = 0.01$ | exact to 4 digits after 30 Lanczos steps |
| van Trees width, $\tau = 0.1$ | interval contains the dense value; its relative width is 19–50% at 30 steps and about 0.1% at 100 steps |
| LAN width (lower bound) | 40–49% of the dense value at 30 steps, 72–79% at 100, 99.3–100% at 300 (three elements); 98–100% at 300 for the six rates |
| merged-outcome width (29% of outcomes rare) | $4.2530\times10^{-3}$ dense and matrix-free, against $4.1424\times10^{-3}$ unmerged |
| nonlinearity ratio, second derivatives versus finite step | 1.264 / 1.262, 0.055 / 0.055, 0.494 / 0.494 |
| deviance of estimate 0 | 939.5, inside the range of script 19 |
| optimizer certificate (Section 4.6) for a test-circuit probability | default tolerance: shift $1.1\times10^{-4}$, i.e. 0.03 widths; converged: $5\times10^{-7}$ |

**Four qubits, simulated truth, $N = 2000$ ($1.7\times$ the minimum), $m = 500$, $L_{\max} = 19$, 50 Lanczos steps** (about 5 minutes per noise level on a laptop; one Fisher product takes 1.2 s).

| | noise strength 0.05 | noise strength 0.3 |
|---|---|---|
| incoherent weight $1-\mu_1$ per gate | $5\times10^{-4}$ | $1.8\times10^{-2}$ |
| outcomes with expected count below 5 / below 1 | 53% / 53% | 14% / 0% |
| smallest Choi eigenvalue of CZ–CZ: value | $1.4\times10^{-4}$ | $5.0\times10^{-3}$ |
| its van Trees width, $\tau = 0.01$ / $0.03$ | $5.3\times10^{-5}$ / $1.2\times10^{-4}$ | $3.1\times10^{-4}$ / $7.1\times10^{-4}$ |
| its LAN width (lower bound) | $\ge2.6\times10^{-4}$ | $\ge1.4\times10^{-3}$ |
| rate resolution $u$ (upper limit) | $\le0.54$ | $\le3.5$ |
| entanglement infidelity of CZ–CZ: value | $6.2\times10^{-4}$ | $2.2\times10^{-2}$ |
| its van Trees width, $\tau = 0.01$ / $0.03$ | $8.6\times10^{-5}$ / $1.6\times10^{-4}$ | $5.2\times10^{-4}$ / $9.6\times10^{-4}$ |
| root-mean-square van Trees width of a CZ–CZ element, $\tau = 0.01$ (one random combination) | $1.2\times10^{-3}$ | $1.6\times10^{-3}$ |

All van Trees intervals had a relative width of 1% or less.

**The saved four-qubit fit** ($N = 10\,000$, $m = 500$; `saved_run` mode, 8 s; only probabilities are evaluated).

- Estimated incoherent weight per gate: $1.2$–$2.7\times10^{-2}$; smallest Choi eigenvalues $0.9$–$2.6\times10^{-3}$. So noise strength 0.3 is the relevant simulation regime, not 0.05.
- 46% of the outcomes have an expected count below 5 and 21% were never observed.
- Deviance $182\,018$ against the Wilks value $k = 15N - n_{\rm id} = 132\,624$, i.e. $N_\sigma = 96$. By Section 5 this number alone does not show a model violation: the reference is uncalibrated, and I did not check whether that fit maximized the likelihood or minimized least squares.

**Reading.**

1. **The matrix-free route reproduces the dense one**, and the van Trees width is cheap: a few tens of Fisher products per quantity.
2. **The LAN width is not available at this cost.** After 50 steps the lower bound is far from converged (at two qubits it reached 99% only after 300 steps). The rate resolution is therefore only bounded from above. At noise strength 0.05 that is enough to conclude that the rates are unresolved ($u\le0.54$). At noise strength 0.3 the bound $u\le3.5$ decides nothing.
3. **Rare outcomes are worse than at two qubits**, as predicted: about half of all outcomes at weak noise.
4. **Not accessible this way:** the largest width over all $391\,680$ elements (only the root mean square, from random combinations), the rank of $F$, and $n_{\rm eff}$.

**Use.**

```python
import fisher_matrix_free as fmf                      # in scripts/
fixed, info = fmf.gauge_fix_to_target(kraus, povm, state, target_kraus)
problem = fmf.fisher_problem(*fixed, indices_list, batch_size=2500, contraction="grouped")
gauge = fmf.gauge_context(problem, target_kraus)                      # only for gauge-dependent quantities
d = fmf.quantity_gradient(problem, fmf.entanglement_infidelity(8, target_kraus), gauge)
quadrature = fmf.lanczos_quadrature(problem, d, num_steps=50)
lower, upper = fmf.van_trees_sigma(quadrature, m=500, tau=0.01)       # any m and tau, no new products
sigma_lan_bound, kernel_share = fmf.lan_sigma(quadrature, m=500)
v, residual = fmf.solve_direction(problem, d, m=500, tau=0.01)
kappa = fmf.nonlinearity_ratio(problem, v, m=500)["kappa"]
```

**Cost at the real design.** One Fisher product at $N = 10\,000$ takes about 6 s with the grouped contraction (extrapolated from 1.2 s at $N = 2000$), so 50 Lanczos steps take about 5 minutes per quantity, and 300 steps for a converged LAN bound about half an hour. These runs were not done.

---

## 9. Sources

Sources already listed in README Section 8 are not repeated. Status of the new ones, as of 2026-11-16:

**Asymptotic statistics**
- A. W. van der Vaart, *Asymptotic Statistics*, Cambridge University Press (1998).
  - Thm. 5.21 (asymptotic normality of estimators defined by a vanishing estimating equation, covariance $V^{-1}(P\psi\psi^\top)V^{-\top}$). **(checked through a secondary source that quotes the theorem with its number: M. I. Jordan, Stat 210B lecture notes, 7 March 2007, Theorem 3)**. Used in Section 4.7.
  - Thm. 5.39 (maximum-likelihood estimate under differentiability in quadratic mean and a Lipschitz condition). Number and content confirmed through papers that cite and paraphrase it (for example arXiv:1612.05178). Not checked against the book. Used in Section 4.3.
  - Def. 7.14 (local asymptotic normality) and Thm. 7.2 (differentiability in quadratic mean implies it). Numbers confirmed through papers that cite them. Wording from memory. Used in Section 4.2.
  - Thm. 8.11 (local asymptotic minimax theorem). Number confirmed through a paper that cites it (arXiv:2606.20642). Used in Section 4.8.
  - Thm. 3.1 (delta method); Ch. 2 (central limit theorem); Ch. 16 (likelihood-ratio statistics). From memory.
- J. Duchi, Stats 300b lecture slides, Stanford, Winter 2021, sets 14 ("Local asymptotics and optimality") and 15 ("Achieving local asymptotic bounds and extensions"). **(read)** Used to check the definition of differentiability in quadratic mean (slide 14–6) and the form of the local asymptotic minimax bound for a function of the parameter (slides 15–7, 15–8).
- H. White, "Maximum likelihood estimation of misspecified models", Econometrica 50(1), 1–25 (1982; some catalogues give pages 1–26). Journal, volume and year checked on the publisher's page. Thm. 2.2 (consistency for the pseudo-truth) and Thm. 3.2 (asymptotic normality) confirmed by number through papers that cite them. Statements not checked against the paper. Used in Section 4.7.
- S. S. Wilks, "The large-sample distribution of the likelihood ratio for testing composite hypotheses", Ann. Math. Statist. 9, 60–62 (1938). From memory. Used in Sections 4.5, 4.7.
- P. McCullagh, J. A. Nelder, *Generalized Linear Models*, 2nd ed., Chapman & Hall (1989), chapters on over-dispersion and quasi-likelihood. From memory. Used in Section 4.7.

**Boundaries and constraints**
- H. Chernoff, "On the distribution of the likelihood ratio", Ann. Math. Statist. 25, 573–578 (1954). From memory.
- S. G. Self, K.-Y. Liang, "Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions", J. Amer. Statist. Assoc. 82(398), 605–610 (1987). **(bibliographic details and abstract checked)**: limiting distributions of likelihood-ratio statistics are mixtures of chi-squared distributions when the truth is on the boundary. Theorem numbers not checked.
- C. J. Geyer, "On the asymptotics of constrained M-estimation", Ann. Statist. 22(4), 1993–2010 (1994). **(bibliographic details and abstract checked)**.
- D. W. K. Andrews, "Inconsistency of the bootstrap when a parameter is on the boundary of the parameter space", Econometrica 68(2), 399–405 (2000). **(bibliographic details and abstract-level content checked)**. Used in Section 7.2.
- D. M. Bates, D. G. Watts, "Relative curvature measures of nonlinearity", J. R. Statist. Soc. B 42(1), 1–25 (1980). Attribution confirmed through secondary sources; volume and pages from memory; definitions not checked. Section 4.5 does not rely on them.

**Gate set tomography**
- E. Nielsen et al., "Gate Set Tomography", Quantum 5, 557 (2021), arXiv:2009.07301, Sec. 6.1 "Goodness-of-fit and Markovianity": Wilks' theorem with $k = N_o - N_p^{\rm nongauge}$ and the measure $N_\sigma$. **(checked against the arXiv text as returned by a web search; the section was not read in full)**. Used in Sections 4.7, 5.

**Numerical linear algebra**
- G. H. Golub, G. Meurant, *Matrices, Moments and Quadrature with Applications*, Princeton University Press (2010), Ch. 6: Gauss and Gauss–Radau quadrature rules from the Lanczos recurrence as lower and upper bounds on $u^\top g(A)u$. From memory; the bracketing was confirmed numerically against the dense two-qubit values (script 20). Used in Section 8.1.

**New files**
- [`scripts/fisher_matrix_free.py`](scripts/fisher_matrix_free.py) and [`scripts/20_matrix_free_check.py`](scripts/20_matrix_free_check.py): matrix-free Fisher analysis for four qubits and its checks (Section 8.1). Modes `validate` (about 4 minutes), `four_qubits` (about 5 minutes at $N = 2000$ and 50 steps), `saved_run` (seconds).
- [`scripts/19_goodness_of_fit_check.py`](scripts/19_goodness_of_fit_check.py): deviance and likelihood-ratio statistic of the saved estimates of script 07, and expected counts at the truth. About 30 s, no fitting. Output: [`data/19_goodness_of_fit_check_output.txt`](data/19_goodness_of_fit_check_output.txt).
