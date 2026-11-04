"""Figure: Cramer-Rao standard deviation 1/sqrt(lambda_k) of every Fisher eigen-direction, sorted (m = 1000 shots).

Reads data/crb_spectra.npz (written by 08_crb_analysis.py and completed by 09_crb_efficiency.py), writes figures/crb_spectra.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
d = np.load(HERE / "data" / "crb_spectra.npz")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

fig = plt.figure(figsize=(13, 9.4), constrained_layout=True)
grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
axes = [fig.add_subplot(grid[0, 0])]
axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
axes.append(fig.add_subplot(grid[1, :]))

def decorate(ax, title):
    ax.set_yscale("log")
    ax.set_xlim(0, 815)
    ax.set_ylim(3e-4, 1e4)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvspan(624, 815, color=GRID, alpha=0.5, linewidth=0)
    ax.text(720, 1.5e-3, "gauge directions:\nnever constrained", ha="center", va="center", color=MUTED, fontsize=9)
    for y, label, style in [(1e-2, "ε = 0.01", "--"), (1.0, "≈ size of the manifold", ":")]:
        ax.axhline(y, color=MUTED, linestyle=style, linewidth=1)
        ax.text(8, y * 1.25, label, color=INK2, fontsize=9, va="bottom")
    ax.set_xlabel("eigen-direction k (sorted, best determined first)")
    ax.set_title(title, loc="left", fontsize=11, color=INK)

def curve(ax, lam, color, label, linestyle="-", label_at=None, label_xy=None, ha="left"):
    std = 1 / np.sqrt(lam)
    k = np.arange(1, len(std) + 1)
    ax.plot(k, std, color=color, linewidth=2, linestyle=linestyle, label=label.split(" (rank")[0].replace(", your init", " (your init)"), solid_capstyle="round")
    ax.plot(k[-1], std[-1], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    if label_xy is not None:
        ax.text(*label_xy, label, color=INK2, fontsize=9, va="center", ha=ha)
        return
    i = len(std) - 1 if label_at is None else label_at
    ax.annotate(label, (k[i], std[i]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")

ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
decorate(axes[0], "(a) True gate set, different numbers of sequences N")
for color, n in zip(ramp, [50, 150, 250, 400, 800]):
    curve(axes[0], d[f"N{n}"], color, f"N = {n}")
axes[0].set_ylabel("Cramér–Rao standard deviation  1/√λ_k  (m = 1000 shots)")
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=INK2, fontsize=9)

decorate(axes[1], "(b) N = 400, different evaluation points")
curve(axes[1], d["truth"], "#2a78d6", "true gate set (rank 624)", label_xy=(605, 300), ha="right")
curve(axes[1], d["estimate"], "#eb6834", "maximum-likelihood estimate (rank 613)", label_xy=(445, 3), ha="right")
curve(axes[1], d["init"], "#1baf7a", "perturbed target, your init (rank 624)", label_xy=(605, 15), ha="right")
curve(axes[1], d["unitary"], MUTED, "exactly unitary gates (rank 137)", linestyle="--", label_xy=(40, 6e-2))
axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=INK2, fontsize=9)

ax = axes[2]
var_e = d["var_e"]
k = np.arange(1, len(var_e) + 1)
window = 25
running = np.array([np.median(var_e[max(0, i - window):i + window + 1]) for i in range(len(var_e))])
ax.scatter(k, var_e, s=6, color=MUTED, alpha=0.5, linewidths=0, label="single direction")
ax.plot(k, running, color="#2a78d6", linewidth=2, label="running median (±25)")
ax.axhline(1, color=INK2, linestyle="--", linewidth=1)
ax.text(612, 1.06, "= 1: attains the Cramér–Rao bound (efficient and unbiased)", ha="right", va="bottom", color=INK2, fontsize=9)
ax.text(612, 0.17, "= 0: estimate does not move along this direction (biased)", ha="right", va="bottom", color=INK2, fontsize=9)
ax.axvline(135, color=MUTED, linestyle=":", linewidth=1)
ax.text(140, 2.0, "k = 135 = n_eff(ε = 0.01) at N = 400", color=INK2, fontsize=9, va="center")
ax.set_xlim(0, 630)
ax.set_ylim(-0.05, 2.4)
ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("eigen-direction k of the Fisher information at the truth (same order as the curves above)")
ax.set_ylabel("mean of e_k² over 40 estimates")
ax.set_title("(c) N = 400, m = 1000: does the maximum-likelihood estimate attain the Cramér–Rao bound along direction k?", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper right", frameon=False, labelcolor=INK2, fontsize=9)

fig.suptitle("Reading: n_eff(ε) = index k where a curve crosses the ε line;  rank = where the curve ends (λ_k = 0 beyond)",
             x=0.01, ha="left", fontsize=10, color=INK2)
out = HERE / "figures" / "crb_spectra.png"
fig.savefig(out, dpi=160)
print(out)
