#!/usr/bin/env python3
"""Illustrative calculations for the revised Perspective
"Geometry and the Emergence of Molecular Order in Early Life" (Discover Life, revision 2).

Online Resource 1. Produces Figs. 1-4 and results_r2.json. Runs in about a minute on a laptop.

  Fig. 1  Independent-site equilibrium baseline: ee_ads = < tanh(dE_i / 2kT) > over a distribution of signed
          local discrimination energies (equal occupancy), its dependence on disorder for three distribution
          shapes matched in mean and variance, and a parameter-uncertainty band.
  Fig. 2  Two surfaces with IDENTICAL site statistics and different spatial arrangement (clustered vs dispersed),
          before and after finite-range sampling with a normalized kernel of length l.
  Fig. 3  Contrast survival: fraction of sites whose sampled preference stays above a threshold e*, versus l, for
          several domain correlation lengths; the same with an identical continuous energy multiset; and a signed
          (racemic) surface where local survival coexists with zero global mean.
  Fig. 4  When the arrangement changes the recovered output: uniform recovery (null), recovery co-located with
          selective sites, and two hypothetical nonlinear local steps of opposite curvature.

Every quantity is a phenomenological illustration with stated assumptions; nothing here is a first-principles
calculation for a specific mineral.

    python chiral_geometry_r2.py [outdir]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, ndimage, stats

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

KB = 8.617333e-5          # eV/K
T = 298.0
KT = KB * T               # 25.68 meV
C2 = 2 * KT               # tanh argument scale

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
PAL = ["#1b4f8a", "#d1495b", "#2e8b57", "#e09f3e", "#6c4f9e", "#555555"]
R = {}


def tag(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# ----------------------------------------------------------------------------------------------------------------
# Fig. 1  independent-site baseline
# ----------------------------------------------------------------------------------------------------------------
def ee_gauss(m, s):
    """< tanh(x/2kT) >, x ~ N(m, s^2); signed sites allowed (x < 0 favours the opposite enantiomer)."""
    if s <= 1e-12:
        return float(np.tanh(m / C2))
    z, w = np.polynomial.hermite_e.hermegauss(80)
    return float(np.sum(w * np.tanh((m + s * z) / C2)) / np.sqrt(2 * np.pi))


def ee_laplace(m, s):
    if s <= 1e-12:
        return float(np.tanh(m / C2))
    b = s / np.sqrt(2)
    f = lambda x: np.tanh(x / C2) * np.exp(-abs(x - m) / b) / (2 * b)
    lo, hi = m - 40 * b, m + 40 * b
    return float(integrate.quad(f, lo, m, limit=200)[0] + integrate.quad(f, m, hi, limit=200)[0])


def ee_twopop(m, s):
    """Terrace + defect band with the same mean and variance: fraction w at dE_t, 1 - w at 0."""
    if s <= 1e-12:
        return float(np.tanh(m / C2))
    w = m * m / (m * m + s * s)
    return float(w * np.tanh((m / w) / C2))


SHAPES = {"Gaussian": ee_gauss, "Two-population (terrace + defect band)": ee_twopop, "Laplace (heavy-tailed)": ee_laplace}


def sigma(m, U, gamma=1.0):
    return (max(1.0 - U, 0.0) ** gamma) * m


fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.35))
dE = np.linspace(0, 0.040, 161)
for k, U in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]):
    axs[0].plot(dE * 1e3, [ee_gauss(m, sigma(m, U)) for m in dE], color=PAL[k], lw=1.8, label=f"$U_\\chi$ = {U:.1f}")
axs[0].set_xlabel(r"Mean discrimination energy $\overline{\Delta E}$ (meV)")
axs[0].set_ylabel(r"$ee_{\mathrm{ads}}$ (equal occupancy)")
axs[0].set_title(r"Baseline vs $\overline{\Delta E}$ (Gaussian, $\sigma_\chi=(1-U_\chi)\overline{\Delta E}$)")
axs[0].legend(frameon=False, loc="upper left")
axs[0].set_ylim(0, 0.75)
tag(axs[0], "a")

m0 = 0.020
Us = np.linspace(0, 1, 101)
R["fig1"] = {"dE_ref_meV": 20, "kT_meV": KT * 1e3}
for k, (name, fn) in enumerate(SHAPES.items()):
    y = [fn(m0, sigma(m0, u)) for u in Us]
    axs[1].plot(Us, y, color=PAL[k], lw=1.8, label=name)
    R["fig1"][name] = {f"U={u:.1f}": round(fn(m0, sigma(m0, u)), 4) for u in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)}
# second-order (small-disorder) expansion: <f> ~ f(m) - (s/c)^2 sech^2(m/c) tanh(m/c)
a = m0 / C2
approx = [np.tanh(a) - (sigma(m0, u) / C2) ** 2 / np.cosh(a) ** 2 * np.tanh(a) for u in Us]
axs[1].plot(Us, approx, color="k", lw=1.0, ls=":", label="Second-order expansion")
axs[1].set_xlabel(r"Morphological descriptor $U_\chi$ (via assumed $\sigma_\chi(U_\chi)$)")
axs[1].set_ylabel(r"$ee_{\mathrm{ads}}$ at $\overline{\Delta E}$ = 20 meV")
axs[1].set_title("Matched mean and variance, three shapes")
axs[1].legend(frameon=False, loc="lower right")
axs[1].set_ylim(0, 0.45)
tag(axs[1], "b")

rng = np.random.default_rng(20261116)
ND = 1000
draws = []
for _ in range(ND):
    m = rng.uniform(0.010, 0.030)
    g = rng.uniform(0.5, 2.0)
    fn = list(SHAPES.values())[rng.integers(3)]
    draws.append([fn(m, sigma(m, u, g)) for u in Us[::5]])
draws = np.array(draws)
u5 = Us[::5]
med, p05, p95 = (np.percentile(draws, q, axis=0) for q in (50, 5, 95))
axs[2].fill_between(u5, p05, p95, color=PAL[0], alpha=0.18, label="5th-95th percentile")
axs[2].plot(u5, med, color=PAL[0], lw=1.8, label="Median")
axs[2].set_xlabel(r"$U_\chi$")
axs[2].set_ylabel(r"$ee_{\mathrm{ads}}$")
axs[2].set_title(r"Uncertainty: $\overline{\Delta E}$ 10-30 meV, $\gamma$ 0.5-2, 3 shapes")
axs[2].legend(frameon=False, loc="upper left")
axs[2].set_ylim(0, 0.6)
tag(axs[2], "c")
R["fig1"]["mc"] = {"draws": ND, "U": [round(x, 2) for x in u5], "median": [round(x, 4) for x in med],
                   "p05": [round(x, 4) for x in p05], "p95": [round(x, 4) for x in p95]}
R["fig1"]["ratio_U0.2_over_U1_gauss_20meV"] = round(ee_gauss(m0, sigma(m0, 0.2)) / ee_gauss(m0, 0), 4)
R["fig1"]["ratio_U0.2_over_U1_gauss_30meV"] = round(ee_gauss(0.03, sigma(0.03, 0.2)) / ee_gauss(0.03, 0), 4)
R["fig1"]["ee_eq_10_20_30meV"] = [round(float(np.tanh(x / C2)), 4) for x in (0.01, 0.02, 0.03)]
# weak-site fraction (a descriptor, not racemization): P(|dE_i| < 0.3 kT) for the Gaussian case at 20 meV
R["fig1"]["f_weak_0.3kT_gauss_20meV"] = {f"U={u:.1f}": round(float(stats.norm.cdf(0.3 * KT, m0, max(sigma(m0, u), 1e-12))
                                                                 - stats.norm.cdf(-0.3 * KT, m0, max(sigma(m0, u), 1e-12))), 4)
                                          for u in (1.0, 0.8, 0.6, 0.4, 0.2)}
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig1_independent_site_baseline.png"))
fig.savefig(os.path.join(OUT, "Fig1_independent_site_baseline.tif"), pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("Fig1 done")

# ----------------------------------------------------------------------------------------------------------------
# Lattice machinery for Figs. 2-4
# ----------------------------------------------------------------------------------------------------------------
L = 128
PHI = 0.25
E_A = float(np.tanh(0.030 / C2))          # selective site, 30 meV  -> 0.526
E_N = 0.0                                  # nonselective site
E_STAR = 0.20
P_STAR = (E_STAR - E_N) / (E_A - E_N)
NREP = 16
ELLS = np.array([0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
XIS = [0, 1, 2, 4, 8]


def field(xi, rng):
    w = rng.standard_normal((L, L))
    return w if xi == 0 else ndimage.gaussian_filter(w, xi, mode="wrap")


def binary_map(xi, rng, phi=PHI):
    """Exactly round(phi*N) selective sites; arrangement set by a (thresholded) correlated field."""
    f = field(xi, rng).ravel()
    n = int(round(phi * f.size))
    c = np.zeros(f.size)
    c[np.argsort(f)[-n:]] = 1.0
    return c.reshape(L, L)


def signed_map(xi, rng, phi=PHI):
    f = field(xi, rng).ravel()
    n = int(round(phi / 2 * f.size))
    o = np.argsort(f)
    c = np.zeros(f.size)
    c[o[-n:]] = 1.0
    c[o[:n]] = -1.0
    return c.reshape(L, L)


def mix(e, ell):
    """Normalized, translation-invariant (hence doubly stochastic) Gaussian sampling kernel on a periodic lattice."""
    return e.copy() if ell == 0 else ndimage.gaussian_filter(e, ell, mode="wrap")


def clusters(c):
    lab, n = ndimage.label(c > 0)
    sizes = np.bincount(lab.ravel())[1:]
    rows = set(np.unique(lab[0][lab[0] > 0])) & set(np.unique(lab[-1][lab[-1] > 0]))
    cols = set(np.unique(lab[:, 0][lab[:, 0] > 0])) & set(np.unique(lab[:, -1][lab[:, -1] > 0]))
    return n, float(sizes.mean()), bool(rows or cols)


# ----------------------------------------------------------------------------------------------------------------
# Fig. 2  identical statistics, different arrangement
# ----------------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(7)
cc = binary_map(4, rng)
cd = binary_map(0, rng)
ec, ed = E_N + cc * (E_A - E_N), E_N + cd * (E_A - E_N)
ELL2 = 2.0
mc, md = mix(ec, ELL2), mix(ed, ELL2)
fig, axs = plt.subplots(1, 4, figsize=(10.6, 2.95))
crop = (slice(0, 64), slice(0, 64))
for ax, img, ttl in [(axs[0], cc, f"Clustered: {int(cc.sum())} selective sites"),
                     (axs[1], cd, f"Dispersed: {int(cd.sum())} selective sites")]:
    ax.imshow(img[crop], cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest")
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
for ax, img, raw, ttl in [(axs[2], mc, cc, "Clustered, sampled over $\\ell$ = 2"), (axs[3], md, cd, "Dispersed, sampled over $\\ell$ = 2")]:
    im = ax.imshow(img[crop], cmap="viridis", vmin=0, vmax=E_A, interpolation="nearest")
    ax.contour(img[crop], levels=[E_STAR], colors="w", linewidths=0.8)
    fs = (img >= E_STAR).mean()
    ax.set_title(ttl)
    ax.text(0.02, 0.02, f"$F_{{surv}}$ = {fs:.3f}\nmean = {img.mean():.4f}", transform=ax.transAxes, color="w", fontsize=7.5,
            va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="k", alpha=0.45, lw=0))
    ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([0.945, 0.12, 0.012, 0.66])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$e_i^{\mathrm{mix}}$", fontsize=8)
for k, ax in enumerate(axs):
    ax.set_title("abcd"[k] + "   " + ax.get_title(), loc="left", fontsize=8.5)
    ax.set_title("")
fig.subplots_adjust(left=0.01, right=0.93, wspace=0.06, top=0.86, bottom=0.03)
fig.savefig(os.path.join(OUT, "Fig2_identical_statistics_arrangement.png"))
fig.savefig(os.path.join(OUT, "Fig2_identical_statistics_arrangement.tif"), pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
R["fig2"] = {"L": L, "crop": 64, "n_selective": [int(cc.sum()), int(cd.sum())], "ell": ELL2,
             "F_surv": [round(float((mc >= E_STAR).mean()), 4), round(float((md >= E_STAR).mean()), 4)],
             "mean_before": [round(float(ec.mean()), 6), round(float(ed.mean()), 6)],
             "mean_after": [round(float(mc.mean()), 6), round(float(md.mean()), 6)]}
print("Fig2 done")

# ----------------------------------------------------------------------------------------------------------------
# Fig. 3  contrast survival
# ----------------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(11)
surv = {xi: np.zeros((NREP, len(ELLS))) for xi in XIS}
meandev = 0.0
conn = {}
for xi in XIS:
    nc, ms, perc = [], [], []
    for r in range(NREP):
        c = binary_map(xi, rng)
        n, s, p = clusters(c)
        nc.append(n); ms.append(s); perc.append(p)
        e = E_N + c * (E_A - E_N)
        for j, ell in enumerate(ELLS):
            em = mix(e, ell)
            surv[xi][r, j] = (em >= E_STAR).mean()
            meandev = max(meandev, abs(em.mean() - e.mean()))
    conn[xi] = {"clusters_per_1000_sites": round(1000 * np.mean(nc) / L ** 2, 2), "mean_cluster_size_sites": round(float(np.mean(ms)), 1),
                "spanning_fraction": round(float(np.mean(perc)), 2)}

# identical continuous energy multiset (Gaussian, mean 8 meV, U = 0.2), random vs rank-correlated placement
MS_M, MS_U = 0.008, 0.2
surv_ms = {"random": np.zeros((NREP, len(ELLS))), "correlated (xi=4)": np.zeros((NREP, len(ELLS)))}
for r in range(NREP):
    dEi = np.sort(rng.normal(MS_M, sigma(MS_M, MS_U), L * L))
    ei = np.tanh(dEi / C2)
    for key, xi in (("random", 0), ("correlated (xi=4)", 4)):
        f = field(xi, rng).ravel()
        e = np.empty(L * L)
        e[np.argsort(f)] = ei                      # same multiset, arrangement follows the field ranks
        e = e.reshape(L, L)
        for j, ell in enumerate(ELLS):
            em = mix(e, ell)
            surv_ms[key][r, j] = (em >= E_STAR).mean()
            meandev = max(meandev, abs(em.mean() - e.mean()))

# signed racemic surface
surv_sg = {xi: np.zeros((NREP, len(ELLS))) for xi in (0, 4)}
mean_sg = {xi: np.zeros((NREP, len(ELLS))) for xi in (0, 4)}
for xi in (0, 4):
    for r in range(NREP):
        e = signed_map(xi, rng) * E_A
        for j, ell in enumerate(ELLS):
            em = mix(e, ell)
            surv_sg[xi][r, j] = (np.abs(em) >= E_STAR).mean()
            mean_sg[xi][r, j] = em.mean()

fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.35))
for k, xi in enumerate(XIS):
    lab = "dispersed (random)" if xi == 0 else f"clustered, $\\xi$ = {xi}"
    axs[0].errorbar(ELLS, surv[xi].mean(0), yerr=surv[xi].std(0), color=PAL[k], lw=1.6, marker="o", ms=2.8, capsize=1.5, label=lab)
axs[0].set_xscale("symlog", linthresh=1)
axs[0].set_xlabel(r"Sampling (transport) length $\ell$ (sites)")
axs[0].set_ylabel(r"$F_{\mathrm{surv}}$ ($e_i^{\mathrm{mix}} \geq e_*$)")
axs[0].set_title(f"Binary surface, $\\phi$ = {PHI}, $e_*$ = {E_STAR}, $p_*$ = {P_STAR:.2f}")
axs[0].legend(frameon=False, loc="lower left")
axs[0].set_ylim(0, 0.30)
tag(axs[0], "a")
for k, key in enumerate(surv_ms):
    axs[1].errorbar(ELLS, surv_ms[key].mean(0), yerr=surv_ms[key].std(0), color=PAL[[0, 3][k]], lw=1.6, marker="o", ms=2.8, capsize=1.5, label=key)
axs[1].set_xscale("symlog", linthresh=1)
axs[1].set_xlabel(r"$\ell$ (sites)")
axs[1].set_ylabel(r"$F_{\mathrm{surv}}$")
axs[1].set_title("Identical energy multiset (8 meV, $U_\\chi$ = 0.2)")
axs[1].legend(frameon=False, loc="lower left")
axs[1].set_ylim(0, 0.5)
tag(axs[1], "b")
for k, xi in enumerate((0, 4)):
    lab = "dispersed" if xi == 0 else "clustered, $\\xi$ = 4"
    axs[2].plot(ELLS, surv_sg[xi].mean(0), color=PAL[[0, 3][k]], lw=1.6, marker="o", ms=2.8, label=f"$F_{{surv}}$ (|e| $\\geq e_*$), {lab}")
    axs[2].plot(ELLS, mean_sg[xi].mean(0), color=PAL[[0, 3][k]], lw=1.2, ls="--", label=f"global mean, {lab}")
axs[2].axhline(0, color="k", lw=0.6)
axs[2].set_xscale("symlog", linthresh=1)
axs[2].set_xlabel(r"$\ell$ (sites)")
axs[2].set_ylabel("Fraction / signed mean")
axs[2].set_title("Signed (racemic) surface: equal $\\pm$ domains")
axs[2].legend(frameon=False, loc="center right", fontsize=6.8)
axs[2].set_ylim(-0.05, 0.3)
axs[2].set_xlim(-0.3, 30)
tag(axs[2], "c")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig3_contrast_survival.png"))
fig.savefig(os.path.join(OUT, "Fig3_contrast_survival.tif"), pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)


def ell_half(curve):
    """Smallest l at which the survival fraction falls below half its l = 0 value (linear interpolation)."""
    y0 = curve[0] / 2
    for j in range(1, len(ELLS)):
        if curve[j] < y0:
            x0, x1, a0, a1 = ELLS[j - 1], ELLS[j], curve[j - 1], curve[j]
            return round(float(x0 + (a0 - y0) * (x1 - x0) / (a0 - a1)), 2)
    return None


R["fig3"] = {"phi": PHI, "e_a": round(E_A, 4), "e_n": E_N, "e_star": E_STAR, "p_star": round(P_STAR, 4), "nrep": NREP, "L": L,
             "ells": ELLS.tolist(), "max_abs_mean_change": float(meandev),
             "binary": {str(xi): {"mean": surv[xi].mean(0).round(4).tolist(), "sd": surv[xi].std(0).round(4).tolist(),
                                  "ell_half": ell_half(surv[xi].mean(0)), **conn[xi]} for xi in XIS},
             "multiset": {k: {"mean": v.mean(0).round(4).tolist(), "ell_half": ell_half(v.mean(0))} for k, v in surv_ms.items()},
             "signed": {str(xi): {"F": surv_sg[xi].mean(0).round(4).tolist(), "global_mean_max_abs": float(np.abs(mean_sg[xi]).max())}
                        for xi in (0, 4)}}
print("Fig3 done")

# ----------------------------------------------------------------------------------------------------------------
# Fig. 4  when the arrangement changes the recovered output
# ----------------------------------------------------------------------------------------------------------------
E_C, NH = 0.30, 4                        # hypothetical threshold-type (sigmoidal) local step
KAP = 6.0                                # hypothetical saturating local step


def g_sig(e):
    a = np.abs(e) ** NH
    return np.sign(e) * a / (a + E_C ** NH)


def g_sat(e):
    return np.tanh(KAP * e) / np.tanh(KAP * E_A)


MODELS = [("Uniform recovery ($q_i$ = 1): null", lambda e, c: e.mean()),
          ("Recovery co-located with selective sites", lambda e, c: (c * e).sum() / c.sum()),
          ("Threshold-type local step (convex at small e)", lambda e, c: g_sig(e).mean()),
          ("Saturating local step (concave)", lambda e, c: g_sat(e).mean())]
rng = np.random.default_rng(23)
res4 = {xi: np.zeros((len(MODELS), NREP, len(ELLS))) for xi in (0, 4)}
for xi in (0, 4):
    for r in range(NREP):
        c = binary_map(xi, rng)
        e = E_N + c * (E_A - E_N)
        for j, ell in enumerate(ELLS):
            em = mix(e, ell)
            for k, (_, fn) in enumerate(MODELS):
                res4[xi][k, r, j] = fn(em, c)
fig, axs = plt.subplots(1, 4, figsize=(10.6, 3.0), sharex=True)
for k, (name, _) in enumerate(MODELS):
    ax = axs[k]
    for q, xi in enumerate((0, 4)):
        lab = "dispersed" if xi == 0 else "clustered ($\\xi$ = 4)"
        ax.plot(ELLS, res4[xi][k].mean(0), color=PAL[[0, 3][q]], lw=1.7, marker="o", ms=2.6, ls="-" if q else "--", label=lab)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_title("abcd"[k] + "   " + name.replace(" (", "\n("), fontsize=8.2, loc="left")
    ax.set_xlabel(r"$\ell$ (sites)")
axs[0].set_ylim(0, 0.3)
axs[0].set_ylabel(r"Recovered $E_{\mathrm{out}}$")
axs[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig4_recovered_output.png"))
fig.savefig(os.path.join(OUT, "Fig4_recovered_output.tif"), pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
R["fig4"] = {"models": [m[0] for m in MODELS], "e_c": E_C, "n_hill": NH, "kappa": KAP, "ells": ELLS.tolist(),
             "dispersed": res4[0].mean(1).round(4).tolist(), "clustered_xi4": res4[4].mean(1).round(4).tolist()}
print("Fig4 done")

json.dump(R, open(os.path.join(OUT, "results_r2.json"), "w"), indent=1)
print(json.dumps({k: R[k] for k in ("fig2",)}, indent=1))
