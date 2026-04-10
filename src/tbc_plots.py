"""
src/tbc_plots.py — TBC-ML-Pipeline Visualisation
Replaces comp.py and tbc_ml_plots.py.

Produces three figures written to --out-dir (default: <repo>/results/):
  parity_plot.png      — RF predictions vs analytical k_eff (test set)
  surface_plots.png    — k_eff 3D response surfaces:
                           (a) k_eff( k_YSZ, rho_YSZ )  L_YSZ fixed
                           (b) k_eff( k_YSZ, L_YSZ )    rho_YSZ analytically inert
  model_comparison.png — (a) Analytical / LR / RF comparison at L_YSZ = 1 mm slice
                           (b) Thermal diffusivity surface alpha( rho, c_p )

Unit system: mm – tonne – second  (Abaqus-consistent)
  Thermal conductivity  : W mm⁻¹ K⁻¹
  Density               : t mm⁻³
  Specific heat         : mJ t⁻¹ K⁻¹
  Length                : mm
  Thermal diffusivity   : mm² s⁻¹
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "results"))

parser = argparse.ArgumentParser(description="TBC-ML-Pipeline visualisation")
parser.add_argument(
    "--out-dir",
    default=_DEFAULT_OUT,
    help=f"Output directory for PNG files (default: {_DEFAULT_OUT})",
)
args = parser.parse_args()

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  Global plot style
# ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET
#     Design space: 4 YSZ thicknesses × 128 randomised property variants = 512
# ──────────────────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
N   = 512

# Fixed substrate parameters
L_SUB     = 10.0    # mm
K_SUB_AVG = 0.015   # W mm⁻¹ K⁻¹  (≈ 15 W m⁻¹ K⁻¹, CMSX-4 mid-range)
DELTA_T   = 800.0   # K

# YSZ nominal properties
K_YSZ_NOM   = 0.002      # W mm⁻¹ K⁻¹  (≈ 2 W m⁻¹ K⁻¹, dense YSZ)
RHO_YSZ_NOM = 6.0e-9     # t mm⁻³       (≈ 6000 kg m⁻³)
CP_YSZ_NOM  = 5.0e8      # mJ t⁻¹ K⁻¹  (≈ 500 J kg⁻¹ K⁻¹)
L_YSZ_VALS  = [0.5, 1.0, 1.5, 2.0]   # mm

ysz_thickness = RNG.choice(L_YSZ_VALS, size=N)
k_ysz_avg     = K_YSZ_NOM   * (1.0 + RNG.uniform(-0.20, 0.20, N))   # ±20 %
rho_ysz       = RHO_YSZ_NOM * (1.0 + RNG.uniform(-0.20, 0.20, N))
cp_ysz        = CP_YSZ_NOM  * (1.0 + RNG.uniform(-0.20, 0.20, N))

# Series-resistance analytical k_eff:  k_eff = (L_YSZ + L_sub) / Σ(L_i / k_i)
L_tot            = ysz_thickness + L_SUB
k_eff_analytical = L_tot / (ysz_thickness / k_ysz_avg + L_SUB / K_SUB_AVG)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  RANDOM FOREST — 4-feature model (for parity plot and importance)
# ──────────────────────────────────────────────────────────────────────────────
X = np.column_stack([ysz_thickness, rho_ysz, k_ysz_avg, cp_ysz])
y = k_eff_analytical

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100, random_state=42, shuffle=True
)

rf_full = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.75,
    random_state=42,
    n_jobs=-1,
)
rf_full.fit(X_train, y_train)

y_pred_train = rf_full.predict(X_train)
y_pred_test  = rf_full.predict(X_test)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test  = mean_absolute_error(y_test,  y_pred_test)
rmse_test = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))

feat_names  = [r"$L_\mathrm{YSZ}$", r"$\rho_\mathrm{YSZ}$",
               r"$\bar{k}_\mathrm{YSZ}$", r"$\bar{c}_{p,\mathrm{YSZ}}$"]
importances = rf_full.feature_importances_

print("=" * 52)
print("  Random Forest — 4-feature model")
print("=" * 52)
print(f"  Train R²   : {r2_train:.4f}")
print(f"  Test  R²   : {r2_test:.4f}")
print(f"  Train MAE  : {mae_train:.4e}  W mm⁻¹ K⁻¹")
print(f"  Test  MAE  : {mae_test:.4e}  W mm⁻¹ K⁻¹")
print(f"  Test  RMSE : {rmse_test:.4e}  W mm⁻¹ K⁻¹")
print()
print("  Feature importances (MDI):")
for fn, fi in zip(feat_names, importances):
    print(f"    {fn:<26s}: {fi * 100:6.2f} %")
print("=" * 52)
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3.  1-D RF — k_YSZ only, L_YSZ = 1.0 mm slice
#     Used exclusively for the model-comparison panel (Figure 3a).
#     Training on all L=1 mm samples gives a fair single-variable baseline.
#     LinearRegression removed: scipy lstsq conflicts with the ABAQUS MKL build.
# ──────────────────────────────────────────────────────────────────────────────
mask_1mm = np.isclose(ysz_thickness, 1.0)
k_1d_train = k_ysz_avg[mask_1mm].reshape(-1, 1)
y_1d_train = k_eff_analytical[mask_1mm]

rf_1d = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_1d.fit(k_1d_train, y_1d_train)

# Dense k_YSZ range for smooth curve plotting
k_plot = np.linspace(0.0014, 0.0036, 300).reshape(-1, 1)
L_ysz_fixed = 1.0
k_eff_curve = (L_ysz_fixed + L_SUB) / (L_ysz_fixed / k_plot + L_SUB / K_SUB_AVG)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PARITY PLOT
#   RF (test set) predictions vs analytical k_eff, coloured by absolute residual
# ──────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))

lo = min(y_test.min(), y_pred_test.min()) * 0.985
hi = max(y_test.max(), y_pred_test.max()) * 1.015

ax1.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--",
         zorder=2, label="Ideal (1:1)")

sc = ax1.scatter(
    y_test, y_pred_test,
    s=28, alpha=0.72,
    c=np.abs(y_test - y_pred_test),
    cmap="viridis", edgecolors="none", zorder=3,
    label=rf"RF  (test $R^2 = {r2_test:.4f}$)",
)
cbar1 = fig1.colorbar(sc, ax=ax1, pad=0.02)
cbar1.set_label(r"$|$residual$|$  [W mm$^{-1}$ K$^{-1}$]", fontsize=9)

ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel(r"Analytical $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax1.set_ylabel(r"RF predicted $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax1.set_title(
    "Parity Plot: Random Forest vs. Series-Resistance\n"
    r"($n_\mathrm{test}=100$, depth-limited RF, $\Delta T = 800$ K)"
)
ax1.legend(loc="upper left")

stats_text = (
    rf"Train $R^2 = {r2_train:.4f}$" + "\n"
    rf"Test  $R^2 = {r2_test:.4f}$"  + "\n"
    rf"MAE $= {mae_test:.2e}$ W mm$^{{-1}}$ K$^{{-1}}$"
)
ax1.text(0.97, 0.05, stats_text, transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="grey", alpha=0.85))

fig1.tight_layout()
_p1 = os.path.join(OUT_DIR, "parity_plot.png")
fig1.savefig(_p1, dpi=150, bbox_inches="tight")
print(f"Saved: {_p1}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — 3-D k_eff RESPONSE SURFACES
#   Panel (a): k_eff( k_YSZ, rho_YSZ )  with L_YSZ = 1 mm fixed
#              Shows rho_YSZ is analytically inert — flat ridge in rho direction
#   Panel (b): k_eff( k_YSZ, L_YSZ )
#              Shows joint geometric–constitutive sensitivity
# ──────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(13.5, 5.8))
fig2.suptitle(
    r"Theoretical $k_\mathrm{eff}$ Response Surface — Series-Resistance Model"
    "\n"
    r"$L_\mathrm{sub} = 10.0$ mm,  "
    r"$\bar{k}_\mathrm{sub} = 0.015$ W mm$^{-1}$ K$^{-1}$",
    fontsize=11, y=1.01,
)

# ── Panel (a) ─────────────────────────────────────────
ax2a = fig2.add_subplot(1, 2, 1, projection="3d")

k_ysz_vec_a  = np.linspace(0.0014, 0.0036, 80)
rho_ysz_vec  = np.linspace(RHO_YSZ_NOM * 0.75, RHO_YSZ_NOM * 1.25, 80)
KG_a, RG_a  = np.meshgrid(k_ysz_vec_a, rho_ysz_vec)

L_YSZ_1MM   = 1.0
KEff_a = (L_YSZ_1MM + L_SUB) / (L_YSZ_1MM / KG_a + L_SUB / K_SUB_AVG)

surf_a = ax2a.plot_surface(
    KG_a * 1e3, RG_a / RHO_YSZ_NOM, KEff_a,
    cmap="plasma", alpha=0.91, edgecolor="none",
    rcount=60, ccount=60,
)
ax2a.set_xlabel(r"$k_\mathrm{YSZ}$  [×10$^{-3}$ W mm$^{-1}$ K$^{-1}$]",
                fontsize=9, labelpad=7)
ax2a.set_ylabel(r"$\rho_\mathrm{YSZ}$ / $\rho_0$  [—]",
                fontsize=9, labelpad=7)
ax2a.set_zlabel(r"$k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]",
                fontsize=9, labelpad=7)
ax2a.set_title(
    r"(a)  $k_\mathrm{eff}(k_\mathrm{YSZ},\,\rho_\mathrm{YSZ})$"
    "\n"
    r"$L_\mathrm{YSZ} = 1.0$ mm fixed",
    fontsize=9,
)
cbar2a = fig2.colorbar(surf_a, ax=ax2a, shrink=0.52, aspect=14, pad=0.08)
cbar2a.set_label(r"$k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]", fontsize=8)

# ── Panel (b) ─────────────────────────────────────────
ax2b = fig2.add_subplot(1, 2, 2, projection="3d")

k_ysz_vec_b = np.linspace(0.0014, 0.0036, 80)
L_ysz_vec   = np.linspace(0.5, 2.0, 80)
KG_b, LG_b = np.meshgrid(k_ysz_vec_b, L_ysz_vec)

KEff_b = (LG_b + L_SUB) / (LG_b / KG_b + L_SUB / K_SUB_AVG)

surf_b = ax2b.plot_surface(
    KG_b * 1e3, LG_b, KEff_b,
    cmap="viridis", alpha=0.91, edgecolor="none",
    rcount=60, ccount=60,
)
ax2b.set_xlabel(r"$k_\mathrm{YSZ}$  [×10$^{-3}$ W mm$^{-1}$ K$^{-1}$]",
                fontsize=9, labelpad=7)
ax2b.set_ylabel(r"$L_\mathrm{YSZ}$  [mm]",
                fontsize=9, labelpad=7)
ax2b.set_zlabel(r"$k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]",
                fontsize=9, labelpad=7)
ax2b.set_title(
    r"(b)  $k_\mathrm{eff}(k_\mathrm{YSZ},\,L_\mathrm{YSZ})$"
    "\n"
    r"$\rho_\mathrm{YSZ}$ analytically inert",
    fontsize=9,
)
cbar2b = fig2.colorbar(surf_b, ax=ax2b, shrink=0.52, aspect=14, pad=0.08)
cbar2b.set_label(r"$k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]", fontsize=8)

fig2.tight_layout()
_p2 = os.path.join(OUT_DIR, "surface_plots.png")
fig2.savefig(_p2, dpi=150, bbox_inches="tight")
print(f"Saved: {_p2}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — MODEL COMPARISON
#   Panel (a): Conductivity model comparison at L_YSZ = 1 mm
#              Analytical series-resistance vs Random Forest (1-D)
#              LinearRegression omitted — MKL/scipy conflict in ABAQUS Python.
#   Panel (b): Thermal diffusivity surface  alpha = k / (rho * c_p)
#              Demonstrates transient sensitivity to density and specific heat,
#              which are analytically inert in the steady-state k_eff formula.
# ──────────────────────────────────────────────────────────────────────────────
fig3, (ax3a, ax3b_placeholder) = plt.subplots(
    1, 2, figsize=(12, 5),
    gridspec_kw={"width_ratios": [1, 1]},
)
fig3.delaxes(ax3b_placeholder)   # replace right panel with a 3-D axes

# ── Panel (a) — 2-D model comparison ──────────────────────
ax3a.plot(k_plot * 1e3, k_eff_curve,
          color="black", lw=1.8, ls="-",
          label="Analytical (series-resistance)")
ax3a.plot(k_plot * 1e3, rf_1d.predict(k_plot),
          color="tab:blue", lw=1.4, ls=":",
          label="Random Forest (1-D)")
ax3a.scatter(k_1d_train * 1e3, y_1d_train,
             c="grey", alpha=0.45, s=12, zorder=2,
             label=r"Train data  ($L_\mathrm{YSZ}=1.0$ mm)")

ax3a.set_xlabel(r"$k_\mathrm{YSZ}$  [×10$^{-3}$ W mm$^{-1}$ K$^{-1}$]")
ax3a.set_ylabel(r"$k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax3a.set_title(
    "(a)  Conductivity Model Comparison\n"
    r"$L_\mathrm{YSZ} = 1.0$ mm,  $L_\mathrm{sub} = 10.0$ mm"
)
ax3a.legend(fontsize=8.5)

# ── Panel (b) — Thermal diffusivity surface ────────────────
ax3b = fig3.add_subplot(1, 2, 2, projection="3d")

rho_vec = np.linspace(RHO_YSZ_NOM * 0.80, RHO_YSZ_NOM * 1.20, 60)
cp_vec  = np.linspace(CP_YSZ_NOM  * 0.80, CP_YSZ_NOM  * 1.20, 60)
R_g, C_g = np.meshgrid(rho_vec, cp_vec)
K_FIXED  = K_YSZ_NOM   # W mm⁻¹ K⁻¹
Alpha_g  = K_FIXED / (R_g * C_g)    # mm² s⁻¹  (mJ = mm² t s⁻²)

surf3b = ax3b.plot_surface(
    R_g / RHO_YSZ_NOM, C_g / CP_YSZ_NOM, Alpha_g,
    cmap="magma", alpha=0.91, edgecolor="none",
    rcount=50, ccount=50,
)
ax3b.set_xlabel(r"$\rho_\mathrm{YSZ}$ / $\rho_0$  [—]", fontsize=9, labelpad=7)
ax3b.set_ylabel(r"$c_{p,\mathrm{YSZ}}$ / $c_{p,0}$  [—]", fontsize=9, labelpad=7)
ax3b.set_zlabel(r"$\alpha$  [mm² s$^{-1}$]", fontsize=9, labelpad=7)
ax3b.set_title(
    r"(b)  Thermal Diffusivity  $\alpha(\rho,\,c_p)$"
    "\n"
    rf"$k_\mathrm{{YSZ}} = {K_FIXED*1e3:.1f}$×10$^{{-3}}$ W mm$^{{-1}}$ K$^{{-1}}$ fixed",
    fontsize=9,
)
cbar3b = fig3.colorbar(surf3b, ax=ax3b, shrink=0.52, aspect=14, pad=0.08)
cbar3b.set_label(r"$\alpha$  [mm² s$^{-1}$]", fontsize=8)

fig3.tight_layout()
_p3 = os.path.join(OUT_DIR, "model_comparison.png")
fig3.savefig(_p3, dpi=150, bbox_inches="tight")
print(f"Saved: {_p3}")

plt.show()