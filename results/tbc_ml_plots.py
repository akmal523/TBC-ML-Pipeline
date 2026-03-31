"""
TBC-ML-Pipeline: Visualisation Script
Generates:
  1. Parity plot — Random Forest predictions vs. series-resistance analytical k_eff
  2. 3D surface   — k_eff as a function of k_YSZ and rho_YSZ (analytical model,
                    L_YSZ = 1.0 mm baseline; additional panel for k_YSZ vs L_YSZ)

Unit system throughout: mm – tonne – second (Abaqus-consistent).
  Thermal conductivity  : W mm⁻¹ K⁻¹
  Density               : t mm⁻³
  Specific heat         : mJ t⁻¹ K⁻¹
  Length                : mm
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

matplotlib.rcParams.update({
    "font.family":  "serif",
    "font.size":    11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":   150,
})


# ─────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET GENERATION
#     Mirrors the parametric design space used in the pipeline:
#     4 YSZ thicknesses × 128 randomised property variants = 512
# ─────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
N   = 512

# Fixed geometric and substrate parameters
L_SUB       = 10.0          # mm  — CMSX-4 substrate thickness
K_SUB_AVG   = 0.015         # W mm⁻¹ K⁻¹ — temperature-averaged CMSX-4 conductivity
                             # (≈15 W m⁻¹ K⁻¹ at mid-range; Epishin et al. 2001)
DELTA_T     = 800.0         # K   — imposed temperature difference (1400 – 600 K)

# YSZ nominal thermophysical properties
K_YSZ_NOM   = 0.002         # W mm⁻¹ K⁻¹  (≈ 2 W m⁻¹ K⁻¹, dense YSZ)
RHO_YSZ_NOM = 6.0e-9        # t mm⁻³       (≈ 6000 kg m⁻³)
CP_YSZ_NOM  = 5.0e8         # mJ t⁻¹ K⁻¹  (≈ 500 J kg⁻¹ K⁻¹)
L_YSZ_VALS  = [0.5, 1.0, 1.5, 2.0]  # mm — four discrete thickness levels

# Parametric feature columns
ysz_thickness = RNG.choice(L_YSZ_VALS, size=N)
k_ysz_avg     = K_YSZ_NOM  * (1.0 + RNG.uniform(-0.20, 0.20, N))   # ±20 % perturbation
rho_ysz       = RHO_YSZ_NOM * (1.0 + RNG.uniform(-0.20, 0.20, N))
cp_ysz        = CP_YSZ_NOM  * (1.0 + RNG.uniform(-0.20, 0.20, N))

# Analytical series-resistance k_eff  (Eq. keff_two_layer from thesis)
#   k_eff = (L_YSZ + L_sub) / (L_YSZ / k_YSZ + L_sub / k_sub)
L_tot           = ysz_thickness + L_SUB
k_eff_analytical = L_tot / (ysz_thickness / k_ysz_avg + L_SUB / K_SUB_AVG)


# ─────────────────────────────────────────────────────────────
# 2.  RANDOM FOREST TRAINING
# ─────────────────────────────────────────────────────────────
# Feature matrix: [L_YSZ, rho_YSZ, k_YSZ_avg, cp_YSZ_avg]
X = np.column_stack([ysz_thickness, rho_ysz, k_ysz_avg, cp_ysz])
y = k_eff_analytical

# 412 train / 100 test — replicates the pipeline split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100, random_state=42, shuffle=True
)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,              # depth-limited to control variance on 512-sample dataset
    min_samples_leaf=3,       # prevents memorisation of individual samples
    max_features=0.75,        # random-subspace fraction at each split node
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test  = mean_absolute_error(y_test,  y_pred_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

print("=" * 52)
print("  Random Forest Performance")
print("=" * 52)
print(f"  Training  R²   : {r2_train:.4f}")
print(f"  Test      R²   : {r2_test:.4f}")
print(f"  Training  MAE  : {mae_train:.4e}  W mm⁻¹ K⁻¹")
print(f"  Test      MAE  : {mae_test:.4e}  W mm⁻¹ K⁻¹")
print(f"  Test      RMSE : {rmse_test:.4e}  W mm⁻¹ K⁻¹")
print()

feat_names = [r"$L_\mathrm{YSZ}$", r"$\rho_\mathrm{YSZ}$",
              r"$\bar{k}_\mathrm{YSZ}$", r"$\bar{c}_{p,\mathrm{YSZ}}$"]
importances = rf.feature_importances_
print("  Feature importances (MDI):")
for fn, fi in zip(feat_names, importances):
    print(f"    {fn:<25s}: {fi*100:6.2f} %")
print("=" * 52)


# ─────────────────────────────────────────────────────────────
# 3.  FIGURE 1 — PARITY PLOT
# ─────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))

lo = min(y_test.min(), y_pred_test.min()) * 0.985
hi = max(y_test.max(), y_pred_test.max()) * 1.015

ax1.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--",
         zorder=2, label="Ideal (1 : 1)")

sc = ax1.scatter(y_test, y_pred_test, s=28, alpha=0.72,
                 c=np.abs(y_test - y_pred_test),
                 cmap="viridis", edgecolors="none", zorder=3,
                 label=rf"RF (test $R^2 = {r2_test:.4f}$)")

cbar1 = fig1.colorbar(sc, ax=ax1, pad=0.02)
cbar1.set_label(r"Absolute residual [W mm$^{-1}$ K$^{-1}$]", fontsize=9)

ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel(r"Analytical $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax1.set_ylabel(r"RF predicted $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax1.set_title("Parity Plot: Random Forest vs. Series-Resistance\n"
              r"($n_\mathrm{test}=100$, depth-limited RF, $\Delta T = 800$ K)")
ax1.legend(loc="upper left")

# Annotate performance metrics
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
fig1.savefig("parity_plot.png", dpi=150, bbox_inches="tight")
print("Saved: parity_plot.png")


# ─────────────────────────────────────────────────────────────
# 4.  FIGURE 2 — 3D SURFACE + CONTOUR
#     Panel A: k_eff( k_YSZ, rho_YSZ ) — demonstrates rho_YSZ
#              is analytically inert (flat ridge in rho direction)
#     Panel B: k_eff( k_YSZ, L_YSZ )   — demonstrates joint
#              geometric–constitutive sensitivity
# ─────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(13.5, 5.8))
fig2.suptitle(
    r"Theoretical $k_\mathrm{eff}$ Response Surface — Series-Resistance Model"
    "\n"
    r"$L_\mathrm{sub} = 10.0$ mm,  "
    r"$\bar{k}_\mathrm{sub} = 0.015$ W mm$^{-1}$ K$^{-1}$",
    fontsize=11, y=1.01,
)

# ── Panel A ────────────────────────────────────────────────
ax2a = fig2.add_subplot(1, 2, 1, projection="3d")

k_ysz_vec  = np.linspace(0.0014, 0.0036, 80)    # W mm⁻¹ K⁻¹
rho_ysz_vec = np.linspace(
    RHO_YSZ_NOM * 0.75, RHO_YSZ_NOM * 1.25, 80  # t mm⁻³  (±25 % range)
)
KG_a, RG_a = np.meshgrid(k_ysz_vec, rho_ysz_vec)

L_YSZ_FIXED = 1.0   # mm — representative baseline thickness
L_tot_fixed  = L_YSZ_FIXED + L_SUB
KEff_a = L_tot_fixed / (L_YSZ_FIXED / KG_a + L_SUB / K_SUB_AVG)
# NOTE: rho_ysz does not appear in the series-resistance formula →
#       the surface is a ruled surface (zero gradient in rho direction).

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

# ── Panel B ────────────────────────────────────────────────
ax2b = fig2.add_subplot(1, 2, 2, projection="3d")

k_ysz_vec2 = np.linspace(0.0014, 0.0036, 80)
L_ysz_vec2 = np.linspace(0.5, 2.0, 80)
KG_b, LG_b = np.meshgrid(k_ysz_vec2, L_ysz_vec2)

L_tot_b = LG_b + L_SUB
KEff_b  = L_tot_b / (LG_b / KG_b + L_SUB / K_SUB_AVG)

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
fig2.savefig("surface_plots.png", dpi=150, bbox_inches="tight")
print("Saved: surface_plots.png")

plt.show()
