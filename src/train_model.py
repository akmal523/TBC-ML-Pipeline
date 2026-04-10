"""
src/train_model.py — TBC-ML-Pipeline: Random Forest training

Reads  : <repo>/results_ml/dataset.json
Writes : <repo>/results/ml_results.json
         <repo>/results/ml_results.png
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Path anchor ───────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(_HERE, ".."))
RESULTS_ML = os.path.join(ROOT, "results_ml")
RESULTS    = os.path.join(ROOT, "results")

os.makedirs(RESULTS, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(RESULTS_ML, "dataset.json")
PNG_PATH     = os.path.join(RESULTS, "ml_results.png")
JSON_PATH    = os.path.join(RESULTS, "ml_results.json")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(DATASET_PATH, "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

feature_names = ["YSZ Thickness (mm)", "YSZ Density", "YSZ k_avg", "YSZ Cp_avg"]
features_list, target_list = [], []

for sample in data:
    feat = sample["features"]
    features_list.append([
        feat["ysz_thickness_mm"],
        feat["ysz_density"],
        feat["ysz_k_avg"],
        feat["ysz_cp_avg"],
    ])
    target_list.append(sample["target"])

X = np.array(features_list)
y = np.array(target_list)

print(f"Feature matrix : {X.shape}")
print(f"Target vector  : {y.shape}")


# ── STEP 1: Feature importance ────────────────────────────────────────────────
print("\nSTEP 1: Feature importance analysis")

rf_full      = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(X, y)
importances  = rf_full.feature_importances_
indices      = np.argsort(importances)[::-1]

print("\nFeature importance ranking:")
for rank, idx in enumerate(indices):
    print(f"  {rank + 1}. {feature_names[idx]:<25s}  {importances[idx]:.4f}")

top_2_indices = indices[:2]
top_2_names   = [feature_names[i] for i in top_2_indices]

print(f"\nTop 2 features selected:")
for i, name in enumerate(top_2_names):
    print(f"  {i + 1}. {name}")

X_selected = X[:, top_2_indices]


# ── STEP 2: Train / test split ────────────────────────────────────────────────
print("\nSTEP 2: Train / test split")

n_train      = 412
n_test       = 100
total_needed = n_train + n_test

rng            = np.random.default_rng(seed=42)
random_indices = rng.choice(len(X_selected), size=total_needed, replace=False)
X_subset       = X_selected[random_indices]
y_subset       = y[random_indices]

X_train, y_train = X_subset[:n_train], y_subset[:n_train]
X_test,  y_test  = X_subset[n_train:], y_subset[n_train:]

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")


# ── STEP 3: Train model ───────────────────────────────────────────────────────
print("\nSTEP 3: Training Random Forest")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
)
rf_model.fit(X_train, y_train)
print("  Model trained.")


# ── STEP 4: Evaluate ──────────────────────────────────────────────────────────
print("\nSTEP 4: Evaluation")

y_train_pred = rf_model.predict(X_train)
y_test_pred  = rf_model.predict(X_test)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)
train_mae  = mean_absolute_error(y_train, y_train_pred)
test_mae   = mean_absolute_error(y_test,  y_test_pred)
train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse  = float(np.sqrt(mean_squared_error(y_test,  y_test_pred)))

print(f"\n  Training  R²   : {train_r2:.4f}")
print(f"  Training  MAE  : {train_mae:.6f}  W mm⁻¹ K⁻¹")
print(f"  Training  RMSE : {train_rmse:.6f}  W mm⁻¹ K⁻¹")
print(f"\n  Test      R²   : {test_r2:.4f}")
print(f"  Test      MAE  : {test_mae:.6f}  W mm⁻¹ K⁻¹")
print(f"  Test      RMSE : {test_rmse:.6f}  W mm⁻¹ K⁻¹")


# ── STEP 5: Visualisation ─────────────────────────────────────────────────────
print("\nSTEP 5: Creating visualisation")

fig = plt.figure(figsize=(16, 10))

# Panel 1 — Feature importance
ax1    = fig.add_subplot(2, 3, 1)
colors = ["tab:green" if i in top_2_indices else "tab:gray"
          for i in range(len(feature_names))]
ax1.barh(
    range(len(feature_names)),
    importances[indices],
    color=[colors[indices[i]] for i in range(len(feature_names))],
)
ax1.set_yticks(range(len(feature_names)))
ax1.set_yticklabels([feature_names[i] for i in indices])
ax1.set_xlabel("Importance")
ax1.set_title("Feature importance (top 2 in green)")
ax1.grid(axis="x", alpha=0.3)

# Panel 2 — Parity, training
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(y_train, y_train_pred, alpha=0.6, s=60, edgecolors="k", label="Training")
lo, hi = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
ax2.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
ax2.set_xlabel(r"Actual $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax2.set_ylabel(r"Predicted $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax2.set_title(f"Training parity  ($R^2 = {train_r2:.4f}$)")
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3 — Parity, test
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.6, s=60,
            color="tab:orange", edgecolors="k", label="Test")
lo, hi = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
ax3.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
ax3.set_xlabel(r"Actual $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax3.set_ylabel(r"Predicted $k_\mathrm{eff}$  [W mm$^{-1}$ K$^{-1}$]")
ax3.set_title(f"Test parity  ($R^2 = {test_r2:.4f}$)")
ax3.legend()
ax3.grid(alpha=0.3)

# Panel 4 — Residuals, training
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(y_train_pred, y_train - y_train_pred, alpha=0.6, s=60, edgecolors="k")
ax4.axhline(0, color="r", ls="--", lw=1.5)
ax4.set_xlabel(r"Predicted $k_\mathrm{eff}$")
ax4.set_ylabel("Residual  [W mm$^{-1}$ K$^{-1}$]")
ax4.set_title("Training residuals")
ax4.grid(alpha=0.3)

# Panel 5 — Residuals, test
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6, s=60,
            color="tab:orange", edgecolors="k")
ax5.axhline(0, color="r", ls="--", lw=1.5)
ax5.set_xlabel(r"Predicted $k_\mathrm{eff}$")
ax5.set_ylabel("Residual  [W mm$^{-1}$ K$^{-1}$]")
ax5.set_title("Test residuals")
ax5.grid(alpha=0.3)

# Panel 6 — Metric comparison bar chart
ax6     = fig.add_subplot(2, 3, 6)
metrics = ["R²", "MAE", "RMSE"]
t_vals  = [train_r2, train_mae, train_rmse]
v_vals  = [test_r2,  test_mae,  test_rmse]
x_pos   = np.arange(len(metrics))
w       = 0.35
bars1   = ax6.bar(x_pos - w / 2, t_vals, w, label="Training", alpha=0.8)
bars2   = ax6.bar(x_pos + w / 2, v_vals, w, label="Test",     alpha=0.8)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metrics)
ax6.set_ylabel("Score")
ax6.set_title("Performance metrics")
ax6.legend()
ax6.grid(axis="y", alpha=0.3)
for bar_group in (bars1, bars2):
    for bar in bar_group:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
print(f"  Saved: {PNG_PATH}")
plt.show()


# ── Save JSON results ─────────────────────────────────────────────────────────
results_dict = {
    "feature_importance": {
        name: float(imp) for name, imp in zip(feature_names, importances)
    },
    "top_2_features":  top_2_names,
    "training_size":   int(n_train),
    "test_size":       int(n_test),
    "training_metrics": {
        "r2":   float(train_r2),
        "mae":  float(train_mae),
        "rmse": float(train_rmse),
    },
    "test_metrics": {
        "r2":   float(test_r2),
        "mae":  float(test_mae),
        "rmse": float(test_rmse),
    },
}

with open(JSON_PATH, "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"  Saved: {JSON_PATH}")