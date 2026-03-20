import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

# Load dataset
print("Loading dataset...")
with open("results_ml/dataset.json", "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Extract features and target
features_list = []
target_list = []

for sample in data:
    feat = sample["features"]
    features_list.append([
        feat["ysz_thickness_mm"],
        feat["ysz_density"],
        feat["ysz_k_avg"],
        feat["ysz_cp_avg"]
    ])
    target_list.append(sample["target"])

X = np.array(features_list)
y = np.array(target_list)

feature_names = ["YSZ Thickness (mm)", "YSZ Density", "YSZ k_avg", "YSZ Cp_avg"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


# STEP 1: Feature Importance Analysis


print("STEP 1: Analyzing Feature Importance")


# Train Random Forest on all data to get feature importance
rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(X, y)

importances = rf_full.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance Ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:<25} {importances[idx]:.4f}")

# Select top 2 features
top_2_indices = indices[:2]
top_2_names = [feature_names[i] for i in top_2_indices]

print(f"\nTop 2 Features Selected:")
for i, name in enumerate(top_2_names):
    print(f"  {i+1}. {name}")

X_selected = X[:, top_2_indices]


# STEP 2: Create Training and Test Datasets


print("STEP 2: Creating Training and Test Datasets")


# Randomly select 20 samples for training, 5 for test
#np.random.seed(42)
n_train = 412
n_test = 100
total_needed = n_train + n_test

# Select random subset
random_indices = np.random.choice(len(X_selected), size=total_needed, replace=False)
X_subset = X_selected[random_indices]
y_subset = y[random_indices]

# Split into train and test
X_train = X_subset[:n_train]
y_train = y_subset[:n_train]
X_test = X_subset[n_train:]
y_test = y_subset[n_train:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# STEP 3: Train Random Forest Model


print("STEP 3: Training Random Forest Model")


rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

rf_model.fit(X_train, y_train)
print("Model trained successfully!")


# STEP 4: Evaluate Model

print("STEP 4: Model Evaluation")


# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nTraining Performance:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE: {train_mae:.6f}")
print(f"  RMSE: {train_rmse:.6f}")

print(f"\nTest Performance:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE: {test_mae:.6f}")
print(f"  RMSE: {test_rmse:.6f}")


# STEP 5: Visualization


print("STEP 5: Creating Visualizations")


fig = plt.figure(figsize=(16, 10))

# Plot 1: Feature Importance
ax1 = plt.subplot(2, 3, 1)
colors = ['green' if i in top_2_indices else 'gray' for i in range(len(feature_names))]
bars = ax1.barh(range(len(feature_names)), importances[indices], color=[colors[indices[i]] for i in range(len(feature_names))])
ax1.set_yticks(range(len(feature_names)))
ax1.set_yticklabels([feature_names[i] for i in indices])
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance (Top 2 in Green)')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Predicted vs Actual (Training)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_train, y_train_pred, alpha=0.6, s=100, edgecolors='k', label='Training')
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
ax2.set_xlabel('Actual k_effective (W/mm·K)')
ax2.set_ylabel('Predicted k_effective (W/mm·K)')
ax2.set_title(f'Training Set\nR² = {train_r2:.4f}')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Predicted vs Actual (Test)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.6, s=100, color='orange', edgecolors='k', label='Test')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
ax3.set_xlabel('Actual k_effective (W/mm·K)')
ax3.set_ylabel('Predicted k_effective (W/mm·K)')
ax3.set_title(f'Test Set\nR² = {test_r2:.4f}')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Residuals (Training)
ax4 = plt.subplot(2, 3, 4)
residuals_train = y_train - y_train_pred
ax4.scatter(y_train_pred, residuals_train, alpha=0.6, s=100, edgecolors='k')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted k_effective (W/mm·K)')
ax4.set_ylabel('Residuals (W/mm·K)')
ax4.set_title('Training Residuals')
ax4.grid(alpha=0.3)

# Plot 5: Residuals (Test)
ax5 = plt.subplot(2, 3, 5)
residuals_test = y_test - y_test_pred
ax5.scatter(y_test_pred, residuals_test, alpha=0.6, s=100, color='orange', edgecolors='k')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted k_effective (W/mm·K)')
ax5.set_ylabel('Residuals (W/mm·K)')
ax5.set_title('Test Residuals')
ax5.grid(alpha=0.3)

# Plot 6: Performance Comparison
ax6 = plt.subplot(2, 3, 6)
metrics = ['R²', 'MAE', 'RMSE']
train_metrics = [train_r2, train_mae, train_rmse]
test_metrics = [test_r2, test_mae, test_rmse]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, train_metrics, width, label='Training', alpha=0.8)
bars2 = ax6.bar(x_pos + width/2, test_metrics, width, label='Test', alpha=0.8)

ax6.set_ylabel('Score')
ax6.set_title('Model Performance Metrics')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metrics)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('ml_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'ml_results.png'")
plt.show()


# Save Results

results = {
    "feature_importance": {
        name: float(imp) for name, imp in zip(feature_names, importances)
    },
    "top_2_features": top_2_names,
    "training_size": int(n_train),
    "test_size": int(n_test),
    "training_metrics": {
        "r2": float(train_r2),
        "mae": float(train_mae),
        "rmse": float(train_rmse)
    },
    "test_metrics": {
        "r2": float(test_r2),
        "mae": float(test_mae),
        "rmse": float(test_rmse)
    }
}

with open("ml_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to 'ml_results.json'")

