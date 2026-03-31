import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. Analytical vs. Machine Learning Models ---

# Generate a dense range for plotting the theoretical curve
k_ysz_range = np.linspace(0.001, 0.005, 100).reshape(-1, 1)
L_sub, k_sub, L_ysz = 10.0, 0.025, 1.0

# Calculate "True" Analytical Effective Conductivity (Series Resistance Model)
# k_eff = (L_total) / sum(L_i / k_i)
k_eff_analytical = (L_ysz + L_sub) / (L_ysz / k_ysz_range + L_sub / k_sub)

# Generate synthetic training data with random uniform distribution
k_train = np.random.uniform(0.001, 0.005, (50, 1))
k_eff_train = (L_ysz + L_sub) / (L_ysz / k_train + L_sub / k_sub)

# Initialize and fit models: Random Forest (Non-linear) and Linear Regression
rf = RandomForestRegressor(n_estimators=100).fit(k_train, k_eff_train.ravel())
lr = LinearRegression().fit(k_train, k_eff_train)

# Explicitly create a figure object to avoid NameError when calling .savefig()
fig = plt.figure(figsize=(12, 5))

# Plotting the 2D comparison results
plt.subplot(1, 2, 1)
plt.plot(k_ysz_range, k_eff_analytical, 'k-', label='Analytical (True)')
plt.plot(k_ysz_range, lr.predict(k_ysz_range), 'r--', label='Linear Fit (Standard)')
plt.plot(k_ysz_range, rf.predict(k_ysz_range), 'b:', label='Random Forest (ML)')
plt.scatter(k_train, k_eff_train, c='gray', alpha=0.5, s=10, label='Train Data')
plt.xlabel('$k_{YSZ}$')
plt.ylabel('$k_{eff}$')
plt.title('Conductivity Model Comparison')
plt.legend()

# --- 2. Thermal Diffusivity Surface (Transient Model) ---

# Define ranges for physical parameters: Density (rho) and Heat Capacity (cp)
rho = np.linspace(1e-9, 9e-9, 50)
cp = np.linspace(4e8, 7e8, 50)
R, C = np.meshgrid(rho, cp)
k_fixed = 0.0025

# Calculate Thermal Diffusivity: Alpha = k / (rho * cp)
Alpha = k_fixed / (R * C)

# Add a 3D subplot to the existing figure object
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(R*1e9, C*1e-8, Alpha, cmap='magma')
ax.set_xlabel('Density $\\rho$')
ax.set_ylabel('Heat Capacity $c_p$')
ax.set_zlabel('Diffusivity $\\alpha$')
ax.set_title('Thermal Diffusivity Surface')

plt.tight_layout()

# CRITICAL: Save the figure BEFORE calling plt.show(). 
# plt.show() starts an event loop and can clear the current figure buffer in some backends.
fig.savefig("surface_plots.png", dpi=150, bbox_inches="tight")

# Display the plot window
plt.show()
