import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('results', exist_ok=True)

complexity = np.linspace(1, 10, 200)

# Analytical: near-zero cost, valid only up to sigma ~ 4.5
analytical = np.where(complexity <= 4.5, 1e-3, np.nan)

# FEA: exponential growth
fea = 0.01 * np.exp(0.9 * complexity)

# ML one-time training cost (paid once, horizontal)
ml_training = np.full_like(complexity, 5.0)

# ML inference: flat and low (above analytical, well below FEA)
ml_inference = np.full_like(complexity, 0.05)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(complexity, analytical,
        color='green', lw=2.5, ls='-',
        label='Analytical model')

ax.plot(complexity, fea,
        color='blue', lw=2.5, ls='-',
        label='FEA (direct solve)')

ax.plot(complexity, ml_training,
        color='red', lw=1.8, ls='--',
        label='ML: one-time training cost')

ax.plot(complexity, ml_inference,
        color='red', lw=2.5, ls='-',
        label='ML: per-query inference')

# Mark where analytical breaks down
ax.axvline(x=4.5, color='green', lw=1.0, ls=':')
ax.text(4.55, 5e-3, 'Analytical\nfails', color='green',
        fontsize=8, va='bottom')

# Mark crossover: FEA vs ML inference
crossover_sigma = np.log(0.05 / 0.01) / 0.9
ax.axvline(x=crossover_sigma, color='purple', lw=1.0, ls=':')
ax.text(crossover_sigma + 0.1, 0.02,
        f'Crossover\n$\\sigma\\approx{crossover_sigma:.1f}$',
        color='purple', fontsize=8, va='bottom')

# Mark current work
ax.axvline(x=2.0, color='grey', lw=1.2, ls='-.')
ax.text(2.05, 2e-4, 'This work\n(2D steady-state)',
        color='grey', fontsize=8, va='bottom')

ax.set_yscale('log')
ax.set_xlabel('Structural / Physical Complexity ($\\sigma$)', fontsize=12)
ax.set_ylabel('Computational Cost (normalised, log scale)', fontsize=12)
ax.set_title('Computational Scaling Paradigms in TBC Modelling', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, which='both', ls='-', alpha=0.2)

ax.set_xlim(1, 10)
ax.set_ylim(5e-5, 5e3)

# Annotation: complexity labels on x-axis
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_xticklabels([
    '1\n(1D steady)', '2\n(2D steady)', '3', '4',
    '5\n(3D steady)', '6', '7\n(3D transient)', '8', '9', '10\n(3D stochastic)'
], fontsize=7)

plt.tight_layout()
plt.savefig('results/crossover.png', dpi=150, bbox_inches='tight')
print('Saved: results/crossover.png')
plt.show()