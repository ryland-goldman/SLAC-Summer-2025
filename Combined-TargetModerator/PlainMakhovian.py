import numpy as np
import matplotlib.pyplot as plt

# Define Makhovian distribution
def makhovian_distribution(z, z0, m=1.89):
    return (m * z**(m - 1) / z0**m) * np.exp(-(z / z0)**m)

# Energy (MeV) and corresponding z0 (micrometers)
energies_z0 = {
    0.1: 3.79,
    0.5: 42.46,
    1.0: 106.46,
    5.0: 649.79
}

# Depth range (micrometers) for plotting
z = np.linspace(0, 100, 2000)

# Plot
plt.figure(figsize=(10, 6))
for energy, z0 in energies_z0.items():
    pz = makhovian_distribution(z, z0)
    plt.plot(z, pz, label=f"{energy} MeV")

plt.xlabel("Depth z (μm)", fontsize=18)
plt.ylabel("Probability Density p(z)", fontsize=18)
plt.legend(title="Positron KE", fontsize=14, title_fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()