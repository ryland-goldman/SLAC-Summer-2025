import numpy as np
import matplotlib.pyplot as plt

# ---- Data grouped by base shape ----
bases = ["Foil", "Single Crystal", "Mesh Grid"]

# Moderator yield (e+/e+ × 1e-7)
mod_no_cav   = np.array([14.9, 24.6, 332.5])
mod_no_cav_e = np.array([5.5, 8.4, 26.0])

mod_cav      = np.array([376.5, 908.6, 849.7])
mod_cav_e    = np.array([34.5, 53.6, 51.8])

# System efficiency (e+/e- × 1e-8)
eff_no_cav   = np.array([38.1, 88.5, 850.7])
eff_no_cav_e = np.array([14.1, 21.5, 66.6])

eff_cav      = np.array([620.0, 1497, 1399.0])
eff_cav_e    = np.array([56.5, 88.3, 85.4])

x = np.arange(len(bases))
width = 0.38

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

fig, ax2 = plt.subplots(1, 1, figsize=(12, 5), sharex=True)


# ---- Left: Moderator yield ----
# ax1.bar(x - width/2, mod_no_cav, width, yerr=mod_no_cav_e, capsize=5, label="No cavity")
# ax1.bar(x + width/2, mod_cav,    width, yerr=mod_cav_e,    capsize=5, label="With cavity")
# ax1.set_title("Moderator Yield")
# ax1.set_ylabel("e+/e+ × 1e-7")
# ax1.set_xticks(x, bases)
# ax1.legend()
# ax1.grid(axis="y", alpha=0.3)

# ---- Right: System efficiency ----
ax2.bar(x - width/2, eff_no_cav, width, yerr=eff_no_cav_e, capsize=5, label="No cavity", color="blue")
ax2.bar(x + width/2, eff_cav,    width, yerr=eff_cav_e,    capsize=5, label="With cavity", color="darkred")
ax2.set_ylabel("System Efficiency (e+/e- ×1e-8)")
ax2.set_xticks(x, bases)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

improvement = eff_cav / eff_no_cav[0]  # baseline: Foil, No cavity
ax2b = ax2.twinx()
ax2b.set_ylabel("Improvement Multiple")
ax2b.set_ylim(ax2.get_ylim()[0] / eff_no_cav[0], ax2.get_ylim()[1] / eff_no_cav[0])
ax2b.tick_params(axis='y', labelcolor='gray')

fig.tight_layout()
plt.show()