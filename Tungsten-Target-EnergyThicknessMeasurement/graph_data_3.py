#!/usr/bin/env python3
"""
this was entirely generated with chatgpt

(This is for both tungsten and tantalum)

Draws TWO stacked panels from batch-scan data:

  1.  e⁺/e⁻ yield vs converter thickness  (identical to your original plot)
  2.  optimum converter thickness vs primary-beam energy
      – with asymmetric error bars derived from the raw counts
      – plus the three literature parameterisations reproduced from the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Load the data ----------------------------------------------------------
# ---------------------------------------------------------------------------
DATA_FILE_TA = Path("batchdata-Ta.csv")
DATA_FILE_W = Path("batchdata-W.csv")

if not DATA_FILE_TA.exists():
    raise FileNotFoundError(f"{DATA_FILE_TA} not found – check the path.")
if not DATA_FILE_W.exists():
    raise FileNotFoundError(f"{DATA_FILE_W} not found – check the path.")

df_ta = pd.read_csv(DATA_FILE_TA)
df_w = pd.read_csv(DATA_FILE_W)

# ---------------------------------------------------------------------------
# 2.  Create a single figure with two rows -----------------------------------
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4))

optimal_points = []      # list of (energy, l_opt [mm], yield_at_lopt)

# ---------------------------------------------------------------------------
# 3.  *** ORIGINAL CODE – MODIFIED FOR ENERGY=100 ONLY ***  ------------------
# ---------------------------------------------------------------------------
energy = 100
datasets = [('Ta', df_ta, 3.5), ('W', df_w, 3.3)]  # (label, dataframe, X0)

for label, df, X0 in datasets:
    group = df[df['Energy'] == energy].sort_values('Thickness')
    if not group.empty:
        color = 'darkblue' if label == 'Ta' else 'darkred'
        style = {'marker': 'o'}

        y     = group['Count'] / 100_000.0
        y_err = np.sqrt(group['Count']) / 100_000.0

        ax.plot(group['Thickness'], y,
                label=label,
                marker=style['marker'],
                linestyle='-',
                color=color)

        ax.errorbar(group['Thickness'], y,
                    yerr=y_err,
                    fmt='none',
                    ecolor=None,
                    elinewidth=1,
                    capsize=2,
                    alpha=0.6)

        opt_row = group.loc[group['Count'].idxmax()]
        optimal_points.append((energy,
                               opt_row['Thickness'],
                               opt_row['Count'] / 100_000.0))

        # Print optimal thickness and 90% yield range for this material at this energy
        opt_thickness = float(opt_row['Thickness'])
        opt_yield = float(opt_row['Count']) / 100_000.0
        threshold = 0.9 * opt_yield
        # Boolean mask where yield >= 90% of optimal
        mask_90 = (group['Count'] / 100_000.0) >= threshold
        if mask_90.any():
            t_range = group.loc[mask_90, 'Thickness']
            t_min = float(t_range.min())
            t_max = float(t_range.max())
            print(f"{label}: optimal thickness = {opt_thickness:.3f} mm; 90% range = [{t_min:.3f}, {t_max:.3f}] mm")
        else:
            print(f"{label}: optimal thickness = {opt_thickness:.3f} mm; 90% range = [n/a]")

        ax.axvline(opt_thickness,
                   color=color,
                   linestyle='--',
                   linewidth=1,
                   alpha=0.8,
                   label=f'{label} optimal')

        # Add horizontal error bar to represent 90% range
        if mask_90.any():
            y_pos = 0.3 * opt_yield
            if label == 'W':
                y_pos = 0.2 * opt_yield
            ax.errorbar(opt_thickness, y_pos,
                        xerr=[[opt_thickness - t_min], [t_max - opt_thickness]],
                        fmt='none',
                        ecolor=color,
                        capsize=3,
                        alpha=0.8)

# Dashed polyline through optimal points
opt_xy = np.array([[p[1], p[2]] for p in optimal_points])
# ax.plot(opt_xy[:, 0], opt_xy[:, 1], 'k--', label=r'Optimal $l_{opt}$')

# Axis styling (unchanged)
ax.set_yscale('log')
ax.set_xlabel('Target Thickness (mm)')
ax.set_ylabel(r'$\gamma_t$ (e$^+$/e$^-$)')

# Pad the x-axis a bit on the right
current_xlim = ax.get_xlim()
ax.set_xlim(current_xlim[0], current_xlim[1] + 0.8)

# --- 4.  Secondary x-axis in radiation lengths ------------------------------
X0_mm = 3.5                               # Ta radiation length (mm)
mm_to_X0 = lambda l_mm: l_mm / X0_mm
X0_to_mm = lambda l_x0: l_x0 * X0_mm

secax1 = ax.secondary_xaxis('top', functions=(mm_to_X0, X0_to_mm))
secax1.set_xlabel(r'Target Thickness ($X_0$, Ta)')
secax1.set_xlim(mm_to_X0(ax.get_xlim()[0]), mm_to_X0(ax.get_xlim()[1]))

X0_W_mm = 3.3  # W radiation length (mm)

secax2 = ax.secondary_xaxis(1.15, functions=(lambda l_mm: l_mm / X0_W_mm, lambda l_x0: l_x0 * X0_W_mm))
secax2.set_xlabel(r'Target Thickness ($X_0$, W)')
secax2.set_xlim(mm_to_X0(ax.get_xlim()[0]), mm_to_X0(ax.get_xlim()[1]))

ax.legend(fontsize=9)

# ---------------------------------------------------------------------------
# 5.  Render -----------------------------------------------------------------
# ---------------------------------------------------------------------------
plt.tight_layout()
plt.show()
