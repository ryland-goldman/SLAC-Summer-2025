#!/usr/bin/env python3
"""
this was entirely generated with chatgpt



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
DATA_FILE = Path("batchdata.csv")          # change this if the CSV lives elsewhere
if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found – check the path.")

df = pd.read_csv(DATA_FILE)

# ---------------------------------------------------------------------------
# 1.  Housekeeping: colours/markers for each beam energy ---------------------
# ---------------------------------------------------------------------------
style_map = {
    100: {'color': 'darkgreen', 'marker': '>'},
    70:  {'color': 'brown',     'marker': '<'},
    40:  {'color': 'magenta',   'marker': 'D'},
    20:  {'color': 'blue',      'marker': 'v'},
    15:  {'color': 'green',     'marker': '^'},
    10:  {'color': 'red',       'marker': 'o'},
    5:   {'color': 'black',     'marker': 's'},
}

# ---------------------------------------------------------------------------
# 2.  Create a single figure with two rows -----------------------------------
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(6.4, 8))
ax1 = fig.add_subplot(2, 1, 1)       # FIRST panel  (your original plot)
ax2 = fig.add_subplot(2, 1, 2)       # SECOND panel (new)

# --- Huge convenience: give ax1 the alias “ax” so the original code is 1-to-1
ax = ax1

optimal_points = []      # list of (energy, l_opt [mm], yield_at_lopt)

# ---------------------------------------------------------------------------
# 3.  *** ORIGINAL CODE – UNTOUCHED ***  -------------------------------------
# ---------------------------------------------------------------------------
for energy in sorted(style_map.keys()):
    group = df[df['Energy'] == energy].sort_values('Thickness')
    if group.empty:
        continue

    style  = style_map[energy]

    # ­­­--- 1.  Compute values and 1 σ Poisson errors  -----------------------
    y      = group['Count'] / 100_000.0
    y_err  = np.sqrt(group['Count']) / 100_000.0

    # ­­­--- 2.  Draw the curve (same as before) ------------------------------
    ax.plot(group['Thickness'], y,
            label=f'{energy}',
            marker=style['marker'],
            color=style['color'],
            linestyle='-')

    # ­­­--- 3.  Add error bars with a transparent line -----------------------
    ax.errorbar(group['Thickness'], y,
                yerr=y_err,
                fmt='none',
                ecolor=style['color'],
                elinewidth=1,
                capsize=2,
                alpha=0.6)

    # Store optimal point (unchanged)
    opt_row = group.loc[group['Count'].idxmax()]
    optimal_points.append((energy,
                           opt_row['Thickness'],
                           opt_row['Count'] / 100_000.0))

# Dashed polyline through optimal points
opt_xy = np.array([[p[1], p[2]] for p in optimal_points])
ax.plot(opt_xy[:, 0], opt_xy[:, 1], 'k--', label=r'Optimal $l_{opt}$')

# Axis styling (unchanged)
ax.set_yscale('log')
ax.set_xlabel('Converter Thickness, $l$ (mm)')
ax.set_ylabel(r'$\gamma_f$ (e$^+$/e$^-$)')

# Pad the x-axis a bit on the right
current_xlim = ax.get_xlim()
ax.set_xlim(current_xlim[0], current_xlim[1] + 0.8)

# --- 4.  Secondary x-axis in radiation lengths ------------------------------
X0_mm = 3.5                               # Ta radiation length (mm)
mm_to_X0 = lambda l_mm: l_mm / X0_mm
X0_to_mm = lambda l_x0: l_x0 * X0_mm

secax1 = ax.secondary_xaxis('top', functions=(mm_to_X0, X0_to_mm))
secax1.set_xlabel(r'Converter Thickness, $l$ ($X_0$)')
secax1.set_xlim(mm_to_X0(ax.get_xlim()[0]), mm_to_X0(ax.get_xlim()[1]))

# Inline labels for the final point of each curve (unchanged)
for energy in sorted(style_map.keys()):
    group = df[df['Energy'] == energy].sort_values('Thickness')
    if group.empty:
        continue
    x_last = group['Thickness'].iloc[-1]
    y_last = group['Count'].iloc[-1] / 100_000.0
    style  = style_map[energy]
    ax.text(x_last + 0.2, y_last,
            f'{energy}' + (' MeV' if energy == 5 else ''),
            color=style['color'],
            va='center', ha='left', fontsize=10, clip_on=False)

ax.legend(fontsize=9)

# ---------------------------------------------------------------------------
# 4.  SECOND PANEL – optimum thickness versus energy with error bars ---------
# ---------------------------------------------------------------------------
# Sort the list so energies are in ascending order
optimal_points.sort(key=lambda t: t[0])

E_vals   = np.array([p[0] for p in optimal_points])
lopt_mm  = np.array([p[1] for p in optimal_points])

# -------- 4a.  Build asymmetric error bars ----------------------------------
lower_err, upper_err = [], []

for E in E_vals:
    g      = df[df['Energy'] == E].sort_values('Thickness')
    idxmax = g['Count'].idxmax()                 # absolute row index of peak
    i      = g.index.get_loc(idxmax)             # positional index (0,1,2,…)
    peak   = g.loc[idxmax, 'Count']
    thresh = peak - np.sqrt(peak)                # 1 σ Poisson drop

    # Walk left
    left_i = i
    while left_i > 0 and g.iloc[left_i]['Count'] >= thresh:
        left_i -= 1
    # Walk right
    right_i = i
    while right_i < len(g)-1 and g.iloc[right_i]['Count'] >= thresh:
        right_i += 1

    lower_err.append(g.iloc[i]['Thickness'] - g.iloc[left_i]['Thickness'])
    upper_err.append(g.iloc[right_i]['Thickness'] - g.iloc[i]['Thickness'])

yerr = np.vstack([lower_err, upper_err])      # shape (2, N) for errorbar

# -------- 4b.  Plot experimental points + error bars ------------------------
ax2.errorbar(E_vals, lopt_mm, yerr=yerr,
             fmt='s', color='black', capsize=3,
             label=r'$l_{\mathrm{opt}}$')

# -------- 4c.  Literature parameterisations ---------------------------------
E_line = np.linspace(5, 110, 500)

ax2.plot(E_line,
         0.6 + 0.64*np.power(np.maximum(E_line - 5, 0), 0.54),
         'k-',  label=r"$0.6 + 0.64(E_e-5)^{0.54}$ [O'Rourke]")

ax2.plot(E_line,
         0.67 + 0.0953 * E_line,
         'k--', label=r'$0.67 + 0.0953\,E_e$  [Ley]')

E_aka = np.linspace(11.1, 110, 500)           # valid for E > 11 MeV
ax2.plot(E_aka,
         X0_mm * np.log(E_aka / 11.0),
         'k-.', label=r'$X_0\ln(E_e/11)$  [Akahane]')

# -------- 4d.  Axes decoration ----------------------------------------------
ax2.set_xlabel(r'Electron Energy, $E_e$ (MeV)')
ax2.set_ylabel(r'Optimum Converter Thickness, $l_{\mathrm{opt}}$ (mm)')
ax2.set_xlim(0, 105)
ax2.set_ylim(0, 12)

secax2 = ax2.secondary_yaxis('right', functions=(mm_to_X0, X0_to_mm))
secax2.set_ylabel(r'Ta Radiation Lengths ($X_0 = 4.1$ mm)')

ax2.legend(fontsize=9)

# ---------------------------------------------------------------------------
# 5.  Render -----------------------------------------------------------------
# ---------------------------------------------------------------------------
plt.tight_layout()
plt.show()
