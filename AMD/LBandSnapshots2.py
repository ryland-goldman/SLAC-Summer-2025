import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
m_e = 0.511  # electron rest mass in MeV/c^2

def name(j):
    if j == 1: return "Initial"
    if j == 6: return "Final"
    return f"Between Cavity {j-1} and {j}"

def load_dataset(i, j2):
    j = j2 -1
    filename = f"data-amd/LBandOut_RF{j}_{i}.txt"
    if j == 5: filename = f"data-amd/LBandOut_Edited_{i}.txt"
    if j == 0: filename = f"data-amd/AMDOut{i}.txt"
    df = pd.read_csv(
        filename,
        skiprows=1,
        sep=r'\s+',
        dtype={
            "x": np.float32, "y": np.float32, "z": np.float32,
            "Px": np.float32, "Py": np.float32, "Pz": np.float32, "t": np.float32,
            "PDGid": str, "EventID": np.uint32, "TrackID": np.uint16
        },
        usecols=["x", "y", "z", "Px", "Py", "Pz", "t", "PDGid", "EventID", "TrackID"],
        on_bad_lines="skip",
        names="x y z Px Py Pz t PDGid EventID TrackID ParentID Weight".split(" "),
        comment="#"
    )
    df = df[df["PDGid"]=="-11"]
    return df

fig, axes = plt.subplots(6, 3, figsize=(24, 20))

'''for j in range(1, 7):
    frames = [load_dataset(i, j) for i in range(1, 41)]
    df = pd.concat(frames, ignore_index=True)

    # Derived quantities
    df["p_total"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2)
    df["energy"] = np.sqrt(df["p_total"]**2 + m_e**2)
    df["theta_mrad"] = 1000 * np.arctan2(np.sqrt(df["Px"]**2 + df["Py"]**2), df["Pz"])

    row = j - 1

    # Histogram: total energy
    axes[row, 0].hist(df["energy"]-m_e, bins=np.arange(0, 10.05, 0.05))
    axes[row, 0].set_title(f"Total Energy | {name(j)}")
    axes[row, 0].set_xlabel("E (MeV)")
    axes[row, 0].set_ylabel("Counts")
    axes[row, 0].set_xlim([0, 10])

    # Histogram: angular divergence
    axes[row, 1].hist(df["theta_mrad"], bins=np.arange(0, 401, 1))
    axes[row, 1].set_title("Angular Divergence")
    axes[row, 1].set_xlabel("θ (mrad)")
    axes[row, 1].set_xlim([0, 400])

    # Scatter: x‑y distribution
    axes[row, 2].scatter(df["x"], df["y"], s=1, alpha=0.5)
    axes[row, 2].set_title("x‑y Distribution")
    axes[row, 2].set_xlabel("x")
    axes[row, 2].set_ylabel("y")
    axes[row, 2].set_xlim([-100, 100])
    axes[row, 2].set_ylim([-100, 100])
    axes[row, 2].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("LBandSnapshots-1.png", dpi=300)
'''
all_frames = [load_dataset(i, j) for j in [6] for i in range(1, 41)]
df_all = pd.concat(all_frames, ignore_index=True)
df_all["p_total"] = np.sqrt(df_all["Px"]**2 + df_all["Py"]**2 + df_all["Pz"]**2)
df_all["energy"] = np.sqrt(df_all["p_total"]**2 + m_e**2)
df_all["KE"] = df_all["energy"] - m_e

plt.figure(figsize=(10, 8))
h = plt.hist2d(
    df_all["t"]-min(df_all["t"]), df_all["KE"],
    bins=[np.linspace(0, 10, 200), np.logspace(-3, 2, 200)],
    cmap="viridis",
    norm=mcolors.LogNorm()
)
plt.colorbar(h[3], label="Counts")
plt.xlabel("Time (ns)")
plt.ylabel("Kinetic Energy (MeV)")
plt.yscale("log")
#plt.title("Time vs Kinetic Energy")
plt.axhline(y=0.2, color="red", linestyle="--", linewidth=1, label="KE = 0.2 MeV")
#plt.legend()
 
# Inset histogram of time for particles under 0.2 MeV and FWHM calculation
mask = df_all["KE"] < 0.2
# Zero time to the first arrival as in the main plot
times = df_all["t"] - df_all["t"].min()
times_sel = times[mask]
times_sel_ps = times_sel * 1000.0

# Build histogram
time_bins = np.linspace(0, 10, 200)
counts, edges = np.histogram(times_sel, bins=time_bins)
centers = 0.5 * (edges[:-1] + edges[1:])

# Compute FWHM
if counts.size > 0 and counts.max() > 0:
    half_max = counts.max() / 2.0
    above = counts >= half_max
    idx = np.where(above)[0]
    fwhm = centers[idx[-1]] - centers[idx[0]]
else:
    fwhm = np.nan

print(f"Time FWHM for KE<0.2 MeV: {fwhm:.4f} ns")

# Add inset plot in the upper-right of the current axes
ax = plt.gca()
axins = inset_axes(ax, width="35%", height="30%", loc='upper right', borderpad=3)
axins.hist(times_sel_ps, bins=np.linspace(0, 800, 100))
axins.set_xlim(0, 800)
axins.set_title("Time Histogram (Filtered for KE<0.2 MeV)", fontsize=10)
axins.set_xlabel("ps", fontsize=8)
axins.set_ylabel("Counts", fontsize=8)
for label in axins.get_xticklabels() + axins.get_yticklabels():
    label.set_fontsize(8)

fwhm *= 1000
# Optionally shade the FWHM region if defined
if counts.size > 0 and counts.max() > 0:
    axins.axvspan(centers[idx[0]], centers[idx[-1]], alpha=0.2)
    axins.text(0.98, 0.95, f"FWHM = {fwhm:.3f} ps", transform=axins.transAxes,
               ha='right', va='top', fontsize=8)

plt.tight_layout()
plt.savefig("LBandSnapshots.png", dpi=300)