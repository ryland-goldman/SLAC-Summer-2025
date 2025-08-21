import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df1 = pd.read_csv(f"LBandIn.txt", skiprows=1, sep=r'\s+', dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

file_names = os.listdir("data-lband2")
file_names = [f"data-lband2/{f}" for f in file_names]
dfs_out = {
    name: pd.read_csv(
        name, skiprows=1, sep=r'\s+',
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    ) for name in file_names
}

from scipy.interpolate import interp1d
fraction_stopped = np.array([7.86885246e-01, 7.36220472e-01, 8.03370787e-01, 8.28901734e-01,
       8.48021583e-01, 8.34603659e-01, 8.51183064e-01, 8.00569801e-01,
       7.01644101e-01, 5.60573165e-01, 4.39335394e-01, 3.13344887e-01,
       2.40242057e-01, 1.54988789e-01, 1.20009678e-01, 8.23844608e-02,
       5.49848943e-02, 3.73443983e-02, 2.94169456e-02, 2.14738897e-02,
       1.47329650e-02, 1.06184531e-02, 9.75265018e-03, 5.85807482e-03,
       6.31552356e-03, 6.48397357e-03, 4.23131171e-03, 6.14824092e-03,
       3.98996808e-03, 3.10625536e-03, 2.47882669e-03, 2.44274809e-03,
       1.40675241e-03, 2.04878049e-03, 1.82446706e-03, 2.16063880e-03,
       2.21811460e-03, 6.25335001e-04, 7.10164225e-04, 7.08027259e-04])
energy_bins = np.linspace(0, 2, len(fraction_stopped))

stopping_fraction_interp = interp1d(
    energy_bins, fraction_stopped, bounds_error=False,
    fill_value=(fraction_stopped[0], fraction_stopped[-1])
)

def estimate_particles_stopped(energies):
    energies = np.array(energies)
    fractions = stopping_fraction_interp(energies)
    return np.sum(fractions)


m = 0.511
energy_bins = np.linspace(0, 10, 100)  # Define energy bins from 0 to 10 MeV
time_offsets = []

flat_times = []
flat_energies = []
flat_counts = []

for name, df in dfs_out.items():
    df = df[df["PDGid"] == "-11"].copy()
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    import re
    match = re.search(r"out-([-+]?\d*\.\d+|\d+)-([-+]?\d*\.\d+|\d+)", name)
    t_val = float(match.group(1)) if match else 0.0
    e_val = float(match.group(2)) if match else 0.0
    count = estimate_particles_stopped(df["E"])
    flat_times.append(t_val)
    flat_energies.append(e_val)
    flat_counts.append(count)

from matplotlib import cm

plt.figure(figsize=(10, 6))
# Create a 2D histogram grid
# Create a DataFrame of the discrete points
import pandas as pd

df_heat = pd.DataFrame({
    "Time": flat_times,
    "Field": flat_energies,
    "Count": flat_counts
})

# Pivot to form a 2D grid (rows=Field, cols=Time)
pivot = df_heat.pivot_table(index="Field", columns="Time", values="Count", fill_value=0)

# Sort the axes if needed
pivot = pivot.sort_index(axis=0).sort_index(axis=1)

from scipy.interpolate import RegularGridInterpolator

# Create interpolation function using RegularGridInterpolator
x = pivot.columns.values  # Time
y = pivot.index.values    # Field
z = pivot.values          # Counts

# Define finer grid
xnew = np.linspace(x.min(), x.max(), 200)
ynew = np.linspace(y.min(), y.max(), 200)

f_interp = RegularGridInterpolator((y, x), z, method='linear')

# Evaluate on new grid
xv, yv = np.meshgrid(xnew, ynew)
points = np.array([yv.ravel(), xv.ravel()]).T
znew = f_interp(points).reshape(len(ynew), len(xnew))

# Plot smoothed heatmap
plt.figure(figsize=(10, 6))
plt.imshow(znew, aspect='auto', origin='lower',
           extent=[xnew.min(), xnew.max(), ynew.min(), ynew.max()],
           cmap='viridis')
plt.xlabel("Time Offset (ns)")
plt.ylabel("Electric Field Gradient (MV/m)")
plt.colorbar(label="Number Stopped in 50µm Moderator")
#plt.title("2D Heatmap of Positron Counts Below 2 MeV")
plt.show()

# The following section generating line plots for different thresholds has been removed as per instructions.