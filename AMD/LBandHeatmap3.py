import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df1 = pd.read_csv(f"LBandIn.txt", skiprows=1, sep=r'\s+', dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

file_names = os.listdir("data-lband3")
file_names = [f"data-lband3/{f}" for f in file_names]
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

m = 0.511
energy_bins = np.linspace(0, 10, 100)  # Define energy bins from 0 to 10 MeV
time_offsets = []

flat_times = []
flat_energies = []
flat_counts = []

from scipy.interpolate import interp1d
fraction_stopped = np.array([
    0.47892977, 0.54151724, 0.53431588, 0.49657534, 0.45890411, 0.39489466,
    0.34305835, 0.28766383, 0.24699651, 0.20913706, 0.18261298, 0.1614937,
    0.14453125, 0.12294028, 0.10805301, 0.09861079, 0.09459773, 0.08315863,
    0.07862578, 0.07565982, 0.06838419, 0.0639747,  0.06471292, 0.06050916,
    0.05827535, 0.05185466, 0.05394137, 0.04890323, 0.0474705,  0.04380054,
    0.04226519, 0.04476574, 0.04326855, 0.03821008, 0.03665386, 0.03426092,
    0.03600374, 0.03678606, 0.03084495, 0.03467543, 0.03548117, 0.02911283,
    0.02838505, 0.02758495, 0.02519597, 0.03175203, 0.02596381, 0.02656622,
    0.02440993
])
energy_bins = np.linspace(0, 10, len(fraction_stopped))

stopping_fraction_interp = interp1d(
    energy_bins, fraction_stopped, bounds_error=False,
    fill_value=(fraction_stopped[0], fraction_stopped[-1])
)

def estimate_particles_stopped(energies):
    energies = np.array(energies)
    fractions = stopping_fraction_interp(energies)
    return np.sum(fractions)


for name, df in dfs_out.items():
    df = df[df["PDGid"] == "-11"].copy()
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    import re
    match = re.search(r"out-([-+]?\d*\.\d+|\d+)", name)
    e_val = float(match.group(1)) if match else 0.0
    t_val = 0.261
    count = estimate_particles_stopped(df["E"])
    flat_times.append(t_val)
    flat_energies.append(e_val)
    flat_counts.append(count)

from matplotlib import cm

plt.figure(figsize=(10, 6))
print(flat_counts)
plt.scatter(flat_energies, flat_counts)
plt.xlabel("Electric Field (MV/m)")
plt.ylabel("Counts of E < 2 MeV")
plt.title("Count vs. Gradient (t=0.261 ns)")
plt.show()

# The following section generating line plots for different thresholds has been removed as per instructions.