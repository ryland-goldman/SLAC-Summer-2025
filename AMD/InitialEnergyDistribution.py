import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 18})

# Constants
ME = 0.511  # Positron rest mass in MeV/c^2

# Storage for kinetic energies
ke_list = []

total_n=0

n_electrons = 100000 * 192

# Loop over file indices
for n in range(193):  # From 0 to 192 inclusive
    file_path = f"data-amd/TargetOutFiltered{n}.txt"
    if not os.path.isfile(file_path):
        continue  # Skip if the file doesn't exist

    df = pd.read_csv(
        file_path, skiprows=1, sep=r'\s+',
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    )

    # Filter for positrons (PDG ID = -11)
    positrons = df[df["PDGid"] == "-11"]

    # Compute total energy: E = sqrt(p^2 + m^2)
    p2 = positrons["Px"]**2 + positrons["Py"]**2 + positrons["Pz"]**2
    total_energy = np.sqrt(p2 + ME**2)

    # Kinetic energy = total energy - rest mass
    ke = total_energy - ME
    ke_list.extend(ke)
    total_n += df.shape[0]

# Plot histogram
plt.figure(figsize=(10, 6))
ke_list = np.array(ke_list)
counts, bins, _ = plt.hist(ke_list, bins=200, range=(0, 30), color='steelblue', edgecolor='black')
plt.clf()
plt.bar((bins[:-1] + bins[1:]) / 2,
        counts / (n_electrons * (bins[1] - bins[0])),
        width=(bins[1] - bins[0]),
        color='steelblue', edgecolor='black')

print(np.sum(counts/ (n_electrons)))

plt.xlabel("Kinetic Energy (MeV)")
plt.ylabel("e$^+$/e$^-$/MeV")
plt.grid(True)
plt.tight_layout()
plt.show()