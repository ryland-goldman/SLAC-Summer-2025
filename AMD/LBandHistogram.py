import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


df0 = pd.read_csv(f"TargetOutFiltered.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df1 = pd.read_csv(f"LBandOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df2 = pd.read_csv(f"AMDOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

file_names = [f for f in os.listdir("data-amd") if f.startswith("LBandOut")]
file_names = [f"data-amd/{f}" for f in file_names]
'''dfs_out = {
    name: pd.read_csv(
        name, skiprows=1, delim_whitespace=True,
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    ) for name in file_names
}
'''
df0 = df0[df0["PDGid"]=="-11"]
df1 = df1[df1["PDGid"]=="-11"]
df2 = df2[df2["PDGid"]=="-11"]

print(np.mean(df1["t"]))

m = 0.511  # MeV/c^2
df0["E"] = np.sqrt(df0["Px"]**2 + df0["Py"]**2 + df0["Pz"]**2 + m**2) - m
df1["E"] = np.sqrt(df1["Px"]**2 + df1["Py"]**2 + df1["Pz"]**2 + m**2) - m
df2["E"] = np.sqrt(df2["Px"]**2 + df2["Py"]**2 + df2["Pz"]**2 + m**2) - m

'''for name, df in dfs_out.items():
    df = df[df["PDGid"] == "-11"]
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    dfs_out[name] = df
'''
count1 = ((df1["E"] >= 0) & (df1["E"] <= 0.2)).sum()
print(f"Number of positrons in df1 with 0 <= E <= 0.2 MeV: {count1}")
count2 = ((df2["E"] >= 0) & (df2["E"] <= 0.2)).sum()
print(f"Number of positrons in df2 with 0 <= E <= 0.2 MeV: {count2}")


count1 = ((df1["E"] >= 0) & (df1["E"] <= 0.5)).sum()
print(f"Number of positrons in df1 with 0 <= E <= 0.5 MeV: {count1}")
count2 = ((df2["E"] >= 0) & (df2["E"] <= 0.5)).sum()
print(f"Number of positrons in df2 with 0 <= E <= 0.5 MeV: {count2}")


max_energy = max(df0["E"].max(), df1["E"].max(), df2["E"].max())
bins = np.arange(0.0, max_energy + 1.0, 1.0)  # 1 MeV bins
# Plot histogram for df1["E"]
fig, (ax_hist, ax_hist2) = plt.subplots(nrows=2, figsize=(8, 10))

'''for name, df in dfs_out.items():
    plt.hist(df["E"], bins=bins, histtype='step', linewidth=1.5, label=f'{name}')
'''

df0_initiale = 100000*192
df1_initiale = 100000*192
df2_initiale = 100000*192

ax_hist.hist(
    df0["E"],
    bins=bins,
    color='steelblue',
    edgecolor='black',
    alpha=0.5,
    label='After Target',
    weights=np.ones_like(df0["E"], dtype=float) / df0_initiale
)

# Plot histogram for df2["E"]
ax_hist.hist(
    df2["E"],
    bins=bins,
    color='darkgreen',
    edgecolor='black',
    alpha=0.5,
    label='After AMD',
    weights=np.ones_like(df2["E"], dtype=float) / df2_initiale
)

ax_hist.hist(
    df1["E"],
    bins=bins,
    color='darkred',
    edgecolor='black',
    alpha=0.5,
    label='After RF Cavity',
    weights=np.ones_like(df1["E"], dtype=float) / df1_initiale
)

handles, labels = ax_hist.get_legend_handles_labels()
order = [0, 2, 1]  # Target, AMD, Cavity
ax_hist.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=12)
ax_hist.set_ylabel("e+ per e- per 1 MeV", fontsize=14)
ax_hist.set_xlabel("Kinetic Energy (MeV)", fontsize=14)

# --- Add second histogram with 100 keV bins ---
bins_100keV = np.arange(0.0, 2.0 + 0.1, 0.1)

ax_hist2.hist(
    df1["E"],
    bins=bins_100keV,
    color='darkred',
    edgecolor='black',
    alpha=0.5,
    label='After RF Cavity',
    weights=np.ones_like(df1["E"], dtype=float) / df1_initiale
)

ax_hist2.hist(
    df0["E"],
    bins=bins_100keV,
    color='steelblue',
    edgecolor='black',
    alpha=0.5,
    label='After Target',
    weights=np.ones_like(df0["E"], dtype=float) / df0_initiale
)

ax_hist2.hist(
    df2["E"],
    bins=bins_100keV,
    color='darkgreen',
    edgecolor='black',
    alpha=0.5,
    label='After AMD',
    weights=np.ones_like(df2["E"], dtype=float) / df2_initiale
)

handles2, labels2 = ax_hist2.get_legend_handles_labels()
order2 = [1, 2, 0]
ax_hist2.legend([handles2[idx] for idx in order2], [labels2[idx] for idx in order2], loc='lower right', fontsize=12)
ax_hist2.set_ylabel("e+ per e- per 100 keV", fontsize=14)
ax_hist2.set_xlabel("Kinetic Energy (MeV)", fontsize=14)

plt.tight_layout()
plt.savefig("energy_histograms.png", dpi=300)
plt.close()

import sys
sys.exit()

# Heatmap of df1["t"] vs df1["E"]
plt.figure()
plt.hist2d(df1["t"], df1["E"], bins=[200, 200], range=[[1.6, 1.8], [df1["E"].min(), 20.0]], cmap='viridis')
plt.xlabel('t (ns)')
plt.ylabel('KE (MeV)')
plt.colorbar(label='Count')
plt.grid(True)
plt.savefig("time_vs_energy_heatmap.png", dpi=300)
plt.close()