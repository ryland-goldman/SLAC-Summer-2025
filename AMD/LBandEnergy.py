import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

points = []

for i in range(1,14):
    df_in = pd.read_csv(
        f"data-amd/AMDOut_Edited_{i}.txt", skiprows=1, delim_whitespace=True,
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    )
    df_out = pd.read_csv(
        f"data-amd/LBandOut_{i}.txt", skiprows=1, delim_whitespace=True,
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    )

    df_in = df_in[df_in["PDGid"]=="-11"]
    df_out = df_out[df_out["PDGid"]=="-11"]

    m = 0.511  # MeV/c^2
    df_in["E"] = np.sqrt(df_in["Px"]**2 + df_in["Py"]**2 + df_in["Pz"]**2 + m**2) - m
    df_out["E"] = np.sqrt(df_out["Px"]**2 + df_out["Py"]**2 + df_out["Pz"]**2 + m**2) - m

    for (event_id, track_id), group in df_out.groupby(["EventID", "TrackID"]):
        initial_E_row = df_in[(df_in["EventID"] == event_id) & (df_in["TrackID"] == track_id)]
        if not initial_E_row.empty:
            initial_E = initial_E_row.iloc[0].E
            final_E = group.iloc[0].E
            points.append([initial_E, final_E])

# Ensure 2D shape even if `points` is empty to avoid indexing errors
points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)

plt.figure(figsize=(8, 6))
plt.hist2d(points_array[:, 0], points_array[:, 1], bins=200, range=[[0, 5], [0, 3]], cmap='viridis')
plt.plot([0, 5], [0, 5], 'r--', linewidth=1)
plt.xlabel("Initial Energy (MeV)")
plt.ylabel("Final Energy (MeV)")
#plt.title("2D Histogram of Initial vs Final Electron Energy")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()
