import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
m_e = 0.511  # electron rest mass in MeV/c^2

def name(j):
    if j == 1: return "Initial"
    if j == 6: return "Final"
    return f"Between Cavity {j-1} and {j}"

def load_dataset(i, j2):
    j = j2 -1
    filename = f"data-amd/LBandOut_RF{j}_{i}.txt"
    if j == 5: filename = f"data-amd/LBandOut{i}.txt"
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

fig, axes = plt.subplots(6, 5, figsize=(24, 20))

for j in range(1, 7):
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

    # Scatter: E vs t
    axes[row, 3].scatter(df["t"], df["energy"] - m_e, s=1, alpha=0.5)
    axes[row, 3].set_title("Time of Detector Hit")
    axes[row, 3].set_xlabel("t (ns)")
    axes[row, 3].set_ylabel("KE (MeV)")
    axes[row, 3].set_yscale('log')

    axes[row, 4].hist(df["t"], bins=np.arange(0,10,0.05))
    axes[row, 4].set_title("Time of Detector Hit")
    axes[row, 4].set_xlabel("t (ns)")
    axes[row, 4].set_yscale('log')
    axes[row, 4].set_ylabel("Count")

plt.tight_layout()
plt.savefig("LBandSnapshots.png", dpi=300)