import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_stopping_distribution(directory, thickness_um, line=False):
    files = [f"{directory}/{f}" for f in os.listdir(directory) if f.startswith("Out") and f.endswith(".dat") and 'r' not in f]
    end_z = []
    all_initial_angle = []
    for file in files:
        try:
            df = pd.read_parquet(file)
        except Exception:
            continue
        df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
        all_initial_angle += list(df["initialAngle"])
        df = df[(df["endz"] > 0.9) & (df["endz"] < 1.1)]
        end_z += list(df["endz"])
    if not all_initial_angle:
        return (np.array([]), np.array([])) if not line else (np.array([]), np.array([]), np.array([]))
    end_z = np.array(end_z)
    end_z = (end_z - 1 + thickness_um / 2000) * 1000  # Convert to µm, centered on thickness
    bins = np.arange(0, thickness_um + 0.5, 0.5)
    counts, _ = np.histogram(end_z, bins=bins)
    counts = np.array(counts) / len(all_initial_angle) * 1e4 * 1/0.5
    if line:
        return bins[:-1], counts, end_z
    else:
        return bins[:-1], counts

# Load and plot
plt.figure(figsize=(8, 6))

for label, thickness in [("50 µm", 50), ("40 µm", 40), ("30 µm", 30), ("20 µm", 20), ("10 µm", 10)]:
    if os.path.isdir(f"Mod{thickness}um"):
        bins, counts = load_stopping_distribution(f"Mod{thickness}um", thickness)
        if len(counts) > 0:
            plt.bar(bins, counts, width=0.5, align='edge', label=label, alpha=0.6, edgecolor='black')
        else:
            print(f"No valid data found in Mod{thickness}um.")
    else:
        print(f"Directory Mod{thickness}um not found.")

# Add electric field version as line plot
if os.path.isdir("ModE"):
    bins, counts = load_stopping_distribution("ModE", 50)
    if len(counts) > 0:
        plt.plot(bins, counts, label="50 µm + E-field", linewidth=2)
    else:
        print("No valid data found in ModE.")
else:
    print("Directory ModE not found.")

plt.xlabel("Penetration Depth (µm)")
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")
plt.title("Stopping Distributions for Different Foil Thicknesses")
plt.suptitle("Using Output from L-Band Cavity", fontsize=10, y=0.95)
plt.legend()
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

'''z = np.linspace(0,50,200)
makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) )
prob = makhovian(z, 1.9, 45)
count = prob * 1e4
plt.plot(z, count)'''

plt.show()