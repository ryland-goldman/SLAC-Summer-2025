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
            print(f"Err {file}")
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

for radius in [50,25,20,15,10,5]:
    directory = f"Mod{radius}mm"
    if os.path.isdir(directory):
        bins, counts = load_stopping_distribution(directory, 50)
        if len(counts) > 0:
            plt.bar(bins, counts, width=0.5, align='edge', label=f"{radius} mm radius", alpha=0.6, edgecolor='black')
        else:
            print(f"No valid data found in {directory}.")
    else:
        print(f"Directory {directory} not found.")

plt.xlabel("Penetration Depth (µm)")
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")
plt.title("Stopping Distributions for Different Moderator Radii (50 µm thickness)")
plt.suptitle("Using Output from L-Band Cavity", fontsize=10, y=0.95)
plt.legend()
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.show()