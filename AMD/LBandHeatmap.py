import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df1 = pd.read_csv(f"LBandIn.txt", skiprows=1, sep=r'\s+', dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

file_names = os.listdir("data-lband")
file_names = [f"data-lband/{f}" for f in file_names]
file_names.sort()
file_names = file_names[::5]
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

for name, df in dfs_out.items():
    df = df[df["PDGid"] == "-11"].copy()
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    import re
    match = re.search(r"out-([-+]?\d*\.\d+|\d+)", name)
    x_val = float(match.group(1)) if match else 0.0
    time_offsets.append(x_val)

    # Bin the energy values manually and append to flat arrays
    inds = np.digitize(df["E"], bins=energy_bins)
    for ind in inds:
        if 0 < ind < len(energy_bins):
            flat_times.append(x_val)
            flat_energies.append(energy_bins[ind - 1])

# Create figure and set frequency
plt.figure(figsize=(10, 6))
frequency = 1.428e9  # Hz
# Use discrete time offsets based on unique x values
unique_offsets = sorted(set(flat_times))
offset_values = flat_times  # use actual time offsets in ns

time_bin_width = unique_offsets[1] - unique_offsets[0] if len(unique_offsets) > 1 else 0.1
time_bins = np.append(unique_offsets, unique_offsets[-1] + time_bin_width)
plt.hist2d(
    offset_values,
    flat_energies,
    bins=[time_bins, energy_bins],
    cmap='viridis'
)
plt.xlabel("Time Offset (ns)")
plt.ylabel("Energy (MeV)")
plt.colorbar(label="Counts")
plt.title("Energy Spectrum of Positrons Over Time Offsets")

# Show only a subset of x-tick labels to prevent overlap
step = max(1, len(unique_offsets) // 10)
shown_ticks = unique_offsets[::step]
plt.xticks(ticks=shown_ticks, labels=[f"{x:.2f}" for x in shown_ticks], rotation=45)

t0 = min(unique_offsets)

def time_to_phase(t_ns):
    t_adj = np.array(t_ns) - t0
    time_s = t_adj * 1e-9
    phase = (time_s * frequency * 360) % 360
    return phase

def phase_to_time(phase_deg):
    t_adj_s = (np.array(phase_deg) / 360) / frequency
    return (t_adj_s + t0 * 1e-9) * 1e9

ax = plt.gca()
secax = ax.secondary_xaxis('top', functions=(time_to_phase, phase_to_time))
secax.set_xlabel("Phase Offset (degrees)")
secax.set_xticks(np.linspace(0, 360, num=9))  # Set phase ticks from 0 to 360 degrees
secax.tick_params(axis='x', rotation=0)

plt.show()

# Count positrons below various energy thresholds at each time offset
import re
thresholds = [5, 3, 2]  # MeV
counts = {thr: [] for thr in thresholds}
for name, df in dfs_out.items():
    df = df[df["PDGid"] == "-11"].copy()
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    match = re.search(r"out-([-+]?\d*\.\d+|\d+)", name)
    x_val = float(match.group(1)) if match else 0.0
    for thr in thresholds:
        counts[thr].append((x_val, (df["E"] < thr).sum()))

# Sort counts by time offset
sorted_counts = {thr: sorted(counts[thr]) for thr in thresholds}
times = [x for x, _ in sorted_counts[thresholds[0]]]

plt.figure(figsize=(10, 6))
for thr in thresholds:
    values = [count for _, count in sorted_counts[thr]]
    plt.plot(times, values, marker='o', label=f"E < {thr} MeV")

# Add vertical lines at the maximum point for each threshold
for thr in thresholds:
    values = [count for _, count in sorted_counts[thr]]
    max_index = np.argmax(values)
    max_time = times[max_index]
    plt.axvline(x=max_time, linestyle='--', label=f"Max E<{thr} MeV", alpha=0.7)
    print(thr, max_time)

plt.xlabel("Time Offset (ns)")
plt.ylabel("Number of Positrons")
plt.title("Counts of Positrons Below Energy Thresholds Over Time")
plt.legend()
plt.grid(True)
plt.show()