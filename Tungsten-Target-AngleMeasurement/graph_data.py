import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

with open("batchdata.json","r") as f:
    raw_data = json.load(f)

raw_data = sorted(raw_data, key=lambda x: x["Angle"])

max_counts = []
for measurement in raw_data:
    positron_E = measurement["Raw"]
    
    energy_bins = np.linspace(0, 100, 201)
    positron_E = np.array(list(positron_E))

    counts, bin_edges, _ = plt.hist(positron_E, bins=energy_bins, color='blue', alpha=0.6, label='Positrons')
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    max_counts.append(max(counts))

prev_measurement = None
for measurement in raw_data:
    plt.clf()

    positron_E = measurement["Raw"]
    angle = measurement["Angle"]
    
    energy_bins = np.linspace(0, 100, 201)
    positron_E = np.array(list(positron_E))

    counts, bin_edges, _ = plt.hist(positron_E, bins=energy_bins, color='blue', alpha=0.6, label='Positrons')
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    if not type(prev_measurement) == type(None):
        plt.hist(prev_measurement, bins=energy_bins, color='red', alpha=0.6, label='Positrons')
    
    prev_measurement = positron_E

    #plt.ylim(0,max(max_counts))
    plt.title(f"Energy Distribution at $\\theta={round(angle,1)}^\circ$")
    plt.xlabel('Pz [MeV/c]')
    plt.ylabel('Count')
    plt.savefig(f'im0-{round(angle,1)}.png')