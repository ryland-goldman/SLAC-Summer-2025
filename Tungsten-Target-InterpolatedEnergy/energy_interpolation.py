import scipy
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.interpolate

def bounded_normal(mean,std,size):
    samples=np.random.normal(mean,std,size)
    mask=(samples<=0)|(samples>=100)
    while np.any(mask):
        samples[mask]=np.random.normal(mean,std,mask.sum())
        mask=(samples<=0)|(samples>=100)
    return samples

input_distribution = bounded_normal(50,20,100000) # generate uniformly random distribution of 10k particles between 1-100 MeV

num_events = 1000000

df = pd.read_csv("batchdata.csv")
with open("batchdata.json","r") as f: raw_data = json.load(f)

energy_bins = np.linspace(0, 100, 201)
bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])
energies = []
bins = []

for measurement in raw_data:
    incident_E = measurement["Energy"]
    positron_E = measurement["Raw"]
    counts, bin_edges, _ = plt.hist(positron_E, bins=energy_bins, color='blue', alpha=0.6, label='Positrons')

    energies.append(incident_E)
    bins.append(counts)

dist_at_E = scipy.interpolate.CubicSpline(energies, bins, bc_type="natural")
output_dist = np.array(dist_at_E(input_distribution)).sum(axis=0) / num_events
plt.close()

fig,ax = plt.subplots(1,2,figsize=(10,7))
ax[1].bar(bin_centers, output_dist, align='edge', width=np.diff(bin_centers).mean())
ax[1].set_title(f"Output Energy Distribution (n={round(output_dist.sum(),1)})")
ax[1].set_xlabel('Pz [MeV/c]')
ax[1].set_ylabel('Count')
ax[0].hist(input_distribution, bins=energy_bins)
ax[0].set_title(f"Input Energy Distribution (n={len(input_distribution)})")
ax[0].set_xlabel("E [MeV]")
ax[0].set_ylabel("Count")
fig.suptitle(f"Energy Interpolation Through 7mm Tungsten")
plt.show()