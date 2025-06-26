import scipy
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.interpolate
import os
import pickle as pkl

def bounded_normal(mean,std,size):
    samples=np.random.normal(mean,std,size)
    mask=(samples<=0)|(samples>=100)
    while np.any(mask):
        samples[mask]=np.random.normal(mean,std,mask.sum())
        mask=(samples<=0)|(samples>=100)
    return samples

input_distribution = bounded_normal(50,20,1000000) # generate uniformly random distribution of 10k particles between 1-100 MeV

num_events = 1000000

energy_bins = np.linspace(0, 100, 201)
bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])

try:
    assert "energy_interpolation.pkl" in os.listdir()
    with open("energy_interpolation.pkl","rb") as f: c,x,axis,extrap = pkl.load(f)
    dist_at_E = scipy.interpolate.PPoly(c,x,axis=axis,extrapolate=extrap)

except Exception as e:
    if not type(e)==AssertionError: print(f"Encountered exception reading energy_interpolation.pkl: {e}")
    if type(e)==AssertionError: print("Could not find energy_interpolation.pkl. Generating...")
    df = pd.read_csv("batchdata.csv")
    with open("batchdata.json","r") as f: raw_data = json.load(f)

    energies = []
    bins = []

    for measurement in raw_data:
        incident_E = measurement["Energy"]
        positron_E = measurement["Raw"]
        counts, bin_edges, _ = plt.hist(positron_E, bins=energy_bins, color='blue', alpha=0.6, label='Positrons')

        energies.append(incident_E)
        bins.append(counts)

    dist_at_E = scipy.interpolate.CubicSpline(energies, bins, bc_type="natural")
    with open("energy_interpolation.pkl","wb") as f: pkl.dump((dist_at_E.c,dist_at_E.x,dist_at_E.axis,dist_at_E.extrapolate), f, protocol=pkl.HIGHEST_PROTOCOL)

output_dist = np.array(dist_at_E(input_distribution)).sum(axis=0) / num_events
plt.close()

fig,ax = plt.subplots(1,2,figsize=(10,7))
ax[0].hist(input_distribution, bins=energy_bins)
ax[0].set_title(f"Input Energy Distribution (n={len(input_distribution)})")
ax[0].set_xlabel("E [MeV]")
ax[0].set_ylabel("Count")
fig.suptitle(f"Energy Interpolation Through 7mm Tungsten")

g4bl_data = np.array([1.779e+03, 4.122e+03, 4.611e+03, 4.567e+03, 4.410e+03, 4.018e+03,
       3.732e+03, 3.428e+03, 3.302e+03, 2.911e+03, 2.721e+03, 2.538e+03,
       2.330e+03, 2.230e+03, 1.960e+03, 1.864e+03, 1.794e+03, 1.631e+03,
       1.557e+03, 1.426e+03, 1.364e+03, 1.191e+03, 1.139e+03, 1.000e+03,
       1.048e+03, 9.440e+02, 8.700e+02, 8.420e+02, 7.130e+02, 7.360e+02,
       6.680e+02, 6.260e+02, 6.210e+02, 5.420e+02, 5.630e+02, 4.990e+02,
       4.750e+02, 4.730e+02, 4.210e+02, 3.940e+02, 3.720e+02, 3.590e+02,
       3.290e+02, 3.210e+02, 2.810e+02, 3.030e+02, 2.810e+02, 2.410e+02,
       2.380e+02, 2.180e+02, 2.140e+02, 2.100e+02, 2.120e+02, 1.910e+02,
       1.680e+02, 1.740e+02, 1.430e+02, 1.300e+02, 1.250e+02, 1.360e+02,
       1.300e+02, 1.150e+02, 1.010e+02, 9.100e+01, 1.040e+02, 9.900e+01,
       9.000e+01, 7.600e+01, 7.000e+01, 7.600e+01, 7.900e+01, 5.600e+01,
       6.700e+01, 4.800e+01, 6.000e+01, 5.000e+01, 5.100e+01, 2.200e+01,
       4.700e+01, 4.800e+01, 4.100e+01, 4.100e+01, 2.600e+01, 3.400e+01,
       2.700e+01, 2.700e+01, 3.100e+01, 2.100e+01, 2.700e+01, 1.600e+01,
       2.700e+01, 1.300e+01, 2.000e+01, 1.600e+01, 2.700e+01, 1.100e+01,
       1.100e+01, 1.700e+01, 1.400e+01, 1.400e+01, 7.000e+00, 9.000e+00,
       1.100e+01, 1.100e+01, 7.000e+00, 7.000e+00, 1.300e+01, 8.000e+00,
       7.000e+00, 9.000e+00, 9.000e+00, 6.000e+00, 5.000e+00, 3.000e+00,
       4.000e+00, 3.000e+00, 5.000e+00, 4.000e+00, 3.000e+00, 2.000e+00,
       2.000e+00, 2.000e+00, 6.000e+00, 3.000e+00, 5.000e+00, 2.000e+00,
       3.000e+00, 3.000e+00, 2.000e+00, 1.000e+00, 0.000e+00, 0.000e+00,
       2.000e+00, 1.000e+00, 1.000e+00, 1.000e+00, 0.000e+00, 3.000e+00,
       2.000e+00, 1.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 2.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00])

ax[1].bar(bin_centers, g4bl_data, align='edge', width=np.diff(bin_centers).mean(),color='r',alpha=0.6,label="g4beamline")
ax[1].bar(bin_centers, output_dist, align='edge', width=np.diff(bin_centers).mean(),color='b',alpha=0.6,label="CubicSpline")
ax[1].set_title(f"Output Energy Distribution (n={round(output_dist.sum(),1)})")
ax[1].set_xlabel('Pz [MeV/c]')
ax[1].set_ylabel('Count')
ax[1].legend()

plt.show()