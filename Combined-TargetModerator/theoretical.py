import os
import pandas as pd
import scipy
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# find the positron energies from the interpolation script
with open("../Tungsten-Target-InterpolatedEnergy/energy_interpolation.pkl","rb") as f: c,x,axis,extrap = pkl.load(f)
dist_at_E = scipy.interpolate.PPoly(c,x,axis=axis,extrapolate=extrap)

num_events = int(1e4)

input_energy = np.ones(num_events) * 100
positron_inital_energy = dist_at_E(input_energy).sum(axis=0) / num_events
energy_bins = np.linspace(0.5,100,200)


# interpolate the z0 values from https://doi.org/10.1063/1.5097607
ke = np.array([10,100,300,500,800,1000,5000,10000],dtype=np.float32) / 1000.0
z0 = np.array([0.083,3.22,16.5,35.24,66.27,88.8,619.2,1222.1])

z0_of_e = scipy.interpolate.CubicSpline(ke,z0,bc_type='natural')

m = 1.828


# calculate the number of positrons that stop somewhere, based on their energies
z = np.linspace(0,50,1000)
stopped = np.zeros(len(z))
for i in range(len(energy_bins)):
    num_pos = positron_inital_energy[i]
    energy = energy_bins[i]
    
    z0 = z0_of_e(energy)

    p = (m * np.pow(z, m-1) / np.pow(z0,m)) * np.exp( -np.pow(z/z0,m) )

    stopped += p

stopped /= len(energy_bins)

# Plot the stopping distribution
#plt.figure()
#plt.xlabel("Depth in Moderator (µm)")
#plt.ylabel("Stopped Positrons (arb. units)")
#plt.title("Stopping Position Distribution in 50 µm Moderator")
#plt.show()



dims_50um = [9.975, 10.025, 50, "Data-50um"]
dims_100um = [9.95, 10.05, 100, "Data-100um"]
dims_150um = [9.925, 10.075, 150, "Data-150um"]
dims_200um = [9.9, 10.1, 200, "Data-200um"]
dims = dims_200um

files = [f"{dims[3]}/{a}" for a in os.listdir(dims[3]) if a[0:3]=="Out" and a[-3:]=="dat"]

end_z = []
all_initial_angle = []

def run_file(file):
    try: df = pd.read_parquet(file)
    except Exception:
        print(f"Err {file}")
        return
    loc_all_initial_angle = list(df["initialAngle"])
    df = df[df["endz"] > dims[0]]
    df = df[df["endz"] < dims[1]]
    loc_end_z = list((df["endz"] - dims[0]) * 1000.0)
    
    global end_z, all_initial_angle
    end_z = end_z + loc_end_z
    all_initial_angle = all_initial_angle + loc_all_initial_angle

for file in files:
    run_file(file)

end_z = np.array(end_z)

counts, bins = np.histogram(end_z,np.linspace(0,50,250))
counts = np.array(counts)
counts = counts / len(all_initial_angle)

# Plotting: simulation histogram with stairs, and theoretical line on twin y-axis
fig, ax1 = plt.subplots()
ax1.stairs(counts, bins, label="Simulation", color="tab:blue")
ax1.set_ylabel("Fraction of Positrons (Sim)", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")

#ax2 = ax1.twinx()
ax1.plot(z, stopped/1.4, label="Theoretical", color="tab:red")
ax1.set_ylabel("Theoretical Stopping Dist.", color="tab:red")
ax1.tick_params(axis='y', labelcolor="tab:red")

plt.title("Stopping Distribution: Simulation vs. Theoretical")

plt.show()
