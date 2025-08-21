'''
Creates a histogram of the depth distribution
'''
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import subprocess
import os
import json
import threading
import queue
import math
import pickle as pkl

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

import warnings
warnings.filterwarnings("ignore")

ke_filter = 100

loc = f"data-amd"

#p = ["Out90.dat","Out91.dat","Out92.dat","Out93.dat","Out94.dat","Out95.dat","Out96.dat","Out97.dat","Out98.dat","Out99.dat"]
p = os.listdir(loc)
all_files = [f"{loc}/{a}" for a in p if a[0:3]=="Out" and a[-3:]=="dat"]
files = [f for f in all_files if not 'r' in f and not '_a' in f]
files2 = [f for f in all_files if 'r' in f]
files3 = [f for f in all_files if '_a' in f]

threshold = 0.001 # 1 keV/c = thermalization

end_x = []
end_y = []
end_z = []
initial_p = []
initial_angle = []
all_initial_p = []
all_initial_angle = []

n_stopped = 0
n_transmitted = 0
n_annihilated = 0
n_reflected = 0

def run_file(file,rfile,afile):
    global n_stopped, n_transmitted, n_annihilated, n_reflected
    try:
        df = pd.read_parquet(file)
        df2 = pd.read_parquet(rfile)
        df3 = pd.read_parquet(afile)
        print(file,rfile)
    except Exception:
        print(f"Err {file}")
        return
    df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
    df2["initialE"] = np.sqrt(df2["initialP"]**2 + 0.511**2) - 0.511
    df3["initialE"] = np.sqrt(df3["initialP"]**2 + 0.511**2) - 0.511
    #df = df[df["initialE"] < 450]
    #df = df[df["initialE"] > 550]
    df=df[df["initialE"]<ke_filter]
    df2=df2[df2["initialE"]<ke_filter]
    df3=df3[df3["initialE"]<ke_filter]

    loc_all_initial_angle = list(df["initialAngle"])
    loc_all_initial_p = list(df["initialE"])

    df = df[df["endz"] > 0.9]
    df = df[df["endz"] < 65]
    loc_end_x = list(df["endx"])
    loc_end_y = list(df["endy"])
    loc_end_z = list(df["endz"])
    loc_initial_p = list(df["initialE"])
    loc_initial_angle = list(df["initialAngle"])

    n_stopped += len(df)
    
    if True:
        global end_x, end_y, end_z, initial_angle, initial_p, all_initial_angle, all_initial_p
        end_x = end_x + loc_end_x
        end_y = end_y + loc_end_y
        end_z = end_z + loc_end_z
        initial_p = initial_p + loc_initial_p
        initial_angle = initial_angle + loc_initial_angle
        all_initial_p = all_initial_p + loc_all_initial_p
        all_initial_angle = all_initial_angle + loc_all_initial_angle
    
    loc_all_initial_angle = list(df2["initialAngle"])
    df2["initialE"] = np.sqrt(df2["initialP"]**2 + 0.511**2) - 0.511
    loc_all_initial_p = list(df2["initialE"])

    n_reflected += len(df2)
    
    if True:
        all_initial_p = all_initial_p + loc_all_initial_p
        all_initial_angle = all_initial_angle + loc_all_initial_angle

    loc_all_initial_angle = list(df3["initialAngle"])
    df3["initialE"] = np.sqrt(df3["initialP"]**2 + 0.511**2) - 0.511
    loc_all_initial_p = list(df3["initialE"])

    n_annihilated += len(df3)
    
    if True:
        all_initial_p = all_initial_p + loc_all_initial_p
        all_initial_angle = all_initial_angle + loc_all_initial_angle

for i,j,k in zip(files,files2,files3): run_file(i,j,k)
n_transmitted = len(all_initial_p) - n_stopped - n_annihilated - n_reflected

end_x = np.array(end_x)
end_y = np.array(end_y)
end_z = np.array(end_z)

dist_to_border_z = (end_z - 1)
dist_to_border_z = np.abs(dist_to_border_z)
#dist_to_border_x = (end_x - 5) % 10
#dist_to_border_x = dist_to_border_x - 10*(dist_to_border_x//5)
#dist_to_border_x = np.abs(dist_to_border_x)
dist_to_border = np.minimum(np.abs(dist_to_border_z-0.025), np.abs(dist_to_border_z+0.025))

out_prob = np.exp(-dist_to_border / 5.5e-5)
print(pd.DataFrame({"x":end_x,"y":end_y,"z":end_z,"dx":dist_to_border,"dz":dist_to_border_z,"d":1000*dist_to_border,"p":np.round(100*out_prob,2)}).sort_values("p"))
std = [[],[],[],[]]
rms = [[],[],[],[]]
for j in range(100):
    n_diff = 0
    diff_x = []
    diff_y = []
    diff_z = []
    layer_distance = 10
    for i in range(len(end_z)):
        if out_prob[i] > np.random.uniform():
            diff_x.append(end_x[i])
            diff_y.append(end_y[i])
            diff_z.append(end_z[i])
            n_diff += 1
    diff_x=np.array(diff_x)
    diff_y=np.array(diff_y)
    diff_z=np.array(diff_z)
    diff_r=np.sqrt(diff_x**2 + diff_y**2)
    if np.std(diff_y) > 20: continue
    std[0].append(np.std(diff_x))
    std[1].append(np.std(diff_y))
    std[2].append(np.std(diff_z))
    std[3].append(np.std(diff_r))
    rms[0].append(np.sqrt(np.mean(diff_x**2)))
    rms[1].append(np.sqrt(np.mean(diff_y**2)))
    rms[2].append(np.sqrt(np.mean(diff_z**2)))
    rms[3].append(np.sqrt(np.mean(diff_r**2)))
print(np.sum(out_prob),n_diff)
n = int(np.max(pd.read_parquet(f"{loc}/Out1.dat")["RunID"])+1) * 100000 * len(files)
print(f"{n} hit target")
print(f"{len(all_initial_angle)} hit moderator, {round(len(all_initial_angle)/n,4)}")
print(f"{len(end_z)} stop in moderator, {round(len(end_z)/len(all_initial_angle),4)}")
print(f"{n_diff} reemitted, {round(np.sum(out_prob)/len(end_z),4)}")
print(f"Moderator efficiency: {1e7*np.sum(out_prob)/len(all_initial_angle)}, pm {1e7*np.sqrt(np.sum(out_prob))/len(all_initial_angle)}")
print(f"System efficiency: {1e8*np.sum(out_prob)/n}, pm {1e8*np.sqrt(np.sum(out_prob))/n}")
print("Standard deviation",np.mean(std[0]),np.mean(std[1]),np.mean(std[2]),np.mean(std[3]))
print("RMS",np.mean(rms[0]),np.mean(rms[1]),np.mean(rms[2]),np.mean(rms[3]))

import scipy.constants

E = 2.59 * scipy.constants.elementary_charge
#p = math.sqrt( 2*scipy.constants.electron_mass*E )
p=1
sigma_px = p * 1/math.sqrt(3)
sigma_py = p * 1/math.sqrt(3)
sigma_pz = p * 1/math.sqrt(3)

sigma_x = np.mean(std[0])
sigma_y = np.mean(std[1])
sigma_z = np.mean(std[2])

N = np.sum(out_prob)
dN = math.sqrt(N)
dsigma_x = np.std(std[0])
dsigma_y = np.std(std[1])
dsigma_z = np.std(std[2])
brightness = N / (((2*scipy.constants.pi)**3) * sigma_px * sigma_py * sigma_pz * sigma_x * sigma_y * sigma_z )
error_bar = brightness * math.sqrt( (dN/N)**2 + (dsigma_x/sigma_x)**2 + (dsigma_y/sigma_y)**2 + (dsigma_z/sigma_z)**2  )
print("Brightness:",brightness,"pm",error_bar)
#print([grid[0],grid[1],brightness,error_bar])
import sys
#sys.exit()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mask = (
    (end_z >= 0) & (end_z <= 2)
)
#ax.scatter(end_x[mask],end_y[mask],end_z[mask])
#ax.set_xlim(-size_x/2 - 10,size_x/2 + 10)
#ax.set_ylim(-10,60)
#ax.set_zlim(0,size_z+18)
ax.scatter(diff_x,diff_y,diff_z)
plt.show()

makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) )
#plt.plot(np.linspace(0,50,100), makhovian(np.linspace(0,50,100), 1.828, 35.24))
#counts, bins = np.histogram(end_z,np.linspace(0,dims[2],dims[2]))
thickness = 0.05
end_z = (end_z - 1 + thickness/2) * 1000
counts, bins = np.histogram(end_z,np.linspace(0,thickness*1000,50))
counts = np.array(counts)
counts = counts / len(all_initial_angle)
counts *= 1e4
counts *= 50/(thickness*1000)
plt.stairs(counts, bins)
#for i in range(8,108,10): plt.axvline(x=i,color='red')
#plt.title("Stopping Distribution")
plt.xlabel("Penetration Depth (µm)")
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")

plt.show()

fig, axs = plt.subplots(2, 1, figsize=(6, 10))
counts1, bins1 = np.histogram(initial_p,bins=np.linspace(0,2,50))
counts2, bins2 = np.histogram(all_initial_p,bins=np.linspace(0,2,50))
ratio = counts1 / counts2
axs[0].stairs(ratio, bins1)
axs[0].set_xlabel("Initial Kinetic Energy (KE)")
axs[0].set_ylabel("Fraction Stopped")
counts1, bins1 = np.histogram(initial_angle,bins=np.linspace(0,90,90))
counts2, bins2 = np.histogram(all_initial_angle,bins=np.linspace(0,90,90))
ratio = counts1 / counts2
axs[1].stairs(ratio, bins1)
axs[1].set_xlabel("Initial Angle (Deg)")
axs[1].set_ylabel("Fraction Stopped")

plt.tight_layout()
plt.show()

# 2D histogram: Fraction stopped vs. KE and angle
ke_bins = np.linspace(0, 2, 50)
angle_bins = np.linspace(0, 90, 18)

# 2D histograms of all and stopped
hist_all, _, _ = np.histogram2d(all_initial_p, all_initial_angle, bins=[ke_bins, angle_bins])
hist_stopped, _, _ = np.histogram2d(initial_p, initial_angle, bins=[ke_bins, angle_bins])

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    fraction_stopped = np.nan_to_num(hist_stopped / hist_all)

plt.figure(figsize=(8, 6))
plt.imshow(
    fraction_stopped.T,
    extent=[ke_bins[0], ke_bins[-1], angle_bins[0], angle_bins[-1]],
    aspect='auto',
    origin='lower',
    interpolation='nearest'
)
plt.colorbar(label='Fraction Stopped')
plt.xlabel("Initial Kinetic Energy (MeV)")
plt.ylabel("Initial Angle (Degrees)")
plt.title("Fraction Stopped vs. KE and Angle")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(6, 10))

counts, bins = np.histogram(initial_p,bins=np.linspace(0,2,50))
axs[1].stairs(counts, bins)
axs[1].set_title("z=0 Energy Distribution (of stopped e$^+$)")
ax2 = axs[1].twinx()
counts, bins = np.histogram(all_initial_p,bins=np.linspace(0,2,50))
ax2.stairs(counts,bins,color='red')

counts, bins = np.histogram(initial_angle,bins=np.linspace(0,90,90))
axs[0].stairs(counts, bins)
axs[0].set_title("z=0 Angular Distribution (of stopped e$^+$)")
ax2 = axs[0].twinx()
counts, bins = np.histogram(all_initial_angle,bins=np.linspace(0,90,90))
ax2.stairs(counts,bins,color='red')


axs[0].set_xlabel("Penetration Depth (µm)")
axs[1].set_xlabel("Initial Kinetic Energy (MeV)")
axs[0].set_xlabel("Initial Angle (Deg)")

axs[0].set_ylabel("Count")
axs[1].set_ylabel("Count")
axs[0].set_ylabel("Count")

plt.tight_layout()
plt.show()



reflected_initE = []
for file in files2:
    df = pd.read_parquet(file)
    df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
    df=df[df["initialE"]<ke_filter]
    reflected_initE += list(df["initialE"])

# Plot histogram of reflected initial kinetic energies
plt.figure(figsize=(6, 4))
counts1, bins1 = np.histogram(reflected_initE, bins=np.linspace(0, 5, 100))
counts2, bins2 = np.histogram(all_initial_p, bins=np.linspace(0, 5, 100))
ratio = counts1 / (counts1 + counts2)
plt.stairs(ratio, bins1)
plt.xlabel("Initial Kinetic Energy (MeV)")
plt.ylabel("Fraction Reflected")
plt.title("Energy of Reflected Positrons")
plt.tight_layout()
plt.show()

# 2D histogram: Fraction reflected vs. KE and angle
ke_bins = np.linspace(0, 5, 100)
angle_bins = np.linspace(0, 90, 10)

# Create arrays for reflected energies and angles
reflected_initE = []
reflected_initAngle = []

for file in files2:
    df = pd.read_parquet(file)
    df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
    df=df[df["initialE"]<ke_filter]
    reflected_initE += list(df["initialE"])
    reflected_initAngle += list(df["initialAngle"])

#all_initial_p += reflected_initE
#all_initial_angle += reflected_initAngle

# 2D histograms of all and reflected
hist_all, _, _ = np.histogram2d(all_initial_p, all_initial_angle, bins=[ke_bins, angle_bins])
hist_reflected, _, _ = np.histogram2d(reflected_initE, reflected_initAngle, bins=[ke_bins, angle_bins])

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    fraction_reflected = np.nan_to_num(hist_reflected / (hist_all))

# Plot
# Plot
plt.figure(figsize=(8, 6))
fraction_reflected = np.nan_to_num(fraction_reflected, nan=0.0, posinf=0.0, neginf=0.0)
plt.imshow(
    fraction_reflected.T,
    extent=[ke_bins[0], ke_bins[-1], angle_bins[0], angle_bins[-1]],
    aspect='auto',
    origin='lower',
    interpolation='nearest',
    vmin=0, vmax=1  # Avoid auto-scaling with NaNs/infs
)
plt.colorbar(label='Fraction Reflected')
plt.xlabel("Initial Kinetic Energy (MeV)")
plt.ylabel("Initial Angle (Degrees)")
plt.title("Fraction Reflected vs. KE and Angle")
plt.tight_layout()
plt.show()

# Energy-dependent fate breakdown
energy_bins = np.linspace(0, 5, 100)

stopped_ke = np.array(initial_p)
reflected_ke = np.array(reflected_initE)
annihilated_ke = []
for file in files3:
    df = pd.read_parquet(file)
    df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
    df=df[df["initialE"]<ke_filter]
    annihilated_ke += list(df["initialE"])
annihilated_ke = np.array(annihilated_ke)

all_ke = np.array(all_initial_p)

hist_total, _ = np.histogram(all_ke, bins=energy_bins)
hist_stopped, _ = np.histogram(stopped_ke, bins=energy_bins)
hist_reflected, _ = np.histogram(reflected_ke, bins=energy_bins)
hist_annihilated, _ = np.histogram(annihilated_ke, bins=energy_bins)
hist_transmitted = hist_total - (hist_stopped + hist_reflected + hist_annihilated)

# Normalize each bin to 1 for stacking
with np.errstate(divide='ignore', invalid='ignore'):
    fraction_stopped = np.nan_to_num(hist_stopped / hist_total)
    fraction_transmitted = np.nan_to_num(hist_transmitted / hist_total)
    fraction_reflected = np.nan_to_num(hist_reflected / hist_total)
    fraction_annihilated = np.nan_to_num(hist_annihilated / hist_total)

bottom = np.zeros_like(fraction_stopped)
width = energy_bins[1] - energy_bins[0]
bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

plt.figure(figsize=(8, 6))
for frac, label, color in zip(
    [fraction_stopped, fraction_transmitted, fraction_reflected, fraction_annihilated],
    ['Stopped', 'Transmitted', 'Reflected', 'Annihilated'],
    ['darkgreen', 'steelblue', 'goldenrod', 'firebrick']
):
    plt.bar(bin_centers, frac, bottom=bottom, width=width, label=label, color=color)
    bottom += frac

print(len(stopped_ke),len(all_ke),len(stopped_ke)/len(all_ke))

plt.xlabel("Initial Kinetic Energy (MeV)")
plt.ylabel("Fraction")
#plt.title("Energy-dependent Positron Fate Breakdown")
plt.legend()
plt.tight_layout()
plt.show()