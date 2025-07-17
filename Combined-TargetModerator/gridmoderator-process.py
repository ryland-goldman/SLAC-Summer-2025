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

grid = [4,4]

loc = f"Data-Grid-{grid[0]}x{grid[1]}"
size_x = (grid[1]+1)*10
size_z = (grid[0]+1)*10

files = [f"{loc}/{a}" for a in os.listdir(loc) if a[0:3]=="Out" and a[-3:]=="dat"]

threshold = 0.001 # 1 keV/c = thermalization

end_x = []
end_y = []
end_z = []
initial_p = []
initial_angle = []
all_initial_p = []
all_initial_angle = []

def run_file(file):
    try: df = pd.read_parquet(file)
    except Exception:
        print(f"Err {file}")
        return
    #df = df[df["initialE"] < 450]
    #df = df[df["initialE"] > 550]
    loc_all_initial_angle = list(df["initialAngle"])
    loc_all_initial_p = list(df["initialP"])
    df = df[df["endz"] > 7.9]
    df = df[df["endz"] < size_z + 8.1]
    df = df[df["endx"] > -(size_x)/2-0.1]
    df = df[df["endx"] < (size_x)/2 + 0.1]
    loc_end_x = list(df["endx"])
    loc_end_y = list(df["endy"])
    loc_end_z = list(df["endz"])
    loc_initial_p = list(df["initialP"])
    loc_initial_angle = list(df["initialAngle"])
    
    if True:
        global end_x, end_y, end_z, initial_angle, initial_p, all_initial_angle, all_initial_p
        end_x = end_x + loc_end_x
        end_y = end_y + loc_end_y
        end_z = end_z + loc_end_z
        initial_p = initial_p + loc_initial_p
        initial_angle = initial_angle + loc_initial_angle
        all_initial_p = all_initial_p + loc_all_initial_p
        all_initial_angle = all_initial_angle + loc_all_initial_angle

for i in files: run_file(i)

end_x = np.array(end_x)
end_y = np.array(end_y)
end_z = np.array(end_z)

dist_to_border_z = (end_z - 8) % 10
dist_to_border_z = dist_to_border_z - 10*(dist_to_border_z//5)
dist_to_border_z = np.abs(dist_to_border_z)
dist_to_border_x = (end_x - 5) % 10
dist_to_border_x = dist_to_border_x - 10*(dist_to_border_x//5)
dist_to_border_x = np.abs(dist_to_border_x)
dist_to_border = np.minimum(dist_to_border_x, dist_to_border_z)
out_prob = np.exp(-dist_to_border / 5.5e-5)
print(pd.DataFrame({"x":end_x,"y":end_y,"z":end_z,"dx":dist_to_border_x,"dz":dist_to_border_z,"d":1000*dist_to_border,"p":np.round(100*out_prob,2)}).sort_values("p"))
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
    std[0].append(np.std(diff_x))
    std[1].append(np.std(diff_y))
    std[2].append(np.std(diff_z))
    std[3].append(np.std(diff_r))
    rms[0].append(np.sqrt(np.mean(diff_x**2)))
    rms[1].append(np.sqrt(np.mean(diff_y**2)))
    rms[2].append(np.sqrt(np.mean(diff_z**2)))
    rms[3].append(np.sqrt(np.mean(diff_r**2)))
print(np.sum(out_prob),n_diff)
n = int(np.max(pd.read_parquet(f"{loc}/OutN0.dat")["RunID"])+1) * 1000 * len(os.listdir(loc))
print(f"{n} hit target")
print(f"{len(all_initial_angle)} hit moderator, {round(len(all_initial_angle)/n,4)}")
print(f"{len(end_z)} stop in moderator, {round(len(end_z)/len(all_initial_angle),4)}")
print(f"{n_diff} reemitted, {round(np.sum(out_prob)/len(end_z),4)}")
print(f"Moderator efficiency: {1e4*np.sum(out_prob)/len(all_initial_angle)}, pm {1e4*np.sqrt(np.sum(out_prob))/len(all_initial_angle)}")
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
print([grid[0],grid[1],brightness,error_bar])
import sys
sys.exit()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mask = (
    (end_x >= -size_x/2 - 10) & (end_x <= size_x/2 + 10) &
    (end_y >= -10) & (end_y <= 60) &
    (end_z >= 0) & (end_z <= size_z + 18)
)
#ax.scatter(end_x[mask],end_y[mask],end_z[mask])
ax.set_xlim(-size_x/2 - 10,size_x/2 + 10)
ax.set_ylim(-10,60)
ax.set_zlim(0,size_z+18)
ax.scatter(diff_x,diff_y,diff_z)
plt.show()

makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) )
#plt.plot(np.linspace(0,50,100), makhovian(np.linspace(0,50,100), 1.828, 35.24))
#counts, bins = np.histogram(end_z,np.linspace(0,dims[2],dims[2]))
counts, bins = np.histogram(end_z,np.linspace(0,106,106))
counts = np.array(counts)
counts = counts / len(all_initial_angle)
plt.stairs(counts, bins)
#for i in range(8,108,10): plt.axvline(x=i,color='red')
plt.title("Stopping Distribution")

plt.show()

fig, axs = plt.subplots(2, 1, figsize=(6, 10))
counts1, bins1 = np.histogram(initial_p,bins=np.linspace(0,10,50))
counts2, bins2 = np.histogram(all_initial_p,bins=np.linspace(0,10,50))
ratio = counts1 / counts2
axs[0].stairs(ratio, bins1)
axs[0].set_xlabel("Initial Momentum (MeV/c)")
axs[0].set_ylabel("Fraction Stopped")
counts1, bins1 = np.histogram(initial_angle,bins=np.linspace(0,90,90))
counts2, bins2 = np.histogram(all_initial_angle,bins=np.linspace(0,90,90))
ratio = counts1 / counts2
axs[1].stairs(ratio, bins1)
axs[1].set_xlabel("Initial Angle (Deg)")
axs[1].set_ylabel("Fraction Stopped")

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(6, 10))

counts, bins = np.histogram(initial_p,bins=np.linspace(0,10,50))
axs[1].stairs(counts, bins)
axs[1].set_title("z=0 Energy Distribution (of stopped e$^+$)")
ax2 = axs[1].twinx()
counts, bins = np.histogram(all_initial_p,bins=np.linspace(0,10,50))
ax2.stairs(counts,bins,color='red')

counts, bins = np.histogram(initial_angle,bins=np.linspace(0,90,90))
axs[0].stairs(counts, bins)
axs[0].set_title("z=0 Angular Distribution (of stopped e$^+$)")
ax2 = axs[0].twinx()
counts, bins = np.histogram(all_initial_angle,bins=np.linspace(0,90,90))
ax2.stairs(counts,bins,color='red')


axs[0].set_xlabel("Penetration Depth (µm)")
axs[1].set_xlabel("Initial Momentum (MeV/c)")
axs[0].set_xlabel("Initial Angle (Deg)")

axs[0].set_ylabel("Count")
axs[1].set_ylabel("Count")
axs[0].set_ylabel("Count")

'''
try:
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) ) * (n_events-fail)
    popt, pcov = curve_fit(makhovian, bin_centers, counts, p0=(2,30))
    theoretical_x = np.linspace(0,50,100)
    plt.plot(theoretical_x, makhovian(theoretical_x, *popt), color='red')
    print(*popt)
except Exception as e:
    print(e)
'''

plt.tight_layout()
plt.show()
