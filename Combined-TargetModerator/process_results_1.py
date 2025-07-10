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

files = [a for a in os.listdir() if a[0:3]=="Out" and a[-3:]=="dat"]

dims_50um = [9.975, 10.025, 50]
dims_200um = [9.9, 10.1, 200]
dims = dims_200um

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
    loc_all_initial_angle = list(df["initialAngle"])
    loc_all_initial_p = list(df["initialP"])
    df = df[df["endz"] > dims[0]]
    df = df[df["endz"] < dims[1]]
    loc_end_x = list(df["endx"])
    loc_end_y = list(df["endy"])
    loc_end_z = list((df["endz"] - dims[0]) * 1000.0)
    loc_initial_p = list(df["initialP"])
    loc_initial_angle = list(df["initialAngle"])
    
    with output_lock:
        global end_x, end_y, end_z, initial_angle, initial_p, all_initial_angle, all_initial_p
        end_x = end_x + loc_end_x
        end_y = end_y + loc_end_y
        end_z = end_z + loc_end_z
        initial_p = initial_p + loc_initial_p
        initial_angle = initial_angle + loc_initial_angle
        all_initial_p = all_initial_p + loc_all_initial_p
        all_initial_angle = all_initial_angle + loc_all_initial_angle

tasks = queue.Queue()
for file in files: tasks.put(file)

def worker(threadnumber):
    while not tasks.empty():
        file = tasks.get()
        run_file(file)
        tasks.task_done()

output_lock = threading.Lock()
threads = []
#for i in range(12):
#    thread = threading.Thread(target=worker, args=(i,))
#    thread.start()
#    threads.append(thread)
for i in files: run_file(i)

#tasks.join()

#end_z = np.where(np.array(end_z) < 1.0, 1.0, np.round(end_z))
end_z = np.array(end_z)

dist_to_border = np.abs(end_z - np.round(end_z / 25.0) * 25.0)
out_prob = np.exp(-dist_to_border / 0.055)
n_diff = 0
diff_x = []
diff_y = []
diff_z = []
layer_distance = 10
for i in range(len(end_z)):
    if out_prob[i] > np.random.uniform():
        print(f"Particle diffused: ({end_x[i]}, {end_y[i]}, {end_z[i]})")
        n_layers = end_z[i] // 25
        diff_x.append(end_x[i])
        diff_y.append(end_y[i])
        diff_z.append(end_z[i]*0.001 + layer_distance * n_layers)
        n_diff += 1
print(np.sum(out_prob),n_diff)
print(len(end_z))
print(len(all_initial_angle))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(diff_x,diff_y,diff_z)
plt.show()

counts, bins = np.histogram(end_z,np.linspace(0,dims[2],dims[2]//5))
plt.stairs(counts, bins)
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
