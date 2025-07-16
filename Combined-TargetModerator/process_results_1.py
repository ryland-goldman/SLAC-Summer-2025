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

dims_50um = [9.975, 10.025, 50, "Data-50um"]
dims_100um = [9.95, 10.05, 100, "Data-100um"]
dims_150um = [9.925, 10.075, 150, "Data-150um"]
dims_200um = [9.9, 10.1, 200, "Data-200um"]
dims = dims_200um

files = [f"{dims[3]}/{a}" for a in os.listdir(dims[3]) if a[0:3]=="Out" and a[-3:]=="dat"]

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
    #print(df.sort_values("endz"))
    n=df.shape[0]
    loc_all_initial_angle = list(df["initialAngle"])
    loc_all_initial_p = list(df["initialP"])
    df = df[df["endz"] > dims[0]]
    df = df[df["endz"] < dims[1]]
    #print(n,df.shape[0],df.shape[0]/n,"\n")
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

end_z = np.array(end_z)

dist_to_border = np.abs(end_z - np.round(end_z / 25.0) * 25.0)
out_prob = np.exp(-dist_to_border / 0.055)
std = [[],[],[]]
rms = [[],[],[]]
for j in range(100):
    n_diff = 0
    diff_x = []
    diff_y = []
    diff_z = []
    layer_distance = 10
    for i in range(len(end_z)):
        if out_prob[i] > np.random.uniform():
            n_layers = end_z[i] // 25.0
            diff_x.append(end_x[i])
            diff_y.append(end_y[i])
            diff_z.append(end_z[i]*0.001 + layer_distance * n_layers)
            print(f"Particle diffused: ({end_x[i]}, {end_y[i]}, {end_z[i]}) {n_layers}")
            n_diff += 1
    diff_x=np.array(diff_x)
    diff_y=np.array(diff_y)
    diff_z=np.array(diff_z)
    std[0].append(np.std(diff_x))
    std[1].append(np.std(diff_y))
    std[2].append(np.std(diff_z))
    rms[0].append(np.sqrt(np.mean(diff_x**2)))
    rms[1].append(np.sqrt(np.mean(diff_y**2)))
    rms[2].append(np.sqrt(np.mean(diff_z**2)))
print(np.sum(out_prob),n_diff)
n = int(np.max(pd.read_parquet(f"{dims[3]}/OutN0.dat")["RunID"])) * 1000 * len(os.listdir(dims[3]))
print(f"{n} hit target")
print(f"{len(all_initial_angle)} hit moderator, {round(len(all_initial_angle)/n,4)}")
print(f"{len(end_z)} stop in moderator, {round(len(end_z)/len(all_initial_angle),4)}")
print(f"{n_diff} reemitted, {round(np.sum(out_prob)/len(end_z),4)}")
print(f"Moderator efficiency: {np.sum(out_prob)/len(all_initial_angle)}")
print(f"System efficiency: {np.sum(out_prob)/n}")

with open("xyz.pkl","wb") as f: pkl.dump([diff_x,diff_y,diff_z], f, protocol=pkl.HIGHEST_PROTOCOL)
#print("Standard deviation",np.std(diff_x),np.std(diff_y),np.std(diff_z))
#diff_x=np.array(diff_x)
#diff_y=np.array(diff_y)
#diff_z=np.array(diff_z)
#print("RMS",np.sqrt(np.mean(diff_x**2)),np.sqrt(np.mean(diff_y**2)),np.sqrt(np.mean(diff_z**2)))
print("Standard deviation",np.mean(std[0]),np.mean(std[1]),np.mean(std[2]))
print("RMS",np.mean(rms[0]),np.mean(rms[1]),np.mean(rms[2]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(diff_x,diff_y,diff_z)
plt.show()

makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) )
#plt.plot(np.linspace(0,50,100), makhovian(np.linspace(0,50,100), 1.828, 35.24))
counts, bins = np.histogram(end_z,np.linspace(0,dims[2],dims[2]))
counts = np.array(counts)
counts = counts / len(all_initial_angle)
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
