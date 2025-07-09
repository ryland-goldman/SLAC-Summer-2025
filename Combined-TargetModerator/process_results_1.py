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

files = [a for a in os.listdir() if a[0:3]=="Out" and a[-3:]=="pkl"]

dims_50um = [9.975, 10.025, 50]
dims_200um = [9.9, 10.1, 200]
dims = dims_50um

threshold = 0.001 # 1 keV/c = thermalization

end_z = []
initial_p = []
initial_angle = []
all_initial_p = []
all_initial_angle = []

def run_file(file):
    with open(file, 'rb') as f: final_out_data = pkl.load(f)
    loc_end_z = final_out_data["end_z"]
    loc_initial_p = final_out_data["initial_p"]
    loc_initial_angle = final_out_data["initial_angle"]
    loc_all_initial_p = []
    loc_all_initial_angle = []
    
    with output_lock:
        global end_z, initial_angle, initial_p, all_initial_angle, all_initial_p
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
for i in range(12):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
    threads.append(thread)

tasks.join()

print(end_z)
end_z = np.where(np.array(end_z) < 1.0, 1.0, np.round(end_z))

print(initial_angle)
print(initial_p)



counts, bins = np.histogram(end_z,bins=np.linspace(0,dims[2],dims[2]))
plt.stairs(counts, bins)
plt.title("Stopping Distribution")

plt.show()
plt.clf()

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
plt.clf()

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
