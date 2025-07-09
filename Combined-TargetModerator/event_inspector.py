'''
Inspects a single event
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

import warnings
warnings.filterwarnings("ignore")

dims_50um = [9.975, 10.025, 50]
dims_200um = [9.9, 10.1, 200]
dims = dims_200um

electron_mass = 0.511  # MeV/c^2
event_count = 1000000

files = [a for a in os.listdir() if a[0:3]=="Out"]

def momentum_to_ke(p_mev_c):
    m_e = 0.510998950 # MeV
    total_energy = (p_mev_c**2 + m_e**2)**0.5
    ke_keV = (total_energy - m_e)*1e3
    return ke_keV

n_plots = 0
def plot_fig(z,pz,t):
    global fig, axes, n_plots
    if n_plots == 9:
        plt.show()
        n_plots = 0
    if n_plots == 0:
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
    sc = axes.flat[n_plots].scatter(z,pz,c=t)
    axes.flat[n_plots].set_xlabel("z (µm)")
    axes.flat[n_plots].set_ylabel("E (keV)")
    axes.flat[n_plots].axvline(x=(last-dims[0])*1000.0)
    axes.flat[n_plots].axvline(x=50.0,color='red')
    cbar = fig.colorbar(sc,ax=axes.flat[n_plots])
    cbar.set_label("Time (ps)")

    n_plots += 1

def plot_fig_3d(x,y,z,t):
    global fig, axes, n_plots
    if n_plots == 9:
        plt.show()
        n_plots = 0
    if n_plots == 0:
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        axes = [fig.add_subplot(3, 3, i + 1, projection='3d') for i in range(9)]
    sc = axes[n_plots].scatter(x,y,z,c=t)
    axes[n_plots].set_xlabel("x (mm)")
    axes[n_plots].set_ylabel("y (mm)")
    axes[n_plots].set_zlabel("z (µm)")
    cbar = fig.colorbar(sc,ax=axes[n_plots])
    cbar.set_label("Time (ps)")

    n_plots += 1

def plot_fig_mixed(x, y, z, t, pz):
    global fig, axes3d, axes2d, n_plots
    if n_plots == 3:
        plt.show()
        n_plots = 0
    if n_plots == 0:
        fig = plt.figure(figsize=(12, 14), constrained_layout=True)
        gs  = fig.add_gridspec(3, 2)
        axes3d = [fig.add_subplot(gs[i, 0], projection='3d') for i in range(3)]
        axes2d = [fig.add_subplot(gs[i, 1])                   for i in range(3)]
    sc3d = axes3d[n_plots].scatter(x, y, z, c=t)
    axes3d[n_plots].set_xlabel("x (mm)")
    axes3d[n_plots].set_ylabel("y (mm)")
    axes3d[n_plots].set_zlabel("z (µm)")
    cb3d = fig.colorbar(sc3d, ax=axes3d[n_plots])
    cb3d.set_label("Time (ps)")
    sc2d = axes2d[n_plots].scatter(z, pz, c=t)
    axes2d[n_plots].set_xlabel("z (µm)")
    axes2d[n_plots].set_ylabel("E (keV)")
    axes2d[n_plots].axvline(x=(last - dims[0]) * 1000.0)
    axes2d[n_plots].axvline(x=50.0, color='red')
    cb2d = fig.colorbar(sc2d, ax=axes2d[n_plots])
    cb2d.set_label("Time (ps)")

    n_plots += 1

for file in files:
    print(file)
    try: df = pd.read_parquet(file)
    except Exception:
        print(f"Err {file}")
        continue

    for runID, rundf in df.groupby('RunID'):
        for eventID, eventdf in rundf.groupby('EventID'):
            for trackID, trackdf in eventdf.groupby("TrackID"):

                x = []
                y = []
                z = []
                t = []
                pz = []


                try:
                    for particle in trackdf.iloc:
                        t.append(float(particle.t))
                        x.append(float(particle.x))
                        y.append(float(particle.y))
                        z.append(float(particle.z))
                        pz.append(momentum_to_ke(math.sqrt(float(particle.Pz)**2 + float(particle.Px)**2 + float(particle.Py)**2)))
                        
                except IndexError: pass

                try:
                    trackdf = trackdf.sort_values('t')
                    #last = trackdf[trackdf["Pz"] < 0.001].iloc[0].z
                    end = trackdf.iloc[-1]
                    last = end.z
                    '''if end.z>10.101:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Passed through moderator")
                        continue # bounce
                    if not np.sum(trackdf["z"]==9.9) == 1:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Bounced off moderator")
                        continue
                    if not round((last-9.9)*1000.0) < 50:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Stop position > 50 µm")
                        continue
                    if np.sum(trackdf["z"]==9.95) == 0:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Did not reach 50 µm")
                        continue
                    if trackdf.iloc[0].z > 9.899:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("No hit")
                        continue # didn't trigger detector 1'''
                    if trackdf.iloc[0].z > dims[0] - 0.001:
                        continue
                    if not np.sum(trackdf["z"]==dims[0] - 0.005) == 1:
                        continue
                    if end.z>10.1:
                        continue
                    if not np.sum(trackdf["z"]==dims[0] + 0.050) == 0:
                        continue
                    if momentum_to_ke(math.sqrt(float(end.Pz)**2 + float(end.Px)**2 + float(end.Py)**2)) < 500: continue
                except IndexError:
                    continue

                print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                print("Found")
                print(trackdf)

                z=(np.array(z)-dims[0])*1000.0
                t=np.array(t)*1000.0

                #plot_fig(z,pz,t)
                #plot_fig_3d(x,y,z,t)
                plot_fig_mixed(x,y,z,t,pz)
