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

electron_mass = 0.511  # MeV/c^2
event_count = 1000000

files = [a for a in os.listdir() if a[0:3]=="Out"]

def momentum_to_ke(p_mev_c):
    m_e = 0.510998950 # MeV
    total_energy = (p_mev_c**2 + m_e**2)**0.5
    ke_keV = (total_energy - m_e)*1e3
    return ke_keV


for file in files:
    print(file)
    df = pd.read_parquet(file)

    for runID, rundf in df.groupby('RunID'):
        for eventID, eventdf in rundf.groupby('EventID'):
            for trackID, trackdf in eventdf.groupby("TrackID"):

                z = []
                t = []
                pz = []


                try:
                    for particle in trackdf.iloc:
                        t.append(float(particle.t))
                        z.append(float(particle.z))
                        pz.append(momentum_to_ke(math.sqrt(float(particle.Pz)**2 + float(particle.Px)**2 + float(particle.Py)**2)))
                        
                except IndexError: pass

                try:
                    trackdf = trackdf.sort_values('t')
                    last = trackdf[trackdf["Pz"] < 0.001].iloc[0].z
                    end = trackdf.iloc[-1]
                    if end.z>10.026:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Passed through moderator")
                        continue # bounce
                    if not np.sum(trackdf["z"]==9.97) == 1:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Bounced off moderator")
                        continue
                    #if not round((last-9.975)*1000.0) < 2:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("Stop position > 1 µm")
                        continue # went in
                    if trackdf.iloc[0].z > 9.971:
                        print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                        print("No hit")
                        continue # didn't trigger detector 1
                except IndexError:
                    continue

                print(f"Run {runID}, event {eventID}, track {trackID}: ",end='')
                print("Found")
                print(trackdf)

                z=(np.array(z)-9.975)*1000.0
                t=np.array(t)*1000.0

                sc = plt.scatter(z,pz,c=t)
                plt.xlabel("z (µm)")
                plt.ylabel("E (keV)")

                plt.axvline(x=(last-9.975)*1000.0)

                cbar = plt.colorbar(sc)
                cbar.set_label("Time (ps)")

                plt.show()
