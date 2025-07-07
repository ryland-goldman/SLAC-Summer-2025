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

files = ["OutN4.dat"]
files = [a for a in os.listdir() if a[0:3]=="Out"]


for file in files:
    print(file)
    df = pd.read_parquet(file)

    for eventID, eventdf in df.groupby('EventID'):
        for trackID, trackdf in eventdf.groupby("TrackID"):

            #if not eventID in [601276, 9931027, 571661145, 813621071, 979691455]: continue

            z = []
            t = []
            pz = []


            try:
                for particle in trackdf.iloc:
                    t.append(float(particle.t))
                    z.append(float(particle.z))
                    pz.append(float(particle.Pz))
                    
            except IndexError: pass

            try:
                trackdf = trackdf.sort_values('t')
                last = trackdf[trackdf["Pz"] < 0.001].iloc[0].z
                end = trackdf.iloc[-1]
                if end.z>10.026:
                    print("Pass through")
                    continue # bounce
                if not np.sum(trackdf["z"]==9.97) == 1:
                    print("Bounce")
                    continue
                if not (last-9.975)*1000.0 < 2:
                    print("Not in")
                    continue # went in
                if trackdf.iloc[0].z > 9.97:
                    print("Invalid")
                    continue # didn't trigger detector 1
            except IndexError:
                continue

            print(eventID)
            print(eventdf)

            z=(np.array(z)-9.975)*1000.0
            t=np.array(t)*1000.0

            sc = plt.scatter(z,pz,c=t)
            plt.xlabel("z (µm)")
            plt.ylabel("Pz (MeV/c)")

            cbar = plt.colorbar(sc)
            cbar.set_label("Time (ps)")

            plt.show()



# OutN4.dat (out.zip):
#   601276    - two positrons
#   9931027   - also two?
#   571661145 - ???
#   813621071 - bounce? but why the yellow at 3µm
#   979691455 - bounce but not triggering -5µm detector?

# OutN4.dat:
#   230/1134 - bounce off moderator, then also bounce off target?
# OutN5.dat:
#   586/1195 - bounce off moderator, then also bounce off target?
# OutN1.dat:
#   865/1140 - ???
# OutN0.dat: 575/1006
