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

files = [f"Det{i}.txt" for i in range(1,50)]

z = []
t = []
pz = []

for file in files:
    df = pd.read_csv(file, skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0)

    df['PDGid'] = df['PDGid'].astype(str)
    df['EventID'] = df['EventID'].astype(str)

    df = df[df["EventID"] == "9743"]
    df = df[df["PDGid"] == "-11"]
    
    try:
        for particle in df.iloc:
            t.append(float(particle.t))
            z.append(float(particle.z))
            pz.append(float(particle.Pz))
            
    except IndexError: pass

z=(np.array(z)-0.975)*1000.0
t=np.array(t)*1000.0

sc = plt.scatter(z,pz,c=t)
plt.xlabel("z (µm)")
plt.ylabel("Pz (MeV/c)")

cbar = plt.colorbar(sc)
cbar.set_label("Time (ps)")


plt.show()