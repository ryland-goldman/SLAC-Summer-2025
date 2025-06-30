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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

import warnings
warnings.filterwarnings("ignore")

electron_mass = 0.511  # MeV/c^2
event_count = 10000

files = [f"Det{i}.txt" for i in range(1,50)]
dfs = []

for file in files:
    dfs.append(pd.read_csv(file, skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0))


df = pd.concat(dfs, ignore_index=True)

df['z'] = pd.to_numeric(df['z'], errors='coerce')
df = df.dropna(subset=['z'])
df = df.reset_index(drop=True)

df['PDGid'] = df['PDGid'].astype(str)
df['EventID'] = df['EventID'].astype(str)
df['z'] = df['z'].astype(float)
df['Px'] = df['Px'].astype(float)
df['Py'] = df['Py'].astype(float)
df['Pz'] = df['Pz'].astype(float)
df = df[df["PDGid"] == "-11"]

end_z = []
fail = 0

n_events = 0

for eventID, eventdf in df.groupby('EventID'):
    n_events += 1
    try:
        eventdf = eventdf.sort_values('t')
        last = eventdf.iloc[-1].z
        end_z.append((last-0.975)*1000.0)
        
    except IndexError: fail += 1
    
end_z = np.array(end_z)
print(fail)

counts, bins = np.histogram(end_z,bins=np.linspace(0,50,50))
counts_trimmed = counts[1:-1]
bins_trimmed = bins[1:-1]
plt.stairs(counts_trimmed, bins_trimmed)
#plt.stairs(counts, bins)

m = 1.8105
z0 = 33.37
theoretical_x = np.linspace(0,50,100)
theoretical_y = (m * np.pow(theoretical_x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(theoretical_x/z0,m) ) * n_events

plt.plot(theoretical_x, theoretical_y, color='red')

plt.xlabel("Penetration Depth (µm)")
plt.ylabel("Count /$10^4$")

plt.show()