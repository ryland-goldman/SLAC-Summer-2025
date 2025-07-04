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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

import warnings
warnings.filterwarnings("ignore")

electron_mass = 0.511  # MeV/c^2
event_count = 10000

files = [a for a in os.listdir() if a[0:3]=="Out"]

threshold = 0.001 # 1 keV/c = thermalization

end_z = []
fail = 0
n_events = 0
bounce = 0

for file in files:
    df = pd.read_parquet(file)

    for eventID, eventdf in df.groupby('EventID'):
        n_events += 1
        try:
            eventdf = eventdf.sort_values('t')
            last = eventdf[eventdf["Pz"] < threshold].iloc[0].z
            end = eventdf.iloc[-1]
            if end.z<9.975 or end.z>10.025:
                bounce += 1
                n_events -= 1
                continue
            end_z.append((last-9.975)*1000.0)
        except IndexError: fail += 1
        except Exception as e:
            print(f"Exception {e} with event {eventID}:", eventdf)
            fail += 1
    
end_z = np.array(end_z)
print(fail, bounce, n_events-fail)

counts, bins = np.histogram(end_z,bins=np.linspace(0,50,50))
plt.stairs(counts, bins)

try:
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    makhovian = lambda x, m, z0: (m * np.pow(x, m-1) / np.pow(z0,m)) * np.exp( -np.pow(x/z0,m) ) * (n_events-fail)
    popt, pcov = curve_fit(makhovian, bin_centers, counts, p0=(2,30))
    theoretical_x = np.linspace(0,50,100)
    plt.plot(theoretical_x, makhovian(theoretical_x, *popt), color='red')
    print(*popt)
except Exception as e:
    print(e)

plt.xlabel("Penetration Depth (µm)")
plt.ylabel("Count /$10^4$")


plt.show()
