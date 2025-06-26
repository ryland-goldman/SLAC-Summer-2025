import matplotlib
matplotlib.use('Agg')
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

num_threads = 1

# Constants
electron_mass = 0.511  # MeV/c^2

# number of events per run
event_count = 1000000

# energy in MeV
energy = 9

micron_to_mm = 0.001
# tungsten thickness in mm
min_thickness = 5*micron_to_mm
max_thickness = 50*micron_to_mm
thicknesses = np.linspace(min_thickness, max_thickness, 10)




def convert_to_builtin_type(obj):
    """
    Recursively convert NumPy and other non-JSON-serializable objects
    to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

i=0
threadnumber = 1

tasks = queue.Queue()
for thickness in thicknesses: tasks.put(thickness)

def worker(threadnumber):
    while not tasks.empty():
        thickness = tasks.get()
        run_sum(energy, thickness, 90, threadnumber)
        tasks.task_done()

def run_sum(energy, thickness, angle, threadnumber):
    global i
    result = subprocess.run(["C:\\Program Files\\Muons, Inc\\G4beamline\\bin\\g4bl.exe","Moderator.g4bl",f"KE={float(energy)}",f"thickness={thickness}",f"detectorRadius={1000}",f"detectorRadiusB={1001}",f"nEvents={event_count}", f"FNB=Detbackward{threadnumber}", f"FNF=Detforward{threadnumber}", f"FNS=Detsideways{threadnumber}", f"FNDS=Detsecond{threadnumber}"], capture_output=True, text=True)
    print(result.stdout)

    if not result.returncode == 0:
        print(f"Running iteration (Thread {threadnumber}, {energy} MeV)... failed with code",result.returncode)

    try:
        # Define energy bins (change as needed)
        energy_bins = np.linspace(0, 100, 201)  # 0 to 100 MeV in 0.5 MeV bins
        bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])

        # Store histogram data for plotting or analysis
        hist_data = {}
        num_pos = {}

        df1 = pd.read_csv(f"Detforward{threadnumber}.txt", skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0)
        df2 = pd.read_csv(f"Detsecond{threadnumber}.txt", skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0)
        df2 = df2[df2['PDGid'] != '-']
        df2['Pz'] = df2['Pz'].astype(float)
        df2 = df2[df2['Pz'] > 0]
        df1['PDGid'] = df1['PDGid'].astype(str)
        df2['PDGid'] = df2['PDGid'].astype(str)

        count = len(df2[df2['PDGid'] == '-11']) - len(df1[df1['PDGid'] == '-11'])

        # save parsed data
        with output_lock:
            try: df = pd.read_csv("batchdata.csv")
            except FileNotFoundError: df = pd.DataFrame(columns=["Energy","Thickness","Angle","Count"])
            df = pd.concat([df, pd.DataFrame([{"Energy":energy,"Thickness":thickness,"Angle":angle,"Count":count}])], ignore_index=True)
            df.to_csv("batchdata.csv",index=False)

            i += 1

        print(f"Running iteration {i} (Thread {threadnumber}, {thickness} mm)... done. ({count} events)")
    except Exception as e:
        print(f"Running iteration {i} (Thread {threadnumber}, {thickness} mm)... failed with exception",e)

output_lock = threading.Lock()
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
    threads.append(thread)

i -= num_threads

tasks.join()