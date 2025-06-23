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

num_threads = 12

# Constants
electron_mass = 0.511  # MeV/c^2

# number of events per run
event_count = 1000000

# energy in MeV
energy = 100

# tungsten thickness in mm
thickness = 7

# detector apex angles in deg
angle_min = 5
angle_max = 75
angles = np.linspace(angle_min, angle_max, 15)

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
for angle in angles: tasks.put(angle)

def worker(threadnumber):
    while not tasks.empty():
        angle = tasks.get()
        run_sum(energy, thickness, angle*math.pi/180, threadnumber)
        tasks.task_done()

def run_sum(energy, thickness, angle, threadnumber):
    global i
    result = subprocess.run(["C:\\Program Files\\Muons, Inc\\G4beamline\\bin\\g4bl.exe","TungstenTarget.g4bl",f"KE={float(energy)}",f"thickness={thickness}",f"detectorRadius={100*math.tan(angle)}",f"detectorRadiusB={100*math.tan(angle)+1}",f"nEvents={event_count}", f"FNB=Detbackward{threadnumber}", f"FNF=Detforward{threadnumber}", f"FNS=Detsideways{threadnumber}"], capture_output=True, text=True)
    #print(result.stdout)

    if not result.returncode == 0:
        print(f"Running iteration (Thread {threadnumber}, {angle} deg)... failed with code",result.returncode)

    try:
        # Define energy bins (change as needed)
        energy_bins = np.linspace(0, 100, 201)  # 0 to 100 MeV in 0.5 MeV bins
        bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])

        # Store histogram data for plotting or analysis
        hist_data = {}
        num_pos = {}

        df = pd.read_csv(f"Detforward{threadnumber}.txt", skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0)

        electrons = df[df['PDGid'] == 11]
        positrons = df[df['PDGid'] == -11]
        electron_E = electrons['Pz']
        positron_E = positrons['Pz']

        num_pos = len(positrons)
        frac_pos = num_pos/event_count


        energy_bins = np.linspace(0, 100, 201)
        positron_E = np.array(list(positron_E))

        counts, bin_edges, _ = plt.hist(positron_E, bins=energy_bins, color='blue', alpha=0.6, label='Positrons')
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        def exp_func(x, A, p0): return A * np.exp(-x / p0)

        mask = counts > 0
        fit_x = bin_centers[mask]
        fit_y = counts[mask]

        try:
            popt, pcov = curve_fit(exp_func, fit_x, fit_y, p0=[1, 10])
        except Exception:
            popt = [0,0]

        angle *= 180/math.pi

        # save parsed data
        with output_lock:
            try: df = pd.read_csv("batchdata.csv")
            except FileNotFoundError: df = pd.DataFrame(columns=["Energy","Thickness","Angle","Count","A","x0"])
            df = pd.concat([df, pd.DataFrame([{"Energy":energy,"Thickness":thickness,"Angle":angle,"Count":len(positron_E),"A":popt[0],"x0":popt[1]}])], ignore_index=True)
            df.to_csv("batchdata.csv",index=False)

            # save raw data
            data = []
            if os.path.exists("batchdata.json"):
                with open("batchdata.json","r") as file: data = json.load(file)
            data.append({"Energy":energy,"Thickness":thickness,"Angle":angle,"Count":len(positron_E),"A":popt[0],"x0":popt[1],"Raw":positron_E})
            with open("batchdata.json","w") as file: json.dump(convert_to_builtin_type(data), file, indent=4)

            i += 1

    except Exception as e:
        print(f"Running iteration {i} (Thread {threadnumber}, {angle} deg)... failed with exception",e)

    print(f"Running iteration {i} (Thread {threadnumber}, {angle} deg)... done. ({len(positron_E)} events)")

output_lock = threading.Lock()
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
    threads.append(thread)

i -= num_threads

tasks.join()