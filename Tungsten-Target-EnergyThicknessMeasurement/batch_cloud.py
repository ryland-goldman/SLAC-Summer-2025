import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import os
import json
import threading
import queue
import requests, subprocess
output_lock_2 = threading.Lock()
def spot_terminated():
    if "stop" in os.listdir(): return True
    try:
        response = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time", timeout=1)
        return response.status_code == 200
    except Exception: return False
def print2(strout=""):
    with output_lock_2:
        with open("stdout.txt","a") as f: f.write(strout+"\n")
print2("===")
num_threads = int(subprocess.run(["nproc"], capture_output=True, text=True).stdout)

# Constants
electron_mass = 0.511  # MeV/c^2

# number of events per run
event_count = 1000000

# energy min/max in MeV
min_E = 131
max_E = 150
energy_range = np.array([5.0,10.0,15.0,20.0,40.0,70.0,100.0])

# tungsten thickness in mm
min_thickness = 0.5
max_thickness = 10
thickness_range = np.linspace(min_thickness, max_thickness, 20)


estimated_rate=800
print2(f"Using energies {min_E} MeV to {max_E} MeV ({len(energy_range)} measurements)")
print2(f"Using thickness {min_thickness} mm to {max_thickness} mm ({len(thickness_range)} measurements)")
print2()
print2(f"At a rate of {estimated_rate} ev/sec, each measurement will take {event_count/estimated_rate} sec")
print2(f"With {len(energy_range)*len(thickness_range)} measurements and {num_threads} threads, this is {len(energy_range)*len(thickness_range)*event_count*(1/estimated_rate)*(1/num_threads)} sec ({len(energy_range)*len(thickness_range)*event_count*(1/estimated_rate)*(1/3600)*(1/num_threads)} hrs)")
print2()


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
total_num=len(energy_range)*len(thickness_range)

tasks = queue.Queue()
aa, bb = np.meshgrid(energy_range, thickness_range, indexing='ij')
allcombos = np.stack((aa.ravel(), bb.ravel()), axis=1)

pairs_done = []

if "batchdata.csv" in os.listdir():
    df = pd.read_csv("batchdata.csv")
    d = df.T.to_dict()
    for key in d.keys(): pairs_done.append([d[key]["Energy"],d[key]["Thickness"]])

pairs_done=np.array(pairs_done)
for combo in allcombos:
    combo=np.array(combo)
    if not len(pairs_done)==0:
        if (combo == pairs_done).all(axis=1).any(): continue
    tasks.put(combo)

def worker(threadnumber):
    while True:
        if spot_terminated():
            print2(f"{threadnumber} terminated due to spot stop")
            break
        task = tasks.get()
        print2(f"{threadnumber} picked up task {task}")
        if not type(task) in [np.ndarray,list]:
            print2(f"{threadnumber} terminated due to tasks empty")
            tasks.task_done()
            break
        run_sum(task[0], task[1], threadnumber)
        tasks.task_done()

def run_sum(energy, thickness, threadnumber):
    global i
    result = subprocess.run(["/home/ubuntu/G4beamline-3.08/bin/g4bl","TungstenTarget.g4bl",f"KE={float(energy)}",f"thickness={thickness}",f"nEvents={event_count}", f"FNB=DetBackward{threadnumber}", f"FNF=DetForward{threadnumber}", f"FNS=DetSideways{threadnumber}"], capture_output=True, text=True)
    #print2(result.stdout)

    if not result.returncode == 0:
        print2(f"Running iteration (Thread {threadnumber}, {energy} MeV, {thickness} mm)... failed with code {result.returncode}")

    try:
        # Define energy bins (change as needed)
        energy_bins = np.linspace(0, 100, 201)  # 0 to 100 MeV in 0.5 MeV bins
        bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])

        df = pd.read_csv(f"DetForward{threadnumber}.txt", skiprows=1, names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), delim_whitespace=True ).drop(index=0)

        electrons = df[df['PDGid'] == 11]
        positrons = df[df['PDGid'] == -11]
        electron_E = electrons['Pz']
        positron_E = positrons['Pz']

        energy_bins = np.linspace(0, 100, 201)
        positron_E = np.array(list(positron_E))


        # save parsed data
        # with output_lock:
        try: df = pd.read_csv(f"batchdata{threadnumber}.csv")
        except FileNotFoundError: df = pd.DataFrame(columns=["Energy","Thickness","Count","Rate"])
        df = pd.concat([df, pd.DataFrame([{"Energy":energy,"Thickness":thickness,"Count":len(positron_E),"Rate":len(positron_E)/event_count}])], ignore_index=True)
        df.to_csv(f"batchdata{threadnumber}.csv",index=False)

        # save raw data
        #data = []
        #if os.path.exists("batchdata.json"):
        #    with open("batchdata.json","r") as file: data = json.load(file)
        #data.append({"Energy":energy,"Thickness":thickness,"Raw":positron_E})
        #with open("batchdata.json","w") as file: json.dump(convert_to_builtin_type(data), file, indent=4)

        i += 1

    except Exception as e:
        print2(f"Running iteration {i} (Thread {threadnumber}, {energy} MeV, {thickness} mm)... failed with exception {e}")

    print2(f"Running iteration {i} (Thread {threadnumber}, {energy} MeV, {thickness} mm)... done. ({len(positron_E)} events)")

output_lock = threading.Lock()
threads = []
for i in range(num_threads):
    tasks.put(None)
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
    threads.append(thread)

tasks.join()
