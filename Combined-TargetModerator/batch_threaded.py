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
import random

import warnings
warnings.filterwarnings("ignore")

num_threads = 12

# Constants
electron_mass = 0.511  # MeV/c^2

# number of events per run
event_count = 100000



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
for i in range(num_threads): tasks.put(f"N{i}")

def worker(threadnumber):
    while not tasks.empty():
        threadnumber = tasks.get()
        run_sum(threadnumber)
        tasks.task_done()

def run_sum(threadnumber):
    result = subprocess.run(["/Applications/G4beamline-3.08.app/Contents/MacOS/g4bl","Combined.g4bl",f"ThreadNumber={threadnumber}",f"RandSeed={random.randint(0,2**32-1)}"], capture_output=True, text=True)

    if not result.returncode == 0:
        print(f"Running iteration (Thread {threadnumber})... failed with code",result.returncode)
        print(result.stdout)

    try:
        subprocess.run(f"cat Det*{threadnumber}.txt > Out{threadnumber}.txt", shell=True)
        subprocess.run(f"rm Det*{threadnumber}.txt",shell=True)

        df = pd.read_csv(f"Out{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"z":float,"Pz":float,"t":float,"PDGid":str,"EventID":int}, usecols=["z","Pz","t","PDGid","EventID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df = df[df["PDGid"] == "-11"]
        df = df.drop('PDGid', axis=1)

        df = df.convert_dtypes()

        if f"Out{threadnumber}.dat" in os.listdir():
            main_df = pd.read_parquet(f"Out{threadnumber}.dat")
            df = pd.concat([main_df,df],ignore_index=True)

        df.to_parquet(f"Out{threadnumber}.dat",engine="pyarrow",compression="brotli",compression_level=10,index=False)

        os.remove(f"Out{threadnumber}.txt")
        
    except Exception as e:
        print(f"Running iteration {i} (Thread {threadnumber})... failed with exception",e)

output_lock = threading.Lock()
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
    threads.append(thread)

i -= num_threads

tasks.join()