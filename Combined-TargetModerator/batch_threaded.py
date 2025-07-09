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
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")


conf = "mac"

if conf=="aws":
    num_threads = 192
    g4blloc = "/home/ubuntu/G4beamline-3.08/bin/g4bl"
    out_dir = "/home/ubuntu"
if conf=="mac":
    num_threads = 14
    g4blloc = "/Applications/G4beamline-3.08.app/Contents/MacOS/g4bl"
    out_dir = "."


dims_50um = [9.975, 10.025, 50, "Combined.g4bl"]
dims_200um = [9.9, 10.1, 200, "Combined200.g4bl"]
dims = dims_200um



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
    result = subprocess.run([g4blloc,dims[3],f"ThreadNumber={threadnumber}",f"RandSeed={random.randint(0,2**32-1)}"], capture_output=True, text=True)

    if not result.returncode == 0:
        print(f"Running iteration (Thread {threadnumber})... failed with code",result.returncode)
        print(result.stdout)

    try:
        subprocess.run(f"cat Det*{threadnumber}.txt > Out{threadnumber}.txt", shell=True)
        subprocess.run(f"rm Det*{threadnumber}.txt",shell=True)

        df = pd.read_csv(f"Out{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df = df[df["PDGid"] == "-11"]
        df = df.drop('PDGid', axis=1)

        tmp_df = df
        df["RunID"] = 0

        if f"Out{threadnumber}.dat" in os.listdir(out_dir):
            main_df = pd.read_parquet(f"{out_dir}/Out{threadnumber}.dat")
            df["RunID"] = main_df['RunID'].max() + 1
            df = pd.concat([main_df,df],ignore_index=True)

        df["RunID"] = df["RunID"].astype('Int16')

        df.to_parquet(f"{out_dir}/Out{threadnumber}.dat",engine="pyarrow",compression="brotli",compression_level=10,index=False)

        os.remove(f"Out{threadnumber}.txt")

        if not f"Out{threadnumber}.pkl" in os.listdir():
            final_out_data = {"initial_angle":[],"initial_p":[],"end_z":[]}
        else:
            with open(f"Out{threadnumber}.pkl", 'rb') as f: final_out_data = pkl.load(f)

        for eventID, eventdf in tmp_df.groupby('EventID'):
            for trackID, trackdf in eventdf.groupby("TrackID"):
                try:
                    trackdf = trackdf.sort_values('t')
                    end = trackdf.iloc[-1]
                    if trackdf.iloc[0].z > dims[0]-0.001:
                        continue
                    if not np.sum(trackdf["z"]==dims[0]-0.005) == 1:
                        continue
                    if end.z>10.101:
                        continue
                    total_p = math.sqrt(trackdf.iloc[0].Px**2 + trackdf.iloc[0].Py**2 + trackdf.iloc[0].Pz**2)
                    angle = math.acos(trackdf.iloc[0].Pz / total_p)
                    final_out_data["initial_angle"].append( angle*180.0/math.pi )
                    final_out_data["initial_p"].append( total_p )
                    final_out_data["end_z"].append((end.z-dims[0])*1000.0)
                except IndexError as e:
                    pass
        
        with open(f"Out{threadnumber}.pkl", 'wb') as f:
            pkl.dump(final_out_data, f, protocol=pkl.HIGHEST_PROTOCOL)

        
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