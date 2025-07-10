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
    num_threads = 1
    g4blloc = "/Applications/G4beamline-3.08.app/Contents/MacOS/g4bl"
    out_dir = "."


dims_50um = [9.975, 10.025, 50, "CombinedWithPhotonTrack.g4bl"]
#dims_200um = [9.9, 10.1, 200, "Combined200.g4bl"]
dims = dims_50um

def momentum_to_ke(p_mev_c):
    m_e = 0.510998950 # MeV
    total_energy = (p_mev_c**2 + m_e**2)**0.5
    ke_keV = (total_energy - m_e)*1e3
    return ke_keV


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
        df_initial = pd.read_csv(f"DetNeg5{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df_stop = pd.read_csv(f"DetKill{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df_newparticles = pd.read_csv(f"NewParticles{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

        df_initial = df_initial[df_initial["PDGid"] == "-11"]
        df_stop = df_stop[df_stop["PDGid"] == "-11"]
        df_newparticles = df_newparticles[df_newparticles["PDGid"] == "22"]
        df_initial = df_initial.drop('PDGid', axis=1)
        df_stop = df_stop.drop('PDGid', axis=1)
        df_newparticles = df_newparticles.drop('PDGid', axis=1)

        df = pd.DataFrame(columns=["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle","endx","endy","endz","endPx","endPy","endPz","endt","endP","endE","EventID","TrackID","RunID"])

        for eventID, eventdf in df_stop.groupby('EventID'):
            current_event_initial = df_initial[df_initial["EventID"] == eventID]
            current_event_newparticles = df_newparticles[df_newparticles["EventID"] == eventID]
            for trackID, trackdf in eventdf.groupby("TrackID"):
                try:
                    initial = current_event_initial[current_event_initial["TrackID"] == trackID]
                    if not initial.shape[0] == 1: continue
                    
                    end = trackdf.iloc[0]

                    df_copy = current_event_newparticles.copy()
                    df_copy["x"] -= end.x
                    df_copy["y"] -= end.y
                    df_copy["z"] -= end.z
                    df_copy["t"] -= end.t
                    df_copy["r2"] = (df_copy["x"]**2) + (df_copy["y"])**2 + (df_copy["z"])**2 + (df_copy["t"])**2
                    df_copy = df_copy.sort_values("r2")
                    photons = df_copy.iloc[0] + df_copy.iloc[1]

                    end.Px = photons.Px
                    end.Py = photons.Py
                    end.Pz = photons.Pz
                    end_p  = math.sqrt(df_copy.iloc[0].Px**2 + df_copy.iloc[0].Py**2 + df_copy.iloc[0].Pz**2)
                    end_p += math.sqrt(df_copy.iloc[1].Px**2 + df_copy.iloc[1].Py**2 + df_copy.iloc[1].Pz**2)
                    end_e = 1000.0*end_p - 2*510.998950

                    if end.z > 9.97 and end.z < 10.13:
                        print(end_e,end.z)

                    initial_p = math.sqrt(initial.iloc[0].Px**2 + initial.iloc[0].Py**2 + initial.iloc[0].Pz**2)
                    initial_e = momentum_to_ke(initial_p)
                    initial_angle = math.acos(initial.iloc[0].Pz / initial_p) * 180.0/math.pi

                    df.loc[len(df)] = [ initial.iloc[0].x, initial.iloc[0].y, initial.iloc[0].z, initial.iloc[0].Px, initial.iloc[0].Py, initial.iloc[0].Pz, initial_p, initial_e, initial_angle, end.x, end.y, end.z, end.Px, end.Py, end.Pz, end.t, end_p, end_e, eventID, trackID, 0]
                    
                except IndexError as e:
                    pass

        if f"Out{threadnumber}.dat" in os.listdir(out_dir):
            main_df = pd.read_parquet(f"{out_dir}/Out{threadnumber}.dat")
            df["RunID"] = main_df['RunID'].max() + 1
            df = pd.concat([main_df,df],ignore_index=True)

        for c in ["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle","endx","endy","endz","endPx","endPy","endPz","endt","endP","endE"]:
            df[c] = df[c].astype('float32')
        
        for c in ["EventID","TrackID","RunID"]:
            df[c] = df[c].astype('Int16')

        df.to_parquet(f"{out_dir}/Out{threadnumber}.dat",engine="pyarrow",compression="brotli",compression_level=10,index=False)

        os.remove(f"DetKill{threadnumber}.txt")
        os.remove(f"DetNeg5{threadnumber}.txt")
        os.remove(f"NewParticles{threadnumber}.txt")
        
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