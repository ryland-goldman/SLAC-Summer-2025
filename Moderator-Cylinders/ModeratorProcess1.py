import math
import numpy as np
import os
import pandas as pd

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

def run_sum(threadnumber):
    print(threadnumber)
    try:
        df_initial = pd.read_csv(f"ModeratorOutC{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df_stop = pd.read_csv(f"ModeratorOut{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
        df_newparticles = pd.read_csv(f"ModeratorOutB{threadnumber}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

        df_initial = df_initial[df_initial["PDGid"] == "-11"]
        df_stop = df_stop[df_stop["PDGid"] == "-11"]
        df_newparticles = df_newparticles[df_newparticles["PDGid"] == "22"]
        df_initial = df_initial.drop('PDGid', axis=1)
        df_stop = df_stop.drop('PDGid', axis=1)
        df_newparticles = df_newparticles.drop('PDGid', axis=1)

        df = pd.DataFrame(columns=["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle","endx","endy","endz","endt","EventID","TrackID","RunID"])
        reflect_df = pd.DataFrame(columns=["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle","EventID","TrackID","RunID"])

        for eventID, eventdf in df_stop.groupby('EventID'):
            current_event_initial = df_initial[df_initial["EventID"] == eventID]
            current_event_newparticles = df_newparticles[df_newparticles["EventID"] == eventID]
            for trackID, trackdf in eventdf.groupby("TrackID"):
                try:                    
                    initial = current_event_initial[current_event_initial["TrackID"] == trackID]

                    initial_p = math.sqrt(initial.iloc[0].Px**2 + initial.iloc[0].Py**2 + initial.iloc[0].Pz**2)
                    initial_e = momentum_to_ke(initial_p)
                    initial_angle = math.acos(initial.iloc[0].Pz / initial_p) * 180.0/math.pi

                    if not len(initial) == 1:
                        reflect_df.loc[len(df)] = [ initial.iloc[0].x, initial.iloc[0].y, initial.iloc[0].z, initial.iloc[0].Px, initial.iloc[0].Py, initial.iloc[0].Pz, initial_p, initial_e, initial_angle, eventID, trackID, 0]
                        continue
                    end = trackdf.iloc[0]

                    df_copy = current_event_newparticles.copy()
                    df_copy["x"] -= end.x
                    df_copy["y"] -= end.y
                    df_copy["z"] -= end.z
                    df_copy["t"] -= end.t
                    df_copy["r2"] = (df_copy["x"]**2) + (df_copy["y"])**2 + (df_copy["z"])**2 + (df_copy["t"])**2
                    min_r2 = np.min(df_copy["r2"])

                    if np.isclose(min_r2, 0.0, atol=1e-3):
                        continue

                    df.loc[len(df)] = [ initial.iloc[0].x, initial.iloc[0].y, initial.iloc[0].z, initial.iloc[0].Px, initial.iloc[0].Py, initial.iloc[0].Pz, initial_p, initial_e, initial_angle, end.x, end.y, end.z, end.t, eventID, trackID, 0]
                    
                except IndexError as e:
                    pass

        for c in ["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle"]:
            reflect_df[c] = reflect_df[c].astype('float32')
        for c in ["initialx","initialy","initialz","initialPx","initialPy","initialPz","initialP","initialE","initialAngle","endx","endy","endz","endt"]:
            df[c] = df[c].astype('float32')
        
        for c in ["EventID","TrackID","RunID"]:
            df[c] = df[c].astype('Int32')
            reflect_df[c] = reflect_df[c].astype('Int32')

        df.to_parquet(f"OutB{threadnumber}.dat",engine="pyarrow",compression="brotli",compression_level=10,index=False)
        reflect_df.to_parquet(f"Out_r{threadnumber}.dat",engine="pyarrow",compression="brotli",compression_level=10,index=False)
        print(df)

        #os.remove(f"ModeratorOut{threadnumber}.txt")
        #os.remove(f"ModeratorOutB{threadnumber}.txt")
        
    except Exception as e:
        print(f"Running iteration {i} (Thread {threadnumber})... failed with exception",e)

#import sys
#run_sum(sys.argv[1])
#sys.exit()

for i in range(1,193):
    print(i)
    run_sum(i)
