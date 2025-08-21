import pandas as pd
import numpy as np
from xopt import Evaluator
from xopt import VOCS
from xopt import AsynchronousXopt as Xopt
from xopt.generators import list_available_generators
from xopt.generators import get_generator
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from scipy.interpolate import interp1d
import math

import subprocess
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Thread‑pool executor for parallel evaluations
N_WORKERS = os.cpu_count() or 1
executor = ThreadPoolExecutor(max_workers=N_WORKERS)

# Thread‑number bookkeeping so each worker can pass a unique index
_thread_id_map: dict[int, int] = {}
_thread_lock = threading.Lock()

def _get_thread_num() -> int:
    """Return a stable 0‑based worker index for the current thread."""
    tid = threading.get_ident()
    with _thread_lock:
        if tid not in _thread_id_map:
            _thread_id_map[tid] = len(_thread_id_map)
        return _thread_id_map[tid]

fraction_stopped = np.array([7.86885246e-01, 7.36220472e-01, 8.03370787e-01, 8.28901734e-01,
       8.48021583e-01, 8.34603659e-01, 8.51183064e-01, 8.00569801e-01,
       7.01644101e-01, 5.60573165e-01, 4.39335394e-01, 3.13344887e-01,
       2.40242057e-01, 1.54988789e-01, 1.20009678e-01, 8.23844608e-02,
       5.49848943e-02, 3.73443983e-02, 2.94169456e-02, 2.14738897e-02,
       1.47329650e-02, 1.06184531e-02, 9.75265018e-03, 5.85807482e-03,
       6.31552356e-03, 6.48397357e-03, 4.23131171e-03, 6.14824092e-03,
       3.98996808e-03, 3.10625536e-03, 2.47882669e-03, 2.44274809e-03,
       1.40675241e-03, 2.04878049e-03, 1.82446706e-03, 2.16063880e-03,
       2.21811460e-03, 6.25335001e-04, 7.10164225e-04, 7.08027259e-04])
energy_bins = np.linspace(0, 2, len(fraction_stopped))

stopping_fraction_interp = interp1d(
    energy_bins, fraction_stopped, bounds_error=False,
    fill_value=(fraction_stopped[0], fraction_stopped[-1])
)

def estimate_particles_stopped(energies):
    energies = np.array(energies)
    fractions = stopping_fraction_interp(energies)
    return np.sum(fractions)*14

m=0.511
def evaluate_function(inputs: dict) -> dict:
    thread_num = _get_thread_num()
    cmd = [
        "/Applications/G4beamline-3.08.app/Contents/MacOS/g4bl",
        "../LBand.g4bl",
        f"T={thread_num}",
        f"TIncA={inputs['T1']}",
        f"TIncB={inputs['T2']}",
        f"TIncC={inputs['T3']}",
        f"TIncD={inputs['T4']}",
        f"TIncE={inputs['T5']}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    #print(result.stdout)
    df = pd.read_csv(
        f"LBandOut{thread_num}.txt", skiprows=1, sep=r'\s+',
        dtype={"x":np.float32,"y":np.float32,"z":np.float32,
               "Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,
               "PDGid":str,"EventID":np.uint32,"TrackID":np.uint16},
        usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"],
        on_bad_lines="skip",
        names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '),
        comment="#"
    )
    df = df[df["PDGid"] == "-11"]
    df["E"] = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2 + m**2) - m
    count = estimate_particles_stopped(df["E"])
    return {"c": count}


evaluator = Evaluator(
    function=evaluate_function,
    executor=executor,
    max_workers=N_WORKERS,
)
vocs = VOCS(
    variables={"T1": [0, 0.7],"T2": [0, 0.7],"T3": [0, 0.7],"T4": [0, 0.7],"T5": [0, 0.7]},
    objectives={"c": "MAXIMIZE"},
)
from xopt.generators.bayesian import ExpectedImprovementGenerator
generator = ExpectedImprovementGenerator(vocs=vocs)
generator.batch_size = N_WORKERS

data_file = "lband_optimizer_data.csv"
X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

# Resume from previous run if data exists, otherwise start fresh
if os.path.exists(data_file):
    df_existing = pd.read_csv(data_file)
    if not df_existing.empty:
        X.add_data(df_existing)
    else:
        X.random_evaluate(5)
else:
    X.random_evaluate(5)

# Kick off the asynchronous optimization (runs until interrupted)
X.run()