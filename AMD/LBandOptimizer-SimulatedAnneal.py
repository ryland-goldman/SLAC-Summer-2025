import numpy as np
import pandas as pd
import os
import subprocess
from scipy.interpolate import interp1d

def rand_unit_vec():
    v = np.random.normal(size=5)
    return v / np.linalg.norm(v)

import sys
n = int(sys.argv[1])
conf = [[1, 0.10129373353897175, 0.6125460992114393, 0.2848487376150007, 0.007828368218380066, 0.6763376729304624, 141], [10, 0.7616705393229455, 0.11305016772431405, 0.2496514099501616, 0.43302654875223623, 0.39071634446258324, 43], [11, 0.23868474918071617, 0.22132357881082446, 0.04531575790179056, 0.3432929993290581, 0.2600465143050439, 144], [12, 0.38345279304072033, 0.35395701742656976, -0.0028364568052885175, 0.3358787842993156, 0.2272677172382287, 47], [2, 0.7122334697207643, 0.1421264838879278, 0.14295209410068316, 0.3674874966625607, 0.27267658931505323, 207], [3, 0.3660644627430134, 0.23467766261491346, 0.01644887740829168, 0.4496251501707291, 0.24617331770152792, 218], [4, 0.09293236598687031, -0.020771636290142455, 0.6682424417719757, 0.5526176525428973, 0.5473191998900407, 99], [5, 0.15713928617837877, 0.5868678722567605, 0.6047141594007506, -0.028907011071096705, 0.6608119702178836, 77], [6, 0.3914467962267991, 0.36735350474484857, 0.6471715277363888, 0.48754294083787025, 0.21301234811123693, 228], [7, 0.29811903410287927, 0.4565963138628748, 0.042209812561958306, 0.022366269143090478, 0.0681147431043738, 143], [8, 0.377679721214223, 0.31113376178506086, 0.6803148494934623, 0.496949049438031, 0.1996178191760017, 204], [9, 0.37176002976924527, 0.5020490889348804, 0.13550837705762517, 0.11412194204704809, 0.21919846432706486, 211]]

conf_dict = {row[0]: row for row in conf}
conf = conf_dict.get(n)

start = 0.3
params = np.array([start, start+0.35, start, start+0.35, start ]) #
#params = np.random.rand(5) * 0.700
result_df = pd.DataFrame(columns=["T1", "T2", "T3", "T4", "T5", "count","accepted","step","temp","stepsize"])

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
    energies = np.asarray(energies, dtype=float)
    #energies = np.array(energies)
    #fractions = stopping_fraction_interp(energies)
    #return np.sum(fractions)
    # Count particles with kinetic energy below 0.2 MeV (200 keV)
    return np.count_nonzero(energies < 0.2)

m=0.511
prev_counts = 0
first = True
steps = 0

params = [conf[1], conf[2], conf[3], conf[4], [conf[5]]]
steps = conf[6]
while True:
    steps += 1
    stepsize = 0.1 * np.exp(-0.005*steps)
    temperature = 15 * np.exp(-0.005*steps)

    next_step = rand_unit_vec() * stepsize
    if not first: params += next_step
    first = False

    result = subprocess.run(["./run-lband.sh",str(params[0]),str(params[1]),str(params[2]),str(params[3]),str(params[4]),str(n)], capture_output=True, text=True)
    df = pd.read_csv(
        f"LBandOut.{n}.txt", skiprows=1, sep=r'\s+',
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

    delta_cost = prev_counts - count
    p_accept_new = np.exp(- delta_cost / temperature)
    rejected = p_accept_new < np.random.uniform()

    result_df.loc[len(result_df)] = list(params) + [count] + [not rejected] + [steps, temperature, stepsize]
    result_df.to_csv(f"simulated_anneal_results_{n}.csv", index=False)

    if rejected:
        params -= next_step
    else:
        prev_counts = count
        
    

    params = np.where(params > 0.769, params - 0.769, params)
    params = np.where(params < 0.000, params + 0.769, params)