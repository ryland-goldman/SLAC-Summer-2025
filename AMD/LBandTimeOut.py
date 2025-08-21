import numpy as np
import pandas as pd

t = []
t_lowenergy = []

for i in range(1,193):
    df = pd.read_csv(f"data-amd/LBandOut{i}.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
    t.extend(df["t"].tolist())

    m_e_MeV = 0.51099895  # electron mass in MeV/c^2
    p_squared = df["Px"]**2 + df["Py"]**2 + df["Pz"]**2
    E = np.sqrt(p_squared + m_e_MeV**2)

    t_lowenergy.extend(df.loc[E < 1, "t"].tolist())

 # end of for loop
t_min = min(t)
t = [ti - t_min for ti in t]
t_lowenergy = [ti - t_min for ti in t_lowenergy]
import matplotlib.pyplot as plt

bins = np.linspace(0,5,101)

fig, ax_hist = plt.subplots()
ax_hist.set_yscale('log')
ax_hist.hist(
    t,
    bins=bins,
    color='darkred',
    edgecolor='black',
    alpha=0.5,
    label='All Particles'
)

ax_hist.hist(
    t_lowenergy,
    bins=bins,
    color='steelblue',
    edgecolor='black',
    alpha=0.5,
    label='Filtered for KE < 1 MeV'
)
ax_hist.legend()
ax_hist.set_ylabel("Frequency")
ax_hist.set_xlabel("Time of Cavity Departure (ns)")
plt.show()