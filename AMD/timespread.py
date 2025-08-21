import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

m = 0.511

df = pd.read_csv(f"TargetOutFiltered.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df1 = pd.read_csv(f"LBandOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df2 = pd.read_csv(f"AMDOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df2["initialE"] = np.sqrt(df2["Px"]**2 + df2["Py"]**2 + df2["Pz"]**2 + m**2) - m
df3 = df2[df2["initialE"] < 0.5]


 # Filter for positrons (PDGid == -11)
positrons = df[df["PDGid"] == "-11"]
std_t = positrons["t"].std()
mean_t = positrons["t"].mean()

# Positron mass in MeV/c^2
m_e = 0.511

# Compute total relativistic energy
p_squared = positrons["Px"]**2 + positrons["Py"]**2 + positrons["Pz"]**2
energy = np.sqrt(p_squared + m_e**2) - m_e


# Plot 2D histogram (E vs t)
plt.figure(figsize=(8, 5))
plt.hist2d(
    positrons["t"], energy, bins=200,
    range=[[mean_t - 2*std_t, mean_t + 2*std_t], [energy.min(), energy.max()]],
    cmap='viridis'
)
plt.xlabel("t (ns)")
plt.ylabel("Relativistic Energy E (MeV)")
plt.title("Relativistic Energy vs Time for Positrons")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()

# Repeat for df1
positrons1 = df1[df1["PDGid"] == "-11"]
std_t1 = positrons1["t"].std()
mean_t1 = positrons1["t"].mean()
p_squared1 = positrons1["Px"]**2 + positrons1["Py"]**2 + positrons1["Pz"]**2
energy1 = np.sqrt(p_squared1 + m_e**2)- m_e

plt.figure(figsize=(8, 5))
plt.hist2d(
    positrons1["t"], energy1, bins=200,
    range=[[mean_t1 - 2*std_t1, mean_t1 + 2*std_t1], [energy1.min(), energy1.max()]],
    cmap='viridis'
)
plt.xlabel("t (ns)")
plt.ylabel("Relativistic Energy E (MeV)")
plt.title("Relativistic Energy vs Time for Positrons (df1)")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()

# Repeat for df2
positrons2 = df2[df2["PDGid"] == "-11"]
std_t2 = positrons2["t"].std()
mean_t2 = positrons2["t"].mean()
p_squared2 = positrons2["Px"]**2 + positrons2["Py"]**2 + positrons2["Pz"]**2
energy2 = np.sqrt(p_squared2 + m_e**2)- m_e

plt.figure(figsize=(8, 5))
plt.hist2d(
    positrons2["t"], energy2, bins=200,
    range=[[mean_t2 - 2*std_t2, mean_t2 + 2*std_t2], [energy2.min(), energy2.max()]],
    cmap='viridis'
)
plt.xlabel("t (ns)")
plt.ylabel("Relativistic Energy E (MeV)")
plt.title("Relativistic Energy vs Time for Positrons (df2)")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()

# Repeat for df3
positrons3 = df3[df3["PDGid"] == "-11"]
std_t3 = positrons3["t"].std()
mean_t3 = positrons3["t"].mean()
p_squared3 = positrons3["Px"]**2 + positrons3["Py"]**2 + positrons3["Pz"]**2
energy3 = np.sqrt(p_squared3 + m_e**2)- m_e

plt.figure(figsize=(8, 5))
plt.hist2d(
    positrons3["t"], energy3, bins=200,
    range=[[mean_t3 - 2*std_t3, mean_t3 + 5], [energy3.min(), energy3.max()]],
    cmap='viridis'
)
plt.xlabel("t (ns)")
plt.ylabel("Relativistic Energy E (MeV)")
plt.title("Relativistic Energy vs Time for Positrons (df3)")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.show()


print(f"Target: {std_t*1000} ps")
print(f"LBand: {std_t1*1000} ps")
print(f"AMD: {std_t2*1000} ps")
print(f"AMD (<500 keV): {std_t3*1000} ps")