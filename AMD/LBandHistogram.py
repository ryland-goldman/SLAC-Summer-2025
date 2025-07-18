import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1 = pd.read_csv(f"AMDOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df2 = pd.read_csv(f"LBandOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

df1 = df1[df1["PDGid"]=="-11"]
df2 = df2[df2["PDGid"]=="-11"]

print(np.mean(df1["t"]))

m = 0.511  # MeV/c^2
df1["E"] = np.sqrt(df1["Px"]**2 + df1["Py"]**2 + df1["Pz"]**2 + m**2)
df2["E"] = np.sqrt(df2["Px"]**2 + df2["Py"]**2 + df2["Pz"]**2 + m**2)

count1 = ((df1["E"] >= 0) & (df1["E"] <= 5)).sum()
count2 = ((df2["E"] >= 0) & (df2["E"] <= 5)).sum()
print(f"Number of positrons in df1 with 0 <= E <= 5 MeV: {count1}")
print(f"Number of positrons in df2 with 0 <= E <= 5 MeV: {count2}")


bins = np.linspace(0,10,101)
plt.hist(df1["Pz"], bins=bins, color='b', alpha=0.6, label='Before L-Band Cavity')
plt.xlabel('Pz (MeV/c)')
plt.ylabel('Count')
plt.hist(df2["Pz"], bins=bins, color='r', alpha=0.6, label='After L-Band Cavity')
plt.legend()
plt.show()