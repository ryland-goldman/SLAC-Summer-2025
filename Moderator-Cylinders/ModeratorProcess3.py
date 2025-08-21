import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

loc = f"."

p = os.listdir(loc)
all_files = [f"{loc}/{a}" for a in p if a[0:3]=="Out" and a[-3:]=="dat"]
files = [f for f in all_files if not 'r' in f]
files = [f for f in files if not 'B' in f]

end_x = []
end_y = []
end_z = []
initial_angle = []

def run_file1(file):
    try: 
        df = pd.read_parquet(file)
    except Exception:
        print(f"Err {file}")
        return
    if not "endz" in df.columns:
        print(df.columns,file)
        return
    loc_initial_angle = list(df["initialAngle"])
    df = df[df["endz"] > 0]
    df = df[df["endz"] < 1]
    loc_end_x = list(df["endx"])
    loc_end_y = list(df["endy"])
    loc_end_z = list(df["endz"])
    global end_x, end_y, end_z, initial_angle
    end_x = end_x + loc_end_x
    end_y = end_y + loc_end_y
    end_z = end_z + loc_end_z
    initial_angle = initial_angle + loc_initial_angle

for i in files: run_file1(i)

end_x = np.array(end_x)
end_y = np.array(end_y)
end_z = np.array(end_z)

end_z = np.abs(np.sqrt(np.pow(end_x,2) + np.pow(end_z-0.015,2))-0.01)

thickness = 0.020
end_z *= 1000
counts, bins = np.histogram(end_z,np.linspace(0,thickness*1000,50))
counts = np.array(counts)
counts = counts / len(initial_angle)
counts *= 1e4
counts *= 50/(thickness*1000)
plt.stairs(counts, bins, label="Cylinder Moderator")
plt.xlabel("Distance to Surface (µm)")
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")




loc = f"."

p = os.listdir(loc)
all_files = [f"{loc}/{a}" for a in p if a[0:3]=="Out" and a[-3:]=="dat"]
files = [f for f in all_files if not 'r' in f]
files = [f for f in files if 'B' in f]

end_x = []
end_y = []
end_z = []
initial_angle = []

def run_file2(file):
    try: 
        df = pd.read_parquet(file)
    except Exception:
        print(f"Err {file}")
        return
    loc_initial_angle = list(df["initialAngle"])
    df = df[df["endz"] > 0]
    df = df[df["endz"] < 1]
    loc_end_x = list(df["endx"])
    loc_end_y = list(df["endy"])
    loc_end_z = list(df["endz"])
    global end_x, end_y, end_z, initial_angle
    end_x = end_x + loc_end_x
    end_y = end_y + loc_end_y
    end_z = end_z + loc_end_z
    initial_angle = initial_angle + loc_initial_angle

for i in files: run_file2(i)

end_x = np.array(end_x)
end_y = np.array(end_y)
end_z = np.array(end_z)

end_z = np.abs(end_z-0.015)
end_z = np.abs(0.01-end_z)

thickness = 0.020
end_z *= 1000
counts, bins = np.histogram(end_z,np.linspace(0,thickness*1000,50))
counts = np.array(counts)
counts = counts / len(initial_angle)
counts *= 1e4
counts *= 50/(thickness*1000)
plt.stairs(counts, bins, label="Foil Moderator")
plt.xlabel("Distance to Surface (µm)")
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
end_x = []
end_y = []
end_z = []
initial_angle = []
for i in all_files:
    if 'r' not in i and 'B' not in i:
        run_file1(i)

end_x = np.array(end_x)
end_y = np.array(end_y)
end_z = np.array(end_z)

plt.scatter(end_x * 1000, end_z * 1000, s=1, alpha=0.5)
plt.title("Cylinder")
plt.xlabel("x (µm)")
plt.axis('equal')
plt.grid(True)
circ = plt.Circle((0, 15), 10, linewidth=1, edgecolor='red', facecolor='none')
plt.gca().add_patch(circ)
plt.xlim(-20, 20)
plt.ylim(-5, 35)
plt.suptitle("Final Positions of Stopped Positrons")
plt.tight_layout()
plt.show()