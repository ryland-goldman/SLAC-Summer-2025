import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize=(18, 12))

for i in range(0,9):
    try:
        # Load the file, skipping the comment lines
        data = []
        with open(f"Track{i}.txt", "r") as f:
            for line in f:
                if not line.startswith("#") and not line.strip()=="":
                    data.append([float(x) for x in line.split()])

        data = np.array(data)
        x = data[:, 0] * 1000
        z = data[:, 2] * 1000
        t = data[:, 6]

        mask = (x <= 100) & (z <= 100)
        x = x[mask]
        z = z[mask]
        t = t[mask]

        ax = axs[i//3, i%3]
        sc = ax.scatter(x, z, c=t, cmap='viridis', s=10)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("z [µm]")
        #ax.set_title("Particle Trajectory (x-z cross section)")
        circle = plt.Circle((0, 10), 10, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
    except Exception as e:
        print(f"{i}: {e}")

fig.colorbar(sc, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label="Time [ns]")
plt.show()