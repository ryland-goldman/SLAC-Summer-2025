import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

event_count = 20000

### Load Data

df = pd.read_csv("batchdata.csv")
df = df.pivot(index="Energy", columns="Thickness", values="Count")
data = df.to_numpy() / event_count



### Heatmap

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(data, cmap='viridis', aspect='auto', origin='upper')

x_labels = list(df.columns)
y_labels = list(df.index)

ax.set_xticks(np.arange(data.shape[1]))
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(np.arange(data.shape[0]))
ax.set_yticklabels(y_labels, fontsize=9)

ax.set_xlabel('Tungsten Thickness (mm)', labelpad=8, fontsize=11)
ax.set_ylabel('Energy (MeV)',   labelpad=8, fontsize=11)

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.80)
cbar.set_label('Positron Fraction')

plt.tight_layout()
plt.show()
