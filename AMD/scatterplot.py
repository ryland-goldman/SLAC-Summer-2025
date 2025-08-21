import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAME_COLOR_SCALE = True  # Set to False to use independent color scales

df_target = pd.read_csv(f"TargetOutFiltered.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")
df_amd = pd.read_csv(f"AMDOut.txt", skiprows=1, delim_whitespace=True, dtype={"x":np.float32,"y":np.float32,"z":np.float32,"Px":np.float32,"Py":np.float32,"Pz":np.float32,"t":np.float32,"PDGid":str,"EventID":np.uint32,"TrackID":np.uint16}, usecols=["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID"], on_bad_lines="skip", names='x y z Px Py Pz t PDGid EventID TrackID ParentID Weight'.split(' '), comment="#")

df_amd = df_amd[df_amd["PDGid"]=="-11"]

print(df_target.shape, df_amd.shape)

def angular_divergence(px, py, pz):
    p_total = np.sqrt(px**2 + py**2 + pz**2)
    theta_exact_rad = np.arccos(pz / p_total)
    theta_exact_mrad = theta_exact_rad * 1000
    return theta_exact_mrad


# Compute angular divergence for each dataset
div_target = angular_divergence(df_target["Px"], df_target["Py"], df_target["Pz"])
div_amd = angular_divergence(df_amd["Px"], df_amd["Py"], df_amd["Pz"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.rcParams.update({'font.size': 14})

vmin = min(div_target.min(), div_amd.min()) if SAME_COLOR_SCALE else None
vmax = 1000 if SAME_COLOR_SCALE else None

# First scatter (Target)
sc0 = axes[0].scatter(
    df_target["x"], df_target["y"],
    c=div_target, cmap='RdBu_r', s=1, vmin=vmin, vmax=vmax, alpha=0.5
)
axes[0].set_title("Before AMD")
axes[0].set_xlabel("x (mm)", fontsize=16)
axes[0].set_ylabel("y (mm)", fontsize=16)
cbar0 = plt.colorbar(sc0, ax=axes[0])
cbar0.set_label("Angular Divergence (mrad)")
axes[0].set_xlim(-75, 75)
axes[0].set_ylim(-75, 75)

# Second scatter (AMD)
sc1 = axes[1].scatter(
    df_amd["x"], df_amd["y"],
    c=div_amd, cmap='RdBu_r', s=1, vmin=vmin, vmax=vmax, alpha=0.5
)
axes[1].set_title("After AMD")
axes[1].set_xlabel("x (mm)", fontsize=16)
axes[1].set_ylabel("y (mm)", fontsize=16)
axes[1].set_xlim(-75, 75)
axes[1].set_ylim(-75, 75)
cbar1 = plt.colorbar(sc1, ax=axes[1])
cbar1.set_label("Angular Divergence (mrad)")

plt.tight_layout()
plt.savefig("angular_divergence_scatter.png", dpi=300)

# Histogram of Angular Divergence (Target and AMD)
fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, 500, 500)

ax_hist.hist(
    div_target,
    bins=bins,
    color='darkred',
    edgecolor='black',
    alpha=0.5,
    label='After Target',
    weights=np.ones_like(div_target) / len(div_target)
)
ax_hist.hist(
    div_amd,
    bins=bins,
    color='steelblue',
    edgecolor='black',
    alpha=0.5,
    label='After AMD',
    weights=np.ones_like(div_amd) / len(div_amd)
)

ax_hist.set_xlabel("Angular Divergence (mrad)", fontsize=16)
ax_hist.set_ylabel("Density of Particles (1/mrad)", fontsize=16)
ax_hist.legend()

plt.tight_layout()
plt.savefig("angular_divergence_histogram.png", dpi=300)