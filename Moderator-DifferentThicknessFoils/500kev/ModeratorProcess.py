import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def makhovian_distribution(z, z0, m):
    return (m * z**(m - 1) / z0**m) * np.exp(-(z / z0)**m) * 1e4

def ramped_makhovian(z, z0, m, k, L):
    return makhovian_distribution(z, z0, m) * (1 - np.exp( -(L - z) / k))

def load_stopping_distribution(directory, thickness_um, line=False):
    files = [f"{directory}/{f}" for f in os.listdir(directory) if f.startswith("Out") and f.endswith(".dat") and 'r' not in f]
    end_z = []
    all_initial_angle = []
    for file in files:
        try:
            df = pd.read_parquet(file)
        except Exception:
            continue
        df["initialE"] = np.sqrt(df["initialP"]**2 + 0.511**2) - 0.511
        all_initial_angle += list(df["initialAngle"])
        df = df[(df["endz"] > 0.9) & (df["endz"] < 1.1)]
        end_z += list(df["endz"])
    if not all_initial_angle:
        return (np.array([]), np.array([]), np.array([])) if not line else (np.array([]), np.array([]), np.array([]))
    end_z = np.array(end_z)
    end_z = (end_z - 1 + thickness_um / 2000) * 1000  # Convert to µm, centered on thickness
    bins = np.arange(0, thickness_um + 0.5, 0.5)
    counts, _ = np.histogram(end_z, bins=bins)
    counts = np.array(counts) / len(all_initial_angle) * 1e4 * 1/0.5
    if line:
        return bins[:-1], counts, end_z
    else:
        return bins[:-1], counts, end_z

# Load and plot
plt.figure(figsize=(8, 6))

color_map = {
    100: 'blue',
    75: 'orange',
    50: 'green',
    25: 'brown'
}

z0_100 = None
m_100 = None

for label, thickness in [("100 µm", 100), ("75 µm", 75), ("50 µm", 50), ("25 µm", 25)]:
    if os.path.isdir(f"Mod{thickness}um"):
        bins, counts, end_z_data = load_stopping_distribution(f"Mod{thickness}um", thickness)
        if len(counts) > 0:
            plt.bar(
                bins, counts, width=0.5, align='edge', label=label, alpha=0.6,
                edgecolor='black', color=color_map[thickness]
            )
            if thickness == 100:
                try:
                    popt, _ = curve_fit(makhovian_distribution, bins, counts, p0=[50, 2], maxfev=10000)
                    z_fit = np.linspace(0, 100, 500)
                    plt.plot(
                        z_fit, makhovian_distribution(z_fit, *popt),
                        label="Makhovian Fit (100 µm)", color='red', linewidth=2
                    )
                    print(f"Fitted parameters for 100 µm: z0 = {popt[0]:.2f}, m = {popt[1]:.2f}")
                    z0_100, m_100 = popt
                except RuntimeError:
                    print("Fit failed for 100 µm.")
            if thickness in [75, 50, 25]:
                if z0_100 is not None and m_100 is not None:
                    try:
                        popt, pcov = curve_fit(
                            lambda z, k: ramped_makhovian(z, z0_100, m_100, k, thickness),
                            bins, counts, p0=[5], maxfev=10000
                        )
                        perr = np.sqrt(np.diag(pcov))
                        k_val = popt[0]
                        k_err = perr[0]
                        z_fit = np.linspace(0, thickness, 500)
                        fit_values = ramped_makhovian(z_fit, z0_100, m_100, k_val, thickness)
                        plt.plot(
                            z_fit, fit_values,
                            label=f"Ramped Makhovian Fit ({label})",
                            linewidth=2, color=color_map[thickness]
                        )
                        print(f"Fitted parameter for {label}: k = {k_val:.2f} ± {k_err:.2f}")
                    except RuntimeError:
                        print(f"Fit failed for {label}.")
                else:
                    print(f"Skipping fit for {label} because 100 µm fit parameters are not available.")
        else:
            print(f"No valid data found in Mod{thickness}um.")
    else:
        print(f"Directory Mod{thickness}um not found.")

# Add fontsize to xlabel and ylabel
plt.xlabel("Penetration Depth (µm)", fontsize=14)
plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
#plt.title("Stopping Distributions for Different Foil Thicknesses")
#plt.suptitle("Using 500 keV Monoenergetic Beam", fontsize=10, y=0.95)
plt.legend(fontsize=12)
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], fontsize=12)

plt.show()
