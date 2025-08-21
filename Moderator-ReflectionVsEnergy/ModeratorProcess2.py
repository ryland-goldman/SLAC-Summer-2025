"""
ModeratorProcess2.py

Uses .dat files

This script analyzes positron transport/parquet outputs to compute stop/reflection
statistics, geometry-based escape probabilities, and a set of diagnostic plots.
"""

from __future__ import annotations

# ===== Standard library imports =====
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import warnings
import argparse

# ===== Third-party imports =====
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  # kept to preserve parity with original imports
import scipy.constants as const

# ===== Display / warnings =====
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 150)
warnings.filterwarnings("ignore")

# ===== Constants =====
KE_FILTER_MEV: float = 100.0  # Upper cut on initial kinetic energy (MeV)
DATA_DIR: str = "data-dir"
THERMALIZATION_THRESHOLD_GEV: float = 0.001  # Unused (kept for parity)

# Geometry (mm) -- Simple cylindrical moderator
OUTER_RADIUS: float = 100.0
LENGTH: float = 0.05
Z_POSITION: float = 1
MAX_STEP: float = 0.001

# Re-emission attenuation length (mm)
LAMBDA_MM: float = 5.5e-5

# Histogram settings
KE_BINS_STOP: np.ndarray = np.linspace(0, 2, 50)
ANGLE_BINS_STOP: np.ndarray = np.linspace(0, 90, 18)


# ===== Data containers =====
@dataclass
class Tally:
    """Container for event tallies and kinematic collections."""

    # Counts
    n_stopped: int = 0
    n_transmitted: int = 0
    n_annihilated: int = 0
    n_reflected: int = 0

    # Collections (lists to avoid repeated numpy reallocations on extend)
    end_x: List[float] = None
    end_y: List[float] = None
    end_z: List[float] = None
    initial_p_stopped: List[float] = None
    initial_angle_stopped: List[float] = None
    all_initial_p: List[float] = None
    all_initial_angle: List[float] = None

    def __post_init__(self) -> None:
        self.end_x = [] if self.end_x is None else self.end_x
        self.end_y = [] if self.end_y is None else self.end_y
        self.end_z = [] if self.end_z is None else self.end_z
        self.initial_p_stopped = [] if self.initial_p_stopped is None else self.initial_p_stopped
        self.initial_angle_stopped = [] if self.initial_angle_stopped is None else self.initial_angle_stopped
        self.all_initial_p = [] if self.all_initial_p is None else self.all_initial_p
        self.all_initial_angle = [] if self.all_initial_angle is None else self.all_initial_angle


# ===== Utilities =====

def list_event_files(root: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (stopped, reflected, annihilated) parquet file paths from directory.

    We mimic original logic: consider names starting with "Out" and ending in "dat",
    then split into three groups by substrings: normal, contains 'r', contains '_a'.
    """
    filenames = os.listdir(root)
    all_files = [f"{root}/{name}" for name in filenames if name.startswith("Out") and name.endswith("dat")]
    stopped = [f for f in all_files if "_r" not in f and "_a" not in f]
    reflected = [f for f in all_files if "_r" in f]
    annihilated = [f for f in all_files if "_a" in f]
    return stopped, reflected, annihilated


def load_parquet_safe(path: str) -> pd.DataFrame | None:
    """Read a parquet file; on failure, report and return None."""
    try:
        df = pd.read_parquet(path)
        return df
    except Exception:
        # Suppress printing on read failure
        return None


def add_initial_energy_mev(df: pd.DataFrame) -> pd.DataFrame:
    """Compute kinetic energy from momentum (MeV) using E = sqrt(p^2 + m^2) - m.

    Assumes columns: 'initialP' (MeV/c).
    Adds/overwrites column 'initialE' (MeV).
    """
    if df is None:
        return df
    m_mev = 0.511
    df = df.copy()
    df["initialE"] = np.sqrt(df["initialP"] ** 2 + m_mev**2) - m_mev
    return df


def filter_ke(df: pd.DataFrame, ke_max_mev: float = KE_FILTER_MEV) -> pd.DataFrame:
    """Return df with initialE < ke_max_mev. Safe if df is None."""
    if df is None:
        return df
    return df[df["initialE"] < ke_max_mev]


# ===== Core processing =====

def process_directory(root: str, ke_filter_mev: float = KE_FILTER_MEV) -> Tally:
    """Load, filter, and tally events from parquet files under *root*.

    Returns a populated `Tally` object with counts and kinematic arrays.
    """
    files, files_r, files_a = list_event_files(root)
    tally = Tally()

    for f_stopped, f_ref, f_ann in zip(files, files_r, files_a):
        df_s = filter_ke(add_initial_energy_mev(load_parquet_safe(f_stopped)), ke_filter_mev)
        df_r = filter_ke(add_initial_energy_mev(load_parquet_safe(f_ref)), ke_filter_mev)
        df_a = filter_ke(add_initial_energy_mev(load_parquet_safe(f_ann)), ke_filter_mev)
        if df_s is None or df_r is None or df_a is None:
            # skip triplet if any is missing
            continue

        # Collect "all" initial distributions (before fate selection)
        tally.all_initial_p.extend(df_s["initialE"].tolist())
        tally.all_initial_angle.extend(df_s["initialAngle"].tolist())

        # Select stopped within moderator z-range (original cut)
        df_s_sel = df_s[(df_s["endz"] > 0.9) & (df_s["endz"] < 65)]

        tally.end_x.extend(df_s_sel["endx"].tolist())
        tally.end_y.extend(df_s_sel["endy"].tolist())
        tally.end_z.extend(df_s_sel["endz"].tolist())
        tally.initial_p_stopped.extend(df_s_sel["initialE"].tolist())
        tally.initial_angle_stopped.extend(df_s_sel["initialAngle"].tolist())
        tally.n_stopped += len(df_s_sel)

        # Reflected: contribute to "all" distributions as well
        tally.all_initial_p.extend(df_r["initialE"].tolist())
        tally.all_initial_angle.extend(df_r["initialAngle"].tolist())
        tally.n_reflected += len(df_r)

        # Annihilated: contribute to "all" distributions as well
        tally.all_initial_p.extend(df_a["initialE"].tolist())
        tally.all_initial_angle.extend(df_a["initialAngle"].tolist())
        tally.n_annihilated += len(df_a)  # FIXED: accumulate rather than overwrite

    # Compute transmitted after iterating all files
    tally.n_transmitted = (
        len(tally.all_initial_p) - tally.n_stopped - tally.n_annihilated - tally.n_reflected
    )
    return tally


# ===== Geometry / distances =====

def nearest_surface_distance_mm(end_x: np.ndarray, end_y: np.ndarray, end_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distances (mm) from points to nearest surface of a cylindrical moderator.

    Returns (dist_overall, dist_to_z_faces, dist_to_radial).
    """
    # Distance to cylindrical side
    dist_radial = np.maximum(0, OUTER_RADIUS - np.sqrt(end_x**2 + end_y**2))
    # Distance to z faces (centered at Z_POSITION, length LENGTH)
    dist_z = np.abs( (LENGTH / 2) - np.abs(end_z - Z_POSITION))
    # Overall distance is minimum of radial and z distances
    dist_overall = dist_z #np.minimum(dist_radial, dist_z)
    return dist_overall, dist_z, dist_radial



# ===== Analysis helpers =====


def fit_and_plot_stopping_distribution(end_z_mm: np.ndarray, *, z_center_mm: float = Z_POSITION, thickness_mm: float = LENGTH, lambda_infinite: bool = False, E: float | None = None) -> None:
    """Plot histogram of stopping depths and fit a ramped Makhovian curve.

    Depth is measured from the upstream face of the moderator.
    The fit uses parameters (A, m, z0_um, lam_um) with fixed L = thickness.
    """
    # Convert end positions to depth within the foil in micrometers
    L_um = thickness_mm * 1000.0
    z_front_mm = z_center_mm - 0.5 * thickness_mm
    depth_um = (end_z_mm - z_front_mm) * 1000.0

    # Keep only points that lie within [0, L]
    depth_um = depth_um[(depth_um >= 0.0) & (depth_um <= L_um)]
    if depth_um.size == 0:
        plt.figure(figsize=(6, 4))
        plt.title("No stops inside moderator")
        plt.xlabel("Depth (µm)")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.show()
        plt.close()
        return

    # Always show 50 bins, but dynamically zoom the x-range to where the
    # distribution lives. Keep y-units as counts per µm.
    n_bins = 50

    # Determine a viewing window up to the 99.5th percentile (with margin),
    # but never exceeding the foil thickness. Enforce a small minimum span
    # to avoid zero-width bins when everything piles up at the surface.
    p995 = np.percentile(depth_um, 99.5)
    x_max_view = float(min(L_um, max(1.1 * p995, 0.5)))  # µm

    edges = np.linspace(0.0, x_max_view, n_bins + 1)
    counts, _ = np.histogram(depth_um, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Convert to density so units are 1/µm regardless of the zoomed range
    density = counts / max(bin_width, 1e-30)

    # Ramped Makhovian model normalized on [0, L].
    # We fit shape parameters (m, z0_um, lam_um) and then scale by total counts.
    N_total = float(np.sum(counts))

    def model_counts(z_um, m, z0_um, lam_um):
        z = np.clip(z_um, 0.0, x_max_view)
        z_safe = np.where(z > 0, z, 1e-12)
        base = (m * (z_safe ** (m - 1))) / (np.maximum(z0_um, 1e-12) ** m) * np.exp(- (z_safe / np.maximum(z0_um, 1e-12)) ** m)
        if lambda_infinite:
            ramp = 1.0
        else:
            ramp = 1.0 - np.exp(- (L_um - z) / np.maximum(lam_um, 1e-12))
        unnorm_pdf = base * ramp
        # Normalize over [0, x_max_view]
        z_grid = np.linspace(0.0, x_max_view, 2000)
        z_grid_safe = np.where(z_grid > 0, z_grid, 1e-12)
        base_g = (m * (z_grid_safe ** (m - 1))) / (np.maximum(z0_um, 1e-12) ** m) * np.exp(- (z_grid_safe / np.maximum(z0_um, 1e-12)) ** m)
        if lambda_infinite:
            ramp_g = 1.0
        else:
            ramp_g = 1.0 - np.exp(- (L_um - z_grid) / np.maximum(lam_um, 1e-12))
        norm = np.trapz(base_g * ramp_g, z_grid)
        pdf = (unnorm_pdf / np.maximum(norm, 1e-30))  # 1/µm
        return N_total * pdf  # counts per µm (density)

    if lambda_infinite:
        # Fit only m and z0_um, hold ramp=1
        m0 = 1.9
        z00 = 0.7 * L_um
        p0 = (m0, z00)
        bounds = (
            (0.1, 1e-3),           # m, z0_um lower bounds
            (5.0, 10.0 * L_um)     # upper bounds
        )

        def model_counts_2(z_um, m, z0_um):
            return model_counts(z_um, m, z0_um, 1.0)  # lam ignored when lambda_infinite=True

        mask = density > 0
        x_fit = centers[mask]
        y_fit = density[mask]

        try:
            popt, pcov = curve_fit(model_counts_2, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            # Suppress fit parameter printing
        except Exception:
            popt = None
    else:
        m0 = 1.9
        z00 = 0.7 * L_um
        lam0 = 0.2 * L_um
        p0 = (m0, z00, lam0)
        bounds = (
            (0.5, 1e-3, 1e-3),           # m, z0_um, lam_um lower bounds
            (5.0, 10.0 * L_um, 10.0 * L_um)  # upper bounds
        )

        mask = density > 0
        x_fit = centers[mask]
        y_fit = density[mask]

        try:
            popt, pcov = curve_fit(model_counts, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            # Suppress fit parameter printing
        except Exception:
            popt = None

    # Plot
    plt.figure(figsize=(7, 5))
    # Plot as a step histogram in density units to remove bar seam artifacts
    plt.stairs(density, edges, alpha=0.6)
    if popt is not None:
        z_dense = np.linspace(0.0, x_max_view, 400)
        if lambda_infinite:
            y_dense = (lambda z: model_counts(z, popt[0], popt[1], 1.0))(z_dense)
        else:
            y_dense = model_counts(z_dense, *popt)
        plt.plot(z_dense, y_dense, linewidth=2.5)
    plt.xlim(0.0, x_max_view)
    plt.xlabel("Depth (µm)")
    plt.ylabel("Counts per µm")
    title_E = f" (E={E:.3g} MeV)" if E is not None else ""
    plt.title("Stopping distribution with ramped Makhovian fit" + title_E)
    plt.tight_layout()
    plt.savefig(f"stopping_distribution_{E}.png", dpi=300)
    plt.close()

def simulate_reemission(end_x: np.ndarray, end_y: np.ndarray, end_z: np.ndarray, out_prob: np.ndarray,
                        n_trials: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Perform simple Monte Carlo sampling of re-emitted endpoints.

    Returns diff_x, diff_y, diff_z from the last accepted trial (to mimic original
    behavior), and a dictionary with mean std/rms stats across trials.
    """
    rng = np.random.default_rng()

    stds = [[], [], [], []]  # x, y, z, r
    rmss = [[], [], [], []]

    diff_x = diff_y = diff_z = np.array([])
    n_diff_last = 0

    for _ in range(n_trials):
        mask = out_prob > rng.uniform(size=out_prob.shape)
        dx = end_x[mask]
        dy = end_y[mask]
        dz = end_z[mask]

        # Guard against wild outliers in y (keep legacy cut)
        if dy.size == 0 or np.std(dy) > 20:
            continue

        r = np.sqrt(dx**2 + dy**2)
        stds[0].append(np.std(dx)); stds[1].append(np.std(dy)); stds[2].append(np.std(dz)); stds[3].append(np.std(r))
        rmss[0].append(np.sqrt(np.mean(dx**2))); rmss[1].append(np.sqrt(np.mean(dy**2)))
        rmss[2].append(np.sqrt(np.mean(dz**2))); rmss[3].append(np.sqrt(np.mean(r**2)))

        # Keep last trial’s sample to plot/print like the original script
        diff_x, diff_y, diff_z = dx, dy, dz
        n_diff_last = dx.size

    stats = {
        "sum_p": float(np.sum(out_prob)),
        "n_diff_last": int(n_diff_last),
        "std_mean": tuple(float(np.mean(s)) for s in stds),
        "rms_mean": tuple(float(np.mean(r)) for r in rmss),
    }
    return diff_x, diff_y, diff_z, stats


def compute_brightness(out_prob: np.ndarray, std_means: Tuple[float, float, float]) -> Tuple[float, float]:
    """Compute brightness and a simple propagated error bar (original formula)."""
    N = float(np.sum(out_prob))
    dN = math.sqrt(max(N, 1.0))

    # Momentum spread (placeholder matches original constant assumption)
    E = 2.59 * const.elementary_charge  # kept for parity; not used numerically downstream
    p = 1.0  # original script used p=1
    sigma_p = p / math.sqrt(3)

    sigma_x, sigma_y, sigma_z = std_means
    dsigma_x = 0.0
    dsigma_y = 0.0
    dsigma_z = 0.0

    brightness = N / (((2 * const.pi) ** 3) * sigma_p**3 * sigma_x * sigma_y * sigma_z)
    error_bar = brightness * math.sqrt((dN / max(N, 1.0)) ** 2 + (dsigma_x / max(sigma_x, 1e-12)) ** 2 +
                                       (dsigma_y / max(sigma_y, 1e-12)) ** 2 + (dsigma_z / max(sigma_z, 1e-12)) ** 2)
    return float(brightness), float(error_bar)


# ===== Plotting =====

def plot_3d_points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z)
    plt.show()


def plot_penetration_hist(end_z_mm: np.ndarray, all_initial_angle: Iterable[float], thickness_mm: float = 0.05) -> None:
    end_z_um = (end_z_mm - 1 + thickness_mm / 2) * 1000
    counts, bins = np.histogram(end_z_um, np.linspace(0, thickness_mm * 1000, 50))
    counts = counts.astype(float)
    counts = counts / max(len(list(all_initial_angle)), 1)
    counts *= 1e4
    counts *= 50 / (thickness_mm * 1000)
    plt.stairs(counts, bins)
    plt.xlabel("Penetration Depth (µm)")
    plt.ylabel("Positrons Stopped / $10^4$ (µm$^{-1}$)")
    plt.tight_layout()
    plt.show()


def plot_stop_fraction_1d(initial_p_stopped: Iterable[float], all_initial_p: Iterable[float],
                          initial_angle_stopped: Iterable[float], all_initial_angle: Iterable[float]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    c1, b1 = np.histogram(list(initial_p_stopped), bins=np.linspace(0, 2, 50))
    c2, b2 = np.histogram(list(all_initial_p), bins=np.linspace(0, 2, 50))
    ratio = np.divide(c1, c2, out=np.zeros_like(c1, dtype=float), where=c2 != 0)
    axs[0].stairs(ratio, b1)
    axs[0].set_xlabel("Initial Kinetic Energy (KE)")
    axs[0].set_ylabel("Fraction Stopped")

    c1, b1 = np.histogram(list(initial_angle_stopped), bins=np.linspace(0, 90, 90))
    c2, b2 = np.histogram(list(all_initial_angle), bins=np.linspace(0, 90, 90))
    ratio = np.divide(c1, c2, out=np.zeros_like(c1, dtype=float), where=c2 != 0)
    axs[1].stairs(ratio, b1)
    axs[1].set_xlabel("Initial Angle (Deg)")
    axs[1].set_ylabel("Fraction Stopped")

    plt.tight_layout(); plt.show()


def plot_stop_fraction_2d(all_initial_p: Iterable[float], all_initial_angle: Iterable[float],
                          stopped_p: Iterable[float], stopped_angle: Iterable[float]) -> None:
    hist_all, _, _ = np.histogram2d(list(all_initial_p), list(all_initial_angle), bins=[KE_BINS_STOP, ANGLE_BINS_STOP])
    hist_stopped, _, _ = np.histogram2d(list(stopped_p), list(stopped_angle), bins=[KE_BINS_STOP, ANGLE_BINS_STOP])
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_stopped = np.nan_to_num(hist_stopped / hist_all)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fraction_stopped.T,
        extent=[KE_BINS_STOP[0], KE_BINS_STOP[-1], ANGLE_BINS_STOP[0], ANGLE_BINS_STOP[-1]],
        aspect='auto', origin='lower', interpolation='nearest'
    )
    plt.colorbar(label='Fraction Stopped')
    plt.xlabel("Initial Kinetic Energy (MeV)")
    plt.ylabel("Initial Angle (Degrees)")
    plt.title("Fraction Stopped vs. KE and Angle")
    plt.tight_layout(); plt.show()


def plot_energy_angle_distributions(initial_p_stopped: Iterable[float], all_initial_p: Iterable[float],
                                    initial_angle_stopped: Iterable[float], all_initial_angle: Iterable[float]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    counts, bins = np.histogram(list(initial_p_stopped), bins=np.linspace(0, 2, 50))
    axs[1].stairs(counts, bins)
    axs[1].set_title("z=0 Energy Distribution (of stopped e$^+$)")
    ax2 = axs[1].twinx()
    counts, bins = np.histogram(list(all_initial_p), bins=np.linspace(0, 2, 50))
    ax2.stairs(counts, bins, color='red')

    counts, bins = np.histogram(list(initial_angle_stopped), bins=np.linspace(0, 90, 90))
    axs[0].stairs(counts, bins)
    axs[0].set_title("z=0 Angular Distribution (of stopped e$^+$)")
    ax2 = axs[0].twinx()
    counts, bins = np.histogram(list(all_initial_angle), bins=np.linspace(0, 90, 90))
    ax2.stairs(counts, bins, color='red')

    axs[1].set_xlabel("Initial Kinetic Energy (MeV)")
    axs[0].set_xlabel("Initial Angle (Deg)")
    axs[0].set_ylabel("Count")
    axs[1].set_ylabel("Count")

    plt.tight_layout(); plt.show()


def plot_reflection_energy(reflected_ke: Iterable[float], all_initial_p: Iterable[float]) -> None:
    plt.figure(figsize=(6, 4))
    c1, b1 = np.histogram(list(reflected_ke), bins=np.linspace(0, 5, 100))
    c2, _ = np.histogram(list(all_initial_p), bins=np.linspace(0, 5, 100))
    ratio = np.divide(c1, (c1 + c2), out=np.zeros_like(c1, dtype=float), where=(c1 + c2) != 0)
    plt.stairs(ratio, b1)
    plt.xlabel("Initial Kinetic Energy (MeV)")
    plt.ylabel("Fraction Reflected")
    plt.title("Energy of Reflected Positrons")
    plt.tight_layout(); plt.show()


def plot_reflection_fraction_2d(reflected_ke: Iterable[float], reflected_angle: Iterable[float],
                                all_initial_p: Iterable[float], all_initial_angle: Iterable[float]) -> None:
    ke_bins = np.linspace(0, 5, 100)
    angle_bins = np.linspace(0, 90, 10)

    hist_all, _, _ = np.histogram2d(list(all_initial_p), list(all_initial_angle), bins=[ke_bins, angle_bins])
    hist_ref, _, _ = np.histogram2d(list(reflected_ke), list(reflected_angle), bins=[ke_bins, angle_bins])

    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_reflected = np.nan_to_num(hist_ref / hist_all, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        fraction_reflected.T,
        extent=[ke_bins[0], ke_bins[-1], angle_bins[0], angle_bins[-1]],
        aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1
    )
    plt.colorbar(label='Fraction Reflected')
    plt.xlabel("Initial Kinetic Energy (MeV)")
    plt.ylabel("Initial Angle (Degrees)")
    plt.title("Fraction Reflected vs. KE and Angle")
    plt.tight_layout(); plt.show()


def plot_energy_fate_breakdown(all_initial_p: Iterable[float],
                               stopped_ke: Iterable[float],
                               reflected_ke: Iterable[float],
                               annihilated_ke: Iterable[float]) -> None:
    energy_bins = np.linspace(0, 5, 100)

    all_ke = np.asarray(list(all_initial_p))
    stopped_ke = np.asarray(list(stopped_ke))
    reflected_ke = np.asarray(list(reflected_ke))
    annihilated_ke = np.asarray(list(annihilated_ke))

    hist_total, _ = np.histogram(all_ke, bins=energy_bins)
    hist_stopped, _ = np.histogram(stopped_ke, bins=energy_bins)
    hist_reflected, _ = np.histogram(reflected_ke, bins=energy_bins)
    hist_annihilated, _ = np.histogram(annihilated_ke, bins=energy_bins)
    hist_transmitted = hist_total - (hist_stopped + hist_reflected + hist_annihilated)

    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_stopped = np.nan_to_num(hist_stopped / hist_total)
        fraction_transmitted = np.nan_to_num(hist_transmitted / hist_total)
        fraction_reflected = np.nan_to_num(hist_reflected / hist_total)
        fraction_annihilated = np.nan_to_num(hist_annihilated / hist_total)

    bottom = np.zeros_like(fraction_stopped)
    width = energy_bins[1] - energy_bins[0]
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    plt.figure(figsize=(8, 6))
    for frac, label, color in zip(
        [fraction_stopped, fraction_transmitted, fraction_reflected, fraction_annihilated],
        ['Stopped', 'Transmitted', 'Reflected', 'Annihilated'],
        ['darkgreen', 'steelblue', 'goldenrod', 'firebrick']
    ):
        plt.bar(bin_centers, frac, bottom=bottom, width=width, label=label, color=color)
        bottom += frac

    

    plt.xlabel("Initial Kinetic Energy (MeV)")
    plt.ylabel("Fraction")
    plt.legend()
    plt.tight_layout(); plt.show()


# ===== Main routine =====


def main(root: str = DATA_DIR, ke_filter_mev: float = KE_FILTER_MEV, E: float | None = None) -> None:
    # Load and tally stopped events
    tally = process_directory(root, ke_filter_mev)

    # Print only: number reflected and total number of events
    total_events = len(tally.all_initial_p)
    print(tally.n_reflected)
    print(total_events)

    # Third line: fraction reflected ± 1σ binomial error
    if total_events > 0:
        frac_reflected = tally.n_reflected / total_events
        err_reflected = math.sqrt(frac_reflected * (1.0 - frac_reflected) / total_events)
    else:
        frac_reflected = 0.0
        err_reflected = 0.0
    print(f"{frac_reflected} ± {err_reflected}")

    # Convert lists to arrays
    end_z = np.asarray(tally.end_z)

    # Plot only the stopping distribution histogram with a ramped Makhovian fit
    lambda_infinite_flag = True #(E is not None and E < 20)
    fit_and_plot_stopping_distribution(end_z, z_center_mm=Z_POSITION, thickness_mm=LENGTH, lambda_infinite=lambda_infinite_flag, E=E)


# ===== Additional collectors (reflections / annihilations) =====

def collect_reflected(root: str) -> Tuple[List[float], List[float]]:
    files, files_r, _ = list_event_files(root)
    reflected_ke: List[float] = []
    reflected_angle: List[float] = []
    for f in files_r:
        df = filter_ke(add_initial_energy_mev(load_parquet_safe(f)), KE_FILTER_MEV)
        if df is None:
            continue
        reflected_ke += df["initialE"].tolist()
        reflected_angle += df["initialAngle"].tolist()
    return reflected_ke, reflected_angle


def collect_annihilated(root: str) -> List[float]:
    _, _, files_a = list_event_files(root)
    ke: List[float] = []
    for f in files_a:
        df = filter_ke(add_initial_energy_mev(load_parquet_safe(f)), KE_FILTER_MEV)
        if df is None:
            continue
        ke += df["initialE"].tolist()
    return ke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze positron transport outputs and fit stopping distribution.")
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Directory containing Out*.dat files")
    parser.add_argument("--ke", type=float, default=KE_FILTER_MEV, help="Upper cut on initial kinetic energy (MeV)")
    parser.add_argument("--E", type=float, default=None, help="Optional beam energy parameter. If E<0.2, lambda is treated as infinite in the fit.")
    args = parser.parse_args()

    main(root=args.root, ke_filter_mev=args.ke, E=args.E)