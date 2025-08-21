"""
ModeratorProcessG.py

This script analyzes positron transport/parquet outputs to compute stop/reflection
statistics, geometry-based escape probabilities (for a *multi-foil* moderator),
and a set of diagnostic plots.
"""

from __future__ import annotations

# ===== Standard library imports =====
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import warnings

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
DATA_DIR: str = "data-amd"
THERMALIZATION_THRESHOLD_GEV: float = 0.001  # Unused (kept for parity)

#
# Geometry (mm) -- Four thin box foils (25 µm each)
FOIL_THICKNESS_MM: float = 0.025
FOIL_CENTERS_MM: Tuple[float, ...] = (18.0, 28.0, 38.0, 48.0)
# Box cross-section (±25 mm in x and y for 50 mm width/height); not currently used in diffusion calc
MODERATOR_X_HALF_MM: float = 25.0
MODERATOR_Y_HALF_MM: float = 25.0
MAX_STEP: float = 0.001

# Side plates (mm): four 25 µm foils normal to x at x = ±15, ±5; span y ∈ [−25,25], z ∈ [33−25, 33+25]
SIDE_FOIL_THICKNESS_MM: float = 0.025
SIDE_FOIL_X_CENTERS_MM: Tuple[float, ...] = (-15.0, -5.0, 5.0, 15.0)
SIDE_FOIL_Z_CENTER_MM: float = 33.0
SIDE_FOIL_Z_HALF_MM: float = 25.0
SIDE_FOIL_Y_HALF_MM: float = 25.0

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
    stopped = [f for f in all_files if "r" not in f and "_a" not in f]
    reflected = [f for f in all_files if "r" in f]
    annihilated = [f for f in all_files if "_a" in f]
    return stopped, reflected, annihilated


def load_parquet_safe(path: str) -> pd.DataFrame | None:
    """Read a parquet file; on failure, report and return None."""
    try:
        df = pd.read_parquet(path)
        return df
    except Exception:
        print(f"Err {path}")
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

        # Select stopped within any foil (multi-foil selection)
        half = FOIL_THICKNESS_MM / 2.0
        z = df_s["endz"].to_numpy()
        mask = np.zeros_like(z, dtype=bool)
        for c in FOIL_CENTERS_MM:
            mask |= (z >= (c - half)) & (z <= (c + half))
        df_s_sel = df_s[mask]

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
    """Compute distances (mm) from points to the nearest moderator surface.

    Surfaces considered:
    - Four *z-oriented* foils (Moderator1–4): thickness FOIL_THICKNESS_MM at z-centers FOIL_CENTERS_MM; valid for |x|≤25, |y|≤25.
    - Four *x-oriented* foils (Moderator9–12): thickness SIDE_FOIL_THICKNESS_MM at x-centers SIDE_FOIL_X_CENTERS_MM,
      spanning z ∈ [SIDE_FOIL_Z_CENTER_MM ± SIDE_FOIL_Z_HALF_MM] and |y|≤SIDE_FOIL_Y_HALF_MM.

    We treat diffusion as primarily normal to the nearest surface; distances in other
    directions are ignored, so we report zeros for the radial/orthogonal component.

    Returns (dist_overall, dist_to_z_faces, dist_to_x_faces).
    """
    end_x = np.asarray(end_x, dtype=float)
    end_y = np.asarray(end_y, dtype=float)
    end_z = np.asarray(end_z, dtype=float)

    n = end_z.size
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    # --- Distances to z-oriented foil faces (if inside their x–y footprint) ---
    half_z = FOIL_THICKNESS_MM / 2.0
    faces_z = np.array([edge for c in FOIL_CENTERS_MM for edge in (c - half_z, c + half_z)], dtype=float)
    # Only consider if inside the foil's x–y area (|x|,|y| ≤ 25)
    in_xy = (np.abs(end_x) <= MODERATOR_X_HALF_MM) & (np.abs(end_y) <= MODERATOR_Y_HALF_MM)
    # Default to +inf when outside footprint so it won't be chosen as min
    dz_all = np.full(n, np.inf, dtype=float)
    if faces_z.size:
        dz = np.min(np.abs(end_z[:, None] - faces_z[None, :]), axis=1)
        dz_all[in_xy] = dz[in_xy]

    # --- Distances to x-oriented side-foil faces (if inside their y–z span) ---
    half_x = SIDE_FOIL_THICKNESS_MM / 2.0
    faces_x = np.array([edge for c in SIDE_FOIL_X_CENTERS_MM for edge in (c - half_x, c + half_x)], dtype=float)
    in_yz = (np.abs(end_y) <= SIDE_FOIL_Y_HALF_MM) & (
        (end_z >= (SIDE_FOIL_Z_CENTER_MM - SIDE_FOIL_Z_HALF_MM)) & (end_z <= (SIDE_FOIL_Z_CENTER_MM + SIDE_FOIL_Z_HALF_MM))
    )
    dx_all = np.full(n, np.inf, dtype=float)
    if faces_x.size:
        dx = np.min(np.abs(end_x[:, None] - faces_x[None, :]), axis=1)
        dx_all[in_yz] = dx[in_yz]

    # Overall nearest surface distance is minimum over applicable families
    dist_overall = np.minimum(dz_all, dx_all)

    # Replace any remaining +inf (outside all footprints) by distances to z-foil faces (no footprint),
    # to maintain legacy behavior of always producing a finite distance along z.
    inf_mask = ~np.isfinite(dist_overall)
    if np.any(inf_mask):
        # Fallback: nearest z-face regardless of x–y
        dist_overall[inf_mask] = np.min(np.abs(end_z[inf_mask][:, None] - faces_z[None, :]), axis=1)
        dz_all[inf_mask] = dist_overall[inf_mask]

    # For compatibility with callers: return component-wise distances
    dist_to_z = dz_all
    dist_to_x = dx_all
    return dist_overall, dist_to_z, dist_to_x


# ===== Analysis helpers =====

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


def plot_penetration_hist(end_z_mm: np.ndarray, all_initial_angle: Iterable[float], thickness_mm: float = FOIL_THICKNESS_MM) -> None:
    """Histogram of penetration depth within each foil (combined).

    Maps each stop z to its depth within the containing foil (0..thickness), then
    builds a single histogram over that range.
    """
    half = thickness_mm / 2.0
    depths_um: List[float] = []
    for z in np.asarray(end_z_mm, dtype=float):
        for c in FOIL_CENTERS_MM:
            if (z >= c - half) and (z <= c + half):
                depths_um.append((z - (c - half)) * 1000.0)  # 0 at entrance face, µm
                break

    if len(depths_um) == 0:
        # Nothing to plot
        return

    depths_um = np.asarray(depths_um)
    counts, bins = np.histogram(depths_um, np.linspace(0, thickness_mm * 1000.0, 50))
    counts = counts.astype(float)
    counts = counts / max(len(list(all_initial_angle)), 1)
    counts *= 1e4
    counts *= 50 / (thickness_mm * 1000.0)
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

def main(root: str = DATA_DIR, ke_filter_mev: float = KE_FILTER_MEV) -> None:
    # Load and tally
    tally = process_directory(root, ke_filter_mev)

    # Convert lists to arrays
    end_x = np.asarray(tally.end_x)
    end_y = np.asarray(tally.end_y)
    end_z = np.asarray(tally.end_z)

    # Distances & escape probability
    dist_overall, dist_z, dist_x = nearest_surface_distance_mm(end_x, end_y, end_z)
    out_prob = np.exp(-dist_overall / LAMBDA_MM)

    # Quick inspection table (sorted by descending probability of escape)
    df_preview = pd.DataFrame({
        "x": end_x, "y": end_y, "z": end_z,
        "dx": dist_overall, "dz": dist_z, "d": 1000 * dist_overall,
        "p": np.round(100 * out_prob, 2),
    }).sort_values("p")
    print(df_preview)

    # Monte Carlo re-emission sampling
    diff_x, diff_y, diff_z, stats = simulate_reemission(end_x, end_y, end_z, out_prob)

    # Event accounting relative to beam/RunID metadata (kept from original)
    try:
        # original script used Out1.dat for RunID max
        n_runs = int(np.max(pd.read_parquet(f"{root}/Out1.dat")["RunID"]) + 1)
        n_hits_target = n_runs * 100000 * len(list_event_files(root)[0])
    except Exception:
        n_hits_target = 0

    # Derived efficiencies & prints (mirroring original outputs)
    print(stats["sum_p"], stats["n_diff_last"])  # sum of probabilities, last trial count
    print(f"{n_hits_target} hit target")
    print(f"{len(tally.all_initial_angle)} hit moderator, {round(len(tally.all_initial_angle) / max(n_hits_target, 1), 4)}")
    print(f"{len(end_z)} stop in moderator, {round(len(end_z) / max(len(tally.all_initial_angle), 1), 4)}")
    print(f"{stats['n_diff_last']} reemitted, {round(stats['sum_p'] / max(len(end_z), 1), 4)}")
    under_500_all = sum(1 for e in tally.all_initial_p if e < 0.5)
    under_500_stopped = sum(1 for e in tally.initial_p_stopped if e < 0.5)
    print(f"Particles under 500 keV: {under_500_all}, stopped under 500 keV: {under_500_stopped}")

    mod_eff = 1e7 * stats["sum_p"] / max(len(tally.all_initial_angle), 1)
    sys_eff = 1e8 * stats["sum_p"] / max(n_hits_target, 1)
    print(f"Moderator efficiency: {mod_eff}, pm {1e7 * math.sqrt(stats['sum_p']) / max(len(tally.all_initial_angle), 1)}")
    print(f"System efficiency: {sys_eff}, pm {1e8 * math.sqrt(stats['sum_p']) / max(n_hits_target, 1)}")

    std_mean = stats["std_mean"]
    rms_mean = stats["rms_mean"]
    print("Standard deviation", *std_mean)
    print("RMS", *rms_mean)

    # Brightness estimate
    brightness, err = compute_brightness(out_prob, (std_mean[0], std_mean[1], std_mean[2]))
    print("Brightness:", brightness, "pm", err)

    # ---- Plots (kept in roughly original order) ----
    plot_3d_points(diff_x, diff_y, diff_z)
    plot_penetration_hist(end_z, tally.all_initial_angle)
    plot_stop_fraction_2d(tally.all_initial_p, tally.all_initial_angle,
                          tally.initial_p_stopped, tally.initial_angle_stopped)
    plot_energy_angle_distributions(tally.initial_p_stopped, tally.all_initial_p,
                                    tally.initial_angle_stopped, tally.all_initial_angle)

    # Reflections & fate breakdown
    reflected_ke, reflected_angle = collect_reflected(root)
    plot_reflection_energy(reflected_ke, tally.all_initial_p)
    plot_reflection_fraction_2d(reflected_ke, reflected_angle, tally.all_initial_p, tally.all_initial_angle)

    annihilated_ke = collect_annihilated(root)
    plot_energy_fate_breakdown(tally.all_initial_p, tally.initial_p_stopped, reflected_ke, annihilated_ke)


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
    main()