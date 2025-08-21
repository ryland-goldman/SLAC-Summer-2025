import re
import math
import argparse
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# ---- fitting helpers ----
def _weights_from_errors(err_arr):
    if err_arr is None:
        return None
    finite = np.isfinite(err_arr) & (err_arr > 0)
    if not np.any(finite):
        return None
    w = np.zeros_like(err_arr, dtype=float)
    w[finite] = 1.0 / err_arr[finite]
    return w
# ---- nonlinear models for m(E) ----

def model_pow_offset(E, m_inf, A, alpha):
    # m(E) = m_inf + A * E^{-alpha}
    E = np.asarray(E, dtype=float)
    return m_inf + A * np.power(E, -alpha)


def model_log(E, m0, beta):
    # m(E) = m0 - beta * ln(E)
    E = np.asarray(E, dtype=float)
    return m0 - beta * np.log(E)


def model_exp_offset(E, m_inf, A, beta):
    # m(E) = m_inf + A * exp(-beta * E)
    E = np.asarray(E, dtype=float)
    return m_inf + A * np.exp(-beta * E)


def fit_curve(x, y, yerr, func, p0=None, bounds=(-np.inf, np.inf)):
    """Weighted nonlinear least squares using scipy.curve_fit.
    Returns popt, pcov and R^2 in linear y-space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        finite = np.isfinite(yerr) & (yerr > 0)
        if np.any(finite):
            sigma = yerr
    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, sigma=sigma, absolute_sigma=(sigma is not None), maxfev=20000)
    y_pred = func(x, *popt)
    r2 = r2_score(y, y_pred)
    return popt, pcov, r2


def polyfit_with_errors(x, y, yerr, deg):
    """Weighted polynomial fit y(x) = sum_i c[i] * x**(deg-i).
    Returns coeffs (highest power first).
    """
    w = _weights_from_errors(yerr)
    return np.polyfit(x, y, deg=deg, w=w) if w is not None else np.polyfit(x, y, deg=deg)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ---- power-law fitting: y = A * x^n ----
def _safe_log(arr):
    arr = np.asarray(arr, dtype=float)
    return np.log(arr)


def fit_power_law(x, y, yerr=None):
    """Fit y = A * x^n via linear regression on logs.
    Returns (A, n) and R^2 computed in log space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = _safe_log(x)
    ly = _safe_log(y)

    w = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        finite = np.isfinite(yerr) & (yerr > 0) & np.isfinite(y) & (y > 0)
        if np.any(finite):
            # propagate errors: sigma_{log y} ≈ sigma_y / y
            sigma_logy = np.zeros_like(y)
            sigma_logy[finite] = yerr[finite] / y[finite]
            good = finite & (sigma_logy > 0)
            if np.any(good):
                w = 1.0 / sigma_logy[good]
                lx, ly = lx[good], ly[good]
            else:
                w = None
        # else leave w=None

    if w is not None:
        b, a = np.polyfit(lx, ly, deg=1, w=w)
    else:
        b, a = np.polyfit(lx, ly, deg=1)

    # Predict and R^2 in log space
    ly_pred = a + b * lx
    ss_res = np.sum((ly - ly_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    A = np.exp(a)
    n = b
    return A, n, r2

# Regex to parse a line like:
# "m, z0, lambda: 1.96228 ± 0.121144, 0.313595 ± 0.0130693, inf"
TRIPLET_RE = re.compile(
    r"""
    m,\s*z0,\s*lambda:\s*
    (?P<m_val>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)     # m value
    (?:\s*±\s*(?P<m_err>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))? # m err
    \s*,\s*
    (?P<z_val>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)     # z0 value
    (?:\s*±\s*(?P<z_err>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))? # z0 err
    \s*,\s*
    (?P<l_val>(?:inf)|(?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)) # lambda value or 'inf'
    (?:\s*±\s*(?P<l_err>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))? # lambda err (optional)
    """,
    re.VERBOSE,
)

def is_float_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_file(path: Path):
    E, m, m_err, z0, z0_err = [], [], [], [], []
    cur_E = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == r"\n":
                continue
            if line.startswith("E") or line.startswith("m, z0, lambda: m(E)"):
                continue

            if is_float_line(line):
                cur_E = float(line)
                continue

            mtrip = TRIPLET_RE.match(line)
            if mtrip and cur_E is not None:
                E.append(cur_E)

                m_val = float(mtrip.group("m_val"))
                m.append(m_val)
                m_err.append(float(mtrip.group("m_err")) if mtrip.group("m_err") else np.nan)

                z_val = float(mtrip.group("z_val"))
                z0.append(z_val)
                z0_err.append(float(mtrip.group("z_err")) if mtrip.group("z_err") else np.nan)

                # lambda is parsed but not used; kept here if needed later
                # l_val = math.inf if mtrip.group("l_val") == "inf" else float(mtrip.group("l_val"))

                cur_E = None  # reset until next E
            # else: silently ignore unrecognized lines

    # Convert to numpy arrays
    E = np.array(E, dtype=float)
    m = np.array(m, dtype=float)
    m_err = np.array(m_err, dtype=float)
    z0 = np.array(z0, dtype=float)
    z0_err = np.array(z0_err, dtype=float)
    return E, m, m_err, z0, z0_err

def main():
    ap = argparse.ArgumentParser(description="Parse out.txt and plot z0(E) and m(E).")
    ap.add_argument("file", nargs="?", default="out.txt", help="Input file (default: out.txt)")
    ap.add_argument("--show", action="store_true", help="Display plots interactively")
    ap.add_argument("--prefix", default="plots", help="Output filename prefix (default: plots)")
    ap.add_argument("--deg-z", type=int, default=2, help="Polynomial degree for z0(E) fit (default: 2)")
    ap.add_argument("--deg-m", type=int, default=2, help="Polynomial degree for m(E) fit (default: 2)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    E, m, m_err, z0, z0_err = parse_file(path)
    if E.size == 0:
        raise SystemExit("No data parsed. Check input format.")

    # z0(E): log-log
    plt.figure()
    # Use errorbars where finite
    z_err = np.where(np.isfinite(z0_err), z0_err, 0.0)
    # Matplotlib does not allow zero error for log scale if value==0; filter zeros if any
    mask = (E > 0) & (z0 > 0)
    plt.errorbar(E[mask], z0[mask], yerr=z_err[mask] if np.any(z_err[mask] > 0) else None,
                 fmt="o", capsize=3)
    # power-law fit for z0(E): z0 = A * E^n
    if np.count_nonzero(mask) >= 2:
        try:
            A_z, n_z, r2_z = fit_power_law(E[mask], z0[mask], z_err[mask])
            lx = np.log(E[mask])
            ly = np.log(z0[mask])
            if np.any(z_err[mask] > 0):
                w = 1 / z_err[mask][z_err[mask] > 0]
            else:
                w = None
            coeffs, cov = np.polyfit(lx, ly, deg=1, cov=True, w=w)
            b, a = coeffs
            A_err = np.exp(a) * np.sqrt(cov[1,1])
            n_err = np.sqrt(cov[0,0])
            xfit_z = np.linspace(np.min(E[mask]), np.max(E[mask]), 300)
            yfit_z = A_z * (xfit_z ** n_z)
            plt.plot(xfit_z, yfit_z, label=f"Curve fit")
            print(f"z0(E) power-law fit: z0 = A * E^n with A={A_z:.8g} ± {A_err:.2g}, n={n_z:.8g} ± {n_err:.2g}, R^2_log={r2_z:.6f}")
            plt.legend()
        except Exception as e:
            print("z0(E) power-law fit failed:", e)
    plt.xscale("log")
    from matplotlib.ticker import LogLocator
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    plt.yscale("log")
    plt.xlabel("Positron Energy (MeV)")
    plt.ylabel("z0")
    plt.title("z0 vs Positron Energy")
    plt.tight_layout()
    out1 = f"{args.prefix}_z0_loglog.png"
    plt.savefig(out1, dpi=200)

    # m(E): log-lin -> log x, linear y
    plt.figure()
    m_err_use = np.where(np.isfinite(m_err), m_err, 0.0)
    mask_m = E > 0
    plt.errorbar(E[mask_m], m[mask_m], yerr=m_err_use[mask_m] if np.any(m_err_use[mask_m] > 0) else None,
                 fmt="o", capsize=3)
    # overlay three candidate fits for m(E)
    xfit_m = np.linspace(np.min(E[mask_m]), np.max(E[mask_m]), 400)

    #plt.legend()
    plt.xscale("log")
    from matplotlib.ticker import LogLocator
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    plt.xlabel("Positron Energy (MeV)")
    plt.ylabel("m")
    plt.title("m vs Positron Energy")
    plt.tight_layout()
    out2 = f"{args.prefix}_m_logx.png"
    plt.savefig(out2, dpi=200)

    if args.show:
        plt.show()

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")

if __name__ == "__main__":
    main()