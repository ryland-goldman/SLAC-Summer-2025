# parse_out_and_integrate.py
import re
import mpmath as mp
from pathlib import Path
import csv
import matplotlib.pyplot as plt

INPUT = Path("out.txt")
OUTPUT = Path("integrals.csv")
L = 50.0
L_PLUS = 0.055

# Regex to capture E, m, z0 (ignore uncertainties and lambda)
re_E = re.compile(r'^\s*([0-9.]+)\s*$')
re_params = re.compile(
    r'm,\s*z0,\s*lambda:\s*'          # header
    r'([0-9.+-Ee]+)\s*±\s*[0-9.+-Ee]+,\s*'  # m and its ±
    r'([0-9.+-Ee]+)\s*±\s*[0-9.+-Ee]+,'     # z0 and its ±
    r'\s*([^\s,]+)'                           # lambda (may be 'inf')
)

pairs = []  # list of (E, m, z0)
lines = INPUT.read_text().splitlines()

i = 0
while i < len(lines):
    # Skip literal "\n" lines or empties
    if lines[i].strip() in {"", r"\n"}:
        i += 1
        continue

    mE = re_E.match(lines[i])
    if mE:
        E = float(mE.group(1))
        # expect params on next non-empty line
        j = i + 1
        while j < len(lines) and lines[j].strip() in {"", r"\n"}:
            j += 1
        if j < len(lines):
            mp_ = re_params.search(lines[j])
            if not mp_:
                raise ValueError(f"Could not parse parameter line near: {lines[j]}")
            m_val = float(mp_.group(1))
            z0_val = float(mp_.group(2))
            pairs.append((E, m_val, z0_val))
        i = j + 1
    else:
        i += 1

# Define integrand components
def p_of_z(z, m, z0):
    # p(z) = m z^{m-1} / z0^m * exp(-(z/z0)^m)
    return m * (z ** (m - 1)) / (z0 ** m) * mp.e ** (- (z / z0) ** m)

def integrand(z, m, z0):
    return p_of_z(z, m, z0) * (mp.e ** (- z / L_PLUS) + 0*mp.e ** (-(L - z) / L_PLUS))

rows = [("E", "m", "z0", "integral_0_to_50")]
for E, mval, z0val in pairs:
    if E <= 0.3:  # only keep energies up to 0.3 MeV
        val = mp.quad(lambda zz: integrand(zz, mval, z0val), [0, L])
        rows.append((E, mval, z0val, float(val)))

energies = [1000.0 * r[0] for r in rows[1:]]  # convert MeV to keV
Pvals = [r[3] for r in rows[1:]]

plt.figure()
plt.loglog(energies, Pvals, marker="o")
import matplotlib.ticker as mticker
ax = plt.gca()
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style="plain", axis="x")
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
ax.xaxis.set_minor_locator(mticker.NullLocator())
plt.xlabel("Energy [keV]")
plt.ylabel("Fraction Thermalized (/$e^+$)") 
plt.title("Thermalization Fraction vs Energy")
plt.savefig("P_vs_E.png", dpi=300)
plt.close()

# Write CSV and also print
with OUTPUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

for r in rows:
    print(r)

print(f"\nWrote {OUTPUT.resolve()}")