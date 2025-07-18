import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# Load data from files, skipping comment lines starting with '#'
data1 = np.loadtxt("probeOutput1.txt", comments="#")
data2 = np.loadtxt("probeOutput2.txt", comments="#")

# Extract z and Bz from file 1
z1 = data1[:, 2]   # z is the 3rd column (index 2)
Bz = data1[:, 6]   # Bz is the 7th column (index 6)

# Extract z and compute Br from file 2
z2 = data2[:, 2]
Bx = data2[:, 4]
By = data2[:, 5]
Br = np.sqrt(Bx**2 + By**2)

# Plot Bz vs z
plt.figure()
plt.plot(z1, Bz, label="Bz")

# Define constants for the analytical expression
B0 = 5.0  # example value in Tesla
a = 0.0258   # example value in 1/m

# Compute analytical Bz for the same z1 range
Bz_analytic = B0 / (1 + a * z1)

# Plot the analytical Bz on the same figure
plt.plot(z1, Bz_analytic, label="Analytical $B_z = B_0 / (1 + a z)$", linestyle='--')

plt.xlabel("$z$ (m)")
plt.ylabel("$B_z$ (T)")
plt.title("$B_z$ vs $z$ ($r=0$)")
plt.grid(True)
plt.legend()

# Plot Br vs z
plt.figure()
plt.plot(z2, Br, label="Br")
plt.xlabel("$z$ (m)")
plt.ylabel("$B_r$ (T)")
plt.title("$B_r$ vs $z$ ($r=1$)")
plt.grid(True)
plt.legend()

plt.show()