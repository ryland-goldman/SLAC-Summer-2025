import scipy
import numpy as np
import pickle as pkl

# find the positron energies from the interpolation script
with open("../Tungsten-Target-InterpolatedEnergy/energy_interpolation.pkl","rb") as f: c,x,axis,extrap = pkl.load(f)
dist_at_E = scipy.interpolate.PPoly(c,x,axis=axis,extrapolate=extrap)

num_events = int(1e4)

input_energy = np.ones(num_events) * 70
positron_inital_energy = dist_at_E(input_energy).sum(axis=0) / num_events
energy_bins = np.linspace(0.5,100,200)


# interpolate the z0 values from https://doi.org/10.1063/1.5097607
ke = np.array([10,100,300,500,800,1000,5000,10000],dtype=np.float32) / 1000.0
z0 = np.array([0.083,3.22,16.5,35.24,66.27,88.8,619.2,1222.1])

z0_of_e = scipy.interpolate.CubicSpline(ke,z0,bc_type='natural')

m = 1.828


# calculate the number of positrons that stop somewhere, based on their energies
z = np.linspace(0,200,1000)
stopped = np.zeros(len(z))
for i in range(len(energy_bins)):
    num_pos = positron_inital_energy[i]
    energy = energy_bins[i]
    
    z0 = z0_of_e(energy)

    p = (m * np.pow(z, m-1) / np.pow(z0,m)) * np.exp( -np.pow(z/z0,m) ) * num_pos

    stopped += p


# calculate the number of positrons out of the moderator, based on the stopping position
y0 = 0.27
Lplus = 0.055

num_out = []
for i in range(len(z)):
    curr_z = z[i]
    reemmited = stopped * np.maximum(np.exp(-z/Lplus), np.minimum(np.exp(-(curr_z-z)/Lplus),1))
    n = y0 * scipy.integrate.cumulative_trapezoid(reemmited, z, initial=0)[i]
    num_out.append(n)
num_out = np.array(num_out)


# plot it
import matplotlib.pyplot as plt
plt.plot(z,num_out)
plt.xlabel("Moderator Thickness (µm)")
plt.ylabel("Positron Yield (/$10^4\\text{e}^+$)")
plt.show()