import numpy as np
import sys

#######################
# Physical Parameters #
#######################
# Reynolds number
Re = 66.0

# Turbulent Reynolds numner
# C_zt = 0.05*Re #Linear d
C_zt = 0.035*Re #Cubic d

# Angle of laminar solution
theta = 24.0*np.pi/180.

# Domain size
Lz = 256.0

# Beta
beta = np.pi/2

# Drag
alpha = 0.01

# Closures
a    = 0.3
c1 = 0.144
eta  = 5.e-3
kappa = 0.045

# Kinetic energy of initial condition:
E0 = 1e-4  # Large scale flow
Eq0 = 0.05 # turbulence
nIC = 8 # Mode number cutoff for noisy IC

# IC file (if None, then blank restart. Otherwise specify a txt file which is an array for q only)
IC_file=None
#IC_file='equil'
#IC_file = '../q0_Lz256.txt'

# Random force
rand_force = False
mu = 0.0
sigma = 1.0

########################
# Numerical Parameters #
########################
# Numerical resolution
Nz = int(Lz*2)
# Viscous time-scale: q_max*C_zt/L_min**2, q_max ~ 0.08. We'll add safety factor of 2.
dt_init= (Lz/Nz)**2/0.08/C_zt/16


# Set how long you want to run for:
##sim_tmax = np.inf  # simulation time units
sim_tmax = np.inf  # simulation time units
##real_tmax = (11+55/60.)*60*60 # 12*60= 12 mins. Real time is in seconds ... 12 hours = 43200
real_tmax = 30*60 # 12*60= 12 mins. Real time is in seconds ... 12 hours = 43200
##real_tmax = (14/60.)*60*60 # 12*60= 12 mins. Real time is in seconds ... 12 hours = 43200
# real_tmax = np.inf
# Simulation will stop when the first of either is reached.

# Real (wall) time interval between snapshots exporting
#tsnap_wall = real_tmax/3.-15*60 # So that we save 3 snapshots 15 minutes before each third
#tsnap_wall = real_tmax/500.
tsnap_wall = np.inf

# Simulation time interval between snapshots exporting
tsnap_sim = 25.

# Time steps between outputting spectra and fluxes:
#tspec = 230
#tspec = 100

# Time steps between outputting time series info
tseries = 25.
#tseries = 10
