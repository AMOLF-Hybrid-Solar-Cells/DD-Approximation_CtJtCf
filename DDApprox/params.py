from scipy.constants import k, e, epsilon_0
import numpy as np
import matplotlib.pyplot as plt
import sys

# Device settings

d_pero = 550.0e-9  # [m] Perovskite thickness
d_htl = 50.0e-9  # [m] HTL thickness
d_etl = 50.0e-9  # [m] ETL thickness

wf_anode = 5.3   # [eV] Work function of anode
wf_cathode = 4.1  # [eV] Work function of cathode

n0_ion = 1e17 * 100**3  # [1/m^3] Mobile ion density
ea_dn_ion = 5  # [eV] Activation energy of temperature activated ion density
dn_ion = 0 * 100**3  # [1/m^3] Temperature activated mobile ion density
# The ion density at temperature T is then n0_ion + dn_ion*exp(-Ea*e/(kT))

# Activate or deactivate doping in CTLs
htl_doping = True
etl_doping = True

n_htl = 5e17 * 100**3  # [1/m^3] Doping density HTL
n_etl = 5e17 * 100**3  # [1/m^3] Doping density ETL

mu_p_htl = 1e-8  # [m^2/Vs] Hole mobility in HTL
mu_p_pero = 1e-4  # [m^2/Vs] Hole mobility in Perovskite
mu_p_etl = 1e-8  # [m^2/Vs] Hole mobility in ETL

mu_n_htl = 1e-8  # [m^2/Vs] Electron mobility in HTL 
mu_n_pero = 1e-4  # [m^2/Vs] Electron mobility in Perovskite
mu_n_etl = 1e-8  # [m^2/Vs] Electron mobility in ETL

D0_ion = 1e-5 * 100**-2  # [m^2/s] Diffusion coefficient of mobile ions
ea_ion_mob = 0.3  # [eV] Activation energy of diffusion coefficient of ions
# The diffusion coefficient at temperature T is then D0_ion * exp(-Ea*e/(k*T))

n0_cb_etl = 2.1e18 * 100**3  # [1/m^3] Effective DOS of CB in ETL
n0_vb_etl = 2.1e18 * 100**3  # [1/m^3] Effective DOS of VB in ETL
n0_cb_htl = 2.1e18 * 100**3  # [1/m^3] Effective DOS of CB in HTL
n0_vb_htl = 2.1e18 * 100**3  # [1/m^3] Effective DOS of VB in HTL
n0_cb_pero = 2.1e18 * 100**3  # [1/m^3] Effective DOS of CB in Pero
n0_vb_pero = 2.1e18 * 100**3  # [1/m^3] Effective DOS of VB in Pero

cb_etl = 4.0  # [eV] Energy level of CB in ETL
vb_etl = 6.0  # [eV] Energy level of VB in ETL
cb_htl = 3.4  # [eV] Energy level of CB in HTL
vb_htl = 5.4  # [eV] Energy level of VB in HTL
cb_pero = 3.9  # [eV] Energy level of CB in Pero
vb_pero = 5.5  # [eV] Energy level of CB in Pero

epsr_htl = 4.0  # Permittivity HTL
epsr_etl = 8.0  # Permittivity ETL
epsr_pero = 62.0  # Permittivity perovskite

r_shunt = 1e6  # [Ohm m^2] Shunt resistance

# Set this to True to consider cations in capcaitance frequency simulations
consider_cations = False

# Settings for Capacitance transient simulations
freq_ct = 10e3 # [Hz] Frequency of the AC perturbation during capacitance transient simulations
t_high = 60.0  # [s] Time the high voltage pulse is applied
vac = 20e-3  # AC perturbation voltage
vapp = 1.2  # Voltage during pulse
vlow = 0.0  # Voltage after pulse
points_early_time = 5 # Number of points added before the first time step (to ensure correct initial conditions in the case of fast ions)
start_sim = -8  # Order of magnitude where simulation starts

# S3A for sinusoidal steady state analysis or PP for parallel plate approximation. PP is faster but does not consider reverse and intermediate accumulation (when ions are accumulated at the perovskite/ETL interface). PP also doesn't allow for the calculation of the frequency dependent capacitance. PP can be used when only computing current transients (then the approximation of the capacitance does not matter)
capacitance_approximation = 'S3A'

################################################################
# Grid settings
################################################################

# Grid points in perovskite, htl, and etl
nx_pero = 80
nx_htl = 40
nx_etl = 40

# Parameter that impacts the resolution at interfaces (higher is finer resolution at interfaces)
sigma_pero = 3.0

################################################################
# Other
################################################################

# Constants for calculation of Bernoulli function
X1 = -29.98554969583478
X2 = -6.623333247568475e-05
X3 = 9.273929305608775e-06
X4 = 28.937746930131606
X5 = 740.5388178707861
