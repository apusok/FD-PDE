# Script to produce figures in Appendix E using data produced with pMELTS Jupyter notebooks
# Steps:
# 1. Run MOR_beta_revised.ipynb in ENKI server/JupyterLab (requires ThermoEngine)
# 2. Run MOR_beta_revised_min.ipynb in ENKI server/JupyterLab (requires ThermoEngine)
# 3. Run python plot_beta_pmelts.py

# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
from scipy.optimize import curve_fit

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class EmptyStruct:
  pass

# -------- LOAD DATA -------- #
fname_pickle = 'sim_1_isobaric.pickle'
pickle_off = open (fname_pickle, 'rb')

sim0 = pickle.load(pickle_off)
sim1 = pickle.load(pickle_off)
sim2 = pickle.load(pickle_off)
sim3 = pickle.load(pickle_off)

P1sim0 = sim0
P1sim1 = sim1
P1sim2 = sim2
P1sim3 = sim3
del sim0, sim1, sim2, sim3

fname_pickle = 'sim_1_isobaric_v2.pickle'
pickle_off = open (fname_pickle, 'rb')

sim1 = pickle.load(pickle_off)
sim2 = pickle.load(pickle_off)
sim3 = pickle.load(pickle_off)

P2sim1 = sim1
P2sim2 = sim2
P2sim3 = sim3
del sim1, sim2, sim3

fname_pickle = 'sim_2_isothermal.pickle'
pickle_off = open (fname_pickle, 'rb')

sim0 = pickle.load(pickle_off)
sim1 = pickle.load(pickle_off)
sim2 = pickle.load(pickle_off)
sim3 = pickle.load(pickle_off)

T1sim0 = sim0
T1sim1 = sim1
T1sim2 = sim2
T1sim3 = sim3
del sim0, sim1, sim2, sim3

fname_pickle = 'sim_2_isothermal_v2.pickle'
pickle_off = open (fname_pickle, 'rb')

sim0 = pickle.load(pickle_off)
sim1 = pickle.load(pickle_off)
sim2 = pickle.load(pickle_off)
sim3 = pickle.load(pickle_off)

T2sim0 = sim0
T2sim1 = sim1
T2sim2 = sim2
T2sim3 = sim3
del sim0, sim1, sim2, sim3

fname_pickle = 'sim_3_isentropic.pickle'
pickle_off = open (fname_pickle, 'rb')

sim0 = pickle.load(pickle_off)
sim1 = pickle.load(pickle_off)

S1sim0 = sim0
S1sim1 = sim1
del sim0, sim1

fname_pickle = 'sim_3_isentropic_v2.pickle'
pickle_off = open (fname_pickle, 'rb')

sim0 = pickle.load(pickle_off)
sim1 = pickle.load(pickle_off)

S2sim0 = sim0
S2sim1 = sim1
del sim0, sim1

# -------- PLOTTING -------- #
# P,T,S conditions
fontsize = 14
linewidth = 2
markersize = 2

colors = plt.cm.viridis(np.linspace(0,1,5))

# PT plot
fig = plt.figure(figsize=(14,4))
ax = plt.subplot(1,3,1)
plt.grid(linestyle=':', linewidth=0.5)
#plt.plot(P1sim0.t, P1sim0.p, 'o', markersize=markersize, color=colors[0], label='0.5GPa')
plt.plot(P1sim1.t, P1sim1.p, 'o', markersize=markersize, color=colors[1], label='1.0GPa')
plt.plot(P1sim2.t, P1sim2.p, 'o', markersize=markersize, color=colors[2], label='1.5GPa')
plt.plot(P1sim3.t, P1sim3.p, 'o', markersize=markersize, color=colors[3], label='2.0GPa')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylabel(r'$P$ (MPa)', fontsize=fontsize)
plt.title(r'a) Isobaric - MM3', fontsize=fontsize)
plt.gca().invert_yaxis()
# plt.legend(fontsize='small')

ax = plt.subplot(1,3,2)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.t, T1sim0.p, 'o', markersize=markersize, color=colors[0], label=r'1300$^o$C')
plt.plot(T1sim1.t, T1sim1.p, 'o', markersize=markersize, color=colors[1], label=r'1350$^o$C')
plt.plot(T1sim2.t, T1sim2.p, 'o', markersize=markersize, color=colors[2], label=r'1400$^o$C')
plt.plot(T1sim3.t, T1sim3.p, 'o', markersize=markersize, color=colors[3], label=r'1450$^o$C')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
#plt.ylabel(r'P (MPa)', fontsize=fontsize)
plt.title(r'b) Isothermal - MM3', fontsize=fontsize)
plt.gca().invert_yaxis()

ax = plt.subplot(1,3,3)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.t, S1sim0.p, 'o', markersize=markersize, color=colors[1], label=r'MM3')
plt.plot(S1sim1.t, S1sim1.p, 'o', markersize=markersize, color=colors[2], label=r'mBas')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
#plt.ylabel(r'P (MPa)', fontsize=fontsize)
plt.gca().invert_yaxis()
plt.title(r'c) Isentropic', fontsize=fontsize)
plt.legend()

plt.savefig('P-T-conditions.pdf', bbox_inches = 'tight')

# -------- PLOTTING -------- #
# ISOBARIC - density vs T, Mg#, phi
ind_P1sim1  = np.where(P1sim1.phi>0.0)
ind_P1sim2  = np.where(P1sim2.phi>0.0)
ind_P1sim3  = np.where(P1sim3.phi>0.0)

ind_P2sim1  = np.where(P2sim1.phi>0.0)
ind_P2sim2  = np.where(P2sim2.phi>0.0)
ind_P2sim3  = np.where(P2sim3.phi>0.0)

# Figure 1 - P=2000 MPa
fig = plt.figure(figsize=(12,16))
ax = plt.subplot(4,3,1)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.t, P1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.t, P1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.t, P1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.t, P2sim1.rhos, linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.t, P2sim2.rhos, linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.t, P2sim3.rhos, linestyle='--', linewidth=linewidth, color=colors[3])

plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylabel(r'Solid $\rho$ (g/cm$^3$)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,2)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.Mgs, P1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.Mgs, P1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.Mgs, P1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.Mgs, P2sim1.rhos, linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.Mgs, P2sim2.rhos, linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.Mgs, P2sim3.rhos, linestyle='--', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Solid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,3,3)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.phi, P1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1], label='1.0GPa')
plt.plot(P1sim2.phi, P1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2], label='1.5GPa')
plt.plot(P1sim3.phi, P1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3], label='2.0GPa')

#plt.plot(P2sim1.phi, P2sim1.rhos, linestyle='--', linewidth=linewidth, color=colors[1], label='1.0GPa')
#plt.plot(P2sim2.phi, P2sim2.rhos, linestyle='--', linewidth=linewidth, color=colors[2], label='1.5GPa')
#plt.plot(P2sim3.phi, P2sim3.rhos, linestyle='--', linewidth=linewidth, color=colors[3], label='2.0GPa')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,3,4)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.t[ind_P1sim1[0]], P1sim1.rhof[ind_P1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.t[ind_P1sim2[0]], P1sim2.rhof[ind_P1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.t[ind_P1sim3[0]], P1sim3.rhof[ind_P1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.t[ind_P2sim1[0]], P2sim1.rhof[ind_P2sim1[0]], linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.t[ind_P2sim2[0]], P2sim2.rhof[ind_P2sim2[0]], linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.t[ind_P2sim3[0]], P2sim3.rhof[ind_P2sim3[0]], linestyle='--', linewidth=linewidth, color=colors[3])

plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylabel(r'Liquid $\rho$ (g/cm$^3$)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,5)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.Mgf[ind_P1sim1[0]], P1sim1.rhof[ind_P1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.Mgf[ind_P1sim2[0]], P1sim2.rhof[ind_P1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.Mgf[ind_P1sim3[0]], P1sim3.rhof[ind_P1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.Mgf[ind_P2sim1[0]], P2sim1.rhof[ind_P2sim1[0]], linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.Mgf[ind_P2sim2[0]], P2sim2.rhof[ind_P2sim2[0]], linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.Mgf[ind_P2sim3[0]], P2sim3.rhof[ind_P2sim3[0]], linestyle='--', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,3,6)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.phi[ind_P1sim1[0]], P1sim1.rhof[ind_P1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1], label='1.0GPa')
plt.plot(P1sim2.phi[ind_P1sim2[0]], P1sim2.rhof[ind_P1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2], label='1.5GPa')
plt.plot(P1sim3.phi[ind_P1sim3[0]], P1sim3.rhof[ind_P1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3], label='2.0GPa')

#plt.plot(P2sim1.phi[ind_P2sim1[0]], P2sim1.rhof[ind_P2sim1[0]], linestyle='--', linewidth=linewidth, color=colors[1], label='1.0GPa')
#plt.plot(P2sim2.phi[ind_P2sim2[0]], P2sim2.rhof[ind_P2sim2[0]], linestyle='--', linewidth=linewidth, color=colors[2], label='1.5GPa')
#plt.plot(P2sim3.phi[ind_P2sim3[0]], P2sim3.rhof[ind_P2sim3[0]], linestyle='--', linewidth=linewidth, color=colors[3], label='2.0GPa')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,3,7)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P2sim1.t[:-1], P2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(P2sim2.t[:-1], P2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(P2sim3.t[:-1], P2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(P1sim1.t[:-1], P1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.t[:-1], P1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.t[:-1], P1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3])

plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylabel(r'Solid $\beta$', fontsize=fontsize)
plt.ylim([-2.5,2.5])
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,8)
plt.grid(linestyle=':', linewidth=0.5)

plt.plot(P2sim1.Mgs[:-1], P2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(P2sim2.Mgs[:-1], P2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(P2sim3.Mgs[:-1], P2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(P1sim1.Mgs[:-1], P1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.Mgs[:-1], P1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.Mgs[:-1], P1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3])

plt.xlabel(r'Solid Mg\#', fontsize=fontsize)
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,3,9)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.phi[:-1], P1sim1.betas, linestyle='-', linewidth=linewidth, color='k', label='bulk')
plt.plot(P2sim1.phi[:-1], P2sim1.betas, linestyle=':', linewidth=linewidth, color='k', label='min')

plt.plot(P2sim1.phi[:-1], P2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(P2sim2.phi[:-1], P2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(P2sim3.phi[:-1], P2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(P1sim1.phi[:-1], P1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1], label='1.0GPa')
plt.plot(P1sim2.phi[:-1], P1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2], label='1.5GPa')
plt.plot(P1sim3.phi[:-1], P1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3], label='2.0GPa')

plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,3,10)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.t[ind_P1sim1[0][:-1]], P1sim1.betaf[ind_P1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.t[ind_P1sim2[0][:-1]], P1sim2.betaf[ind_P1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.t[ind_P1sim3[0][:-1]], P1sim3.betaf[ind_P1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.t[ind_P2sim1[0][:-1]], P2sim1.betaf[ind_P2sim1[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.t[ind_P2sim2[0][:-1]], P2sim2.betaf[ind_P2sim2[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.t[ind_P2sim3[0][:-1]], P2sim3.betaf[ind_P2sim3[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[3])
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylabel(r'Liquid $\beta$', fontsize=fontsize)
plt.ylim([-10,10])
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,11)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.Mgf[ind_P1sim1[0][:-1]], P1sim1.betaf[ind_P1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(P1sim2.Mgf[ind_P1sim2[0][:-1]], P1sim2.betaf[ind_P1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(P1sim3.Mgf[ind_P1sim3[0][:-1]], P1sim3.betaf[ind_P1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3])

#plt.plot(P2sim1.Mgf[ind_P2sim1[0][:-1]], P2sim1.betaf[ind_P2sim1[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[1])
#plt.plot(P2sim2.Mgf[ind_P2sim2[0][:-1]], P2sim2.betaf[ind_P2sim2[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[2])
#plt.plot(P2sim3.Mgf[ind_P2sim3[0][:-1]], P2sim3.betaf[ind_P2sim3[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)
plt.ylim([-10,10])

ax = plt.subplot(4,3,12)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(P1sim1.phi[ind_P1sim1[0][:-1]], P1sim1.betaf[ind_P1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1], label='1.0GPa')
plt.plot(P1sim2.phi[ind_P1sim2[0][:-1]], P1sim2.betaf[ind_P1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2], label='1.5GPa')
plt.plot(P1sim3.phi[ind_P1sim3[0][:-1]], P1sim3.betaf[ind_P1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3], label='2.0GPa')

#plt.plot(P2sim1.phi[ind_P2sim1[0][:-1]], P2sim1.betaf[ind_P2sim1[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[1], label='1.0GPa')
#plt.plot(P2sim2.phi[ind_P2sim2[0][:-1]], P2sim2.betaf[ind_P2sim2[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[2], label='1.5GPa')
#plt.plot(P2sim3.phi[ind_P2sim3[0][:-1]], P2sim3.betaf[ind_P2sim3[0][:-1]], linestyle='--', linewidth=linewidth, color=colors[3], label='2.0GPa')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-10,10])

plt.savefig('density_change_1_isobaric.pdf', bbox_inches = 'tight')

# -------- PLOTTING -------- #
# ISOTHRMAL
ind_T1sim0  = np.where(T1sim0.phi>0.0)
ind_T1sim1  = np.where(T1sim1.phi>0.0)
ind_T1sim2  = np.where(T1sim2.phi>0.0)
ind_T1sim3  = np.where(T1sim3.phi>0.0)

# Figure 2
fig = plt.figure(figsize=(12,16))
ax = plt.subplot(4,3,1)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.p, T1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.p, T1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.p, T1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.p, T1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Solid $\rho$ (g/cm$^3$)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,2)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.Mgs, T1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.Mgs, T1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.Mgs, T1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.Mgs, T1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Solid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,3,3)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.phi, T1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[0], label=r'1300$^o$C')
plt.plot(T1sim1.phi, T1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[1], label=r'1350$^o$C')
plt.plot(T1sim2.phi, T1sim2.rhos, linestyle='-', linewidth=linewidth, color=colors[2], label=r'1400$^o$C')
plt.plot(T1sim3.phi, T1sim3.rhos, linestyle='-', linewidth=linewidth, color=colors[3], label=r'1450$^o$C')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,3,4)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.p[ind_T1sim0[0]], T1sim0.rhof[ind_T1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.p[ind_T1sim1[0]], T1sim1.rhof[ind_T1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.p[ind_T1sim2[0]], T1sim2.rhof[ind_T1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.p[ind_T1sim3[0]], T1sim3.rhof[ind_T1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Liquid $\rho$ (g/cm$^3$)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,5)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.Mgf[ind_T1sim0[0]], T1sim0.rhof[ind_T1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.Mgf[ind_T1sim1[0]], T1sim1.rhof[ind_T1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.Mgf[ind_T1sim2[0]], T1sim2.rhof[ind_T1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.Mgf[ind_T1sim3[0]], T1sim3.rhof[ind_T1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,3,6)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.phi[ind_T1sim0[0]], T1sim0.rhof[ind_T1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[0], label=r'1300$^o$C')
plt.plot(T1sim1.phi[ind_T1sim1[0]], T1sim1.rhof[ind_T1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[1], label=r'1350$^o$C')
plt.plot(T1sim2.phi[ind_T1sim2[0]], T1sim2.rhof[ind_T1sim2[0]], linestyle='-', linewidth=linewidth, color=colors[2], label=r'1400$^o$C')
plt.plot(T1sim3.phi[ind_T1sim3[0]], T1sim3.rhof[ind_T1sim3[0]], linestyle='-', linewidth=linewidth, color=colors[3], label=r'1450$^o$C')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,3,7)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T2sim0.p[:-1], T2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[0])
plt.plot(T2sim1.p[:-1], T2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(T2sim2.p[:-1], T2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(T2sim3.p[:-1], T2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(T1sim0.p[:-1], T1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.p[:-1], T1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.p[:-1], T1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.p[:-1], T1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Solid $\beta$', fontsize=fontsize)
plt.ylim([-2.5,2.5])
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,8)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T2sim0.Mgs[:-1], T2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[0])
plt.plot(T2sim1.Mgs[:-1], T2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(T2sim2.Mgs[:-1], T2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(T2sim3.Mgs[:-1], T2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(T1sim0.Mgs[:-1], T1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.Mgs[:-1], T1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.Mgs[:-1], T1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.Mgs[:-1], T1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Solid Mg\#', fontsize=fontsize)
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,3,9)
plt.grid(linestyle=':', linewidth=0.5)

plt.plot(T1sim0.phi[:-1], T1sim0.betas, linestyle='-', linewidth=linewidth, color='k', label=r'bulk')
plt.plot(T2sim0.phi[:-1], T2sim0.betas, linestyle=':', linewidth=linewidth, color='k', label=r'min')

plt.plot(T2sim0.phi[:-1], T2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[0])
plt.plot(T2sim1.phi[:-1], T2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(T2sim2.phi[:-1], T2sim2.betas, linestyle=':', linewidth=linewidth, color=colors[2])
plt.plot(T2sim3.phi[:-1], T2sim3.betas, linestyle=':', linewidth=linewidth, color=colors[3])

plt.plot(T1sim0.phi[:-1], T1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[0], label=r'1300$^o$C')
plt.plot(T1sim1.phi[:-1], T1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[1], label=r'1350$^o$C')
plt.plot(T1sim2.phi[:-1], T1sim2.betas, linestyle='-', linewidth=linewidth, color=colors[2], label=r'1400$^o$C')
plt.plot(T1sim3.phi[:-1], T1sim3.betas, linestyle='-', linewidth=linewidth, color=colors[3], label=r'1450$^o$C')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-2.5,2.5])

indi=35
ax = plt.subplot(4,3,10)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.p[ind_T1sim0[0][indi:-1]], T1sim0.betaf[ind_T1sim0[0][indi:-1]], linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.p[ind_T1sim1[0][:-1]], T1sim1.betaf[ind_T1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.p[ind_T1sim2[0][:-1]], T1sim2.betaf[ind_T1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.p[ind_T1sim3[0][:-1]], T1sim3.betaf[ind_T1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Liquid $\beta$', fontsize=fontsize)
plt.ylim([-30,2.5])
#plt.legend(fontsize='small')

ax = plt.subplot(4,3,11)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.Mgf[ind_T1sim0[0][indi:-1]], T1sim0.betaf[ind_T1sim0[0][indi:-1]], linestyle='-', linewidth=linewidth, color=colors[0])
plt.plot(T1sim1.Mgf[ind_T1sim1[0][:-1]], T1sim1.betaf[ind_T1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(T1sim2.Mgf[ind_T1sim2[0][:-1]], T1sim2.betaf[ind_T1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.plot(T1sim3.Mgf[ind_T1sim3[0][:-1]], T1sim3.betaf[ind_T1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)
plt.ylim([-30,2.5])

ax = plt.subplot(4,3,12)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(T1sim0.phi[ind_T1sim0[0][indi:-1]], T1sim0.betaf[ind_T1sim0[0][indi:-1]], linestyle='-', linewidth=linewidth, color=colors[0], label=r'1300$^o$C')
plt.plot(T1sim1.phi[ind_T1sim1[0][:-1]], T1sim1.betaf[ind_T1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1], label=r'1350$^o$C')
plt.plot(T1sim2.phi[ind_T1sim2[0][:-1]], T1sim2.betaf[ind_T1sim2[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2], label=r'1400$^o$C')
plt.plot(T1sim3.phi[ind_T1sim3[0][:-1]], T1sim3.betaf[ind_T1sim3[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[3], label=r'1450$^o$C')
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-30,2.5])

plt.savefig('density_change_2_isothermal.pdf', bbox_inches = 'tight')

# -------- PLOTTING -------- #
# ISENTROPIC
ind_S1sim0  = np.where(S1sim0.phi>0.0)
ind_S1sim1  = np.where(S1sim1.phi>0.0)

# Figure 3
fig = plt.figure(figsize=(16,16))
ax = plt.subplot(4,4,1)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.p, S1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[1], label=r'MM3')
plt.plot(S1sim1.p, S1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[2], label=r'mBas')
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Solid $\rho$ (g/cm$^3$)', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,4,2)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.t, S1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[1], label='MM3')
plt.plot(S1sim1.t, S1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[2], label='mBas')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,4,3)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.Mgs, S1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.Mgs, S1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Solid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,4,4)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.phi, S1sim0.rhos, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.phi, S1sim1.rhos, linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)

ax = plt.subplot(4,4,5)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.p[ind_S1sim0[0]], S1sim0.rhof[ind_S1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[1], label=r'MM3')
plt.plot(S1sim1.p[ind_S1sim1[0]], S1sim1.rhof[ind_S1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[2], label=r'mBas')
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Liquid $\rho$ (g/cm$^3$)', fontsize=fontsize)
plt.legend(fontsize='small')

ax = plt.subplot(4,4,6)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.t, S1sim0.rhof, linestyle='-', linewidth=linewidth, color=colors[1], label='MM3')
plt.plot(S1sim1.t, S1sim1.rhof, linestyle='-', linewidth=linewidth, color=colors[2], label='mBas')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
#plt.legend(fontsize='small')

ax = plt.subplot(4,4,7)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.Mgf[ind_S1sim0[0]], S1sim0.rhof[ind_S1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.Mgf[ind_S1sim1[0]], S1sim1.rhof[ind_S1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)

ax = plt.subplot(4,4,8)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.phi[ind_S1sim0[0]], S1sim0.rhof[ind_S1sim0[0]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.phi[ind_S1sim1[0]], S1sim1.rhof[ind_S1sim1[0]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)

ax = plt.subplot(4,4,9)
plt.grid(linestyle=':', linewidth=0.5)

plt.plot(S1sim0.p[:-1], S1sim0.betas, linestyle='-', linewidth=linewidth, color='k', label=r'bulk')
plt.plot(S2sim0.p[:-1], S2sim0.betas, linestyle=':', linewidth=linewidth, color='k', label=r'min')

plt.plot(S2sim0.p[:-1], S2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(S2sim1.p[:-1], S2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[2])

plt.plot(S1sim0.p[:-1], S1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[1], label=r'MM3')
plt.plot(S1sim1.p[:-1], S1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[2], label=r'mBas')
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Solid $\beta$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,4,10)
plt.grid(linestyle=':', linewidth=0.5)

plt.plot(S2sim0.t[:-1], S2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[1], label='MM3')
plt.plot(S2sim1.t[:-1], S2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[2], label='mBas')

plt.plot(S1sim0.t[:-1], S1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[1], label='MM3')
plt.plot(S1sim1.t[:-1], S1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[2], label='mBas')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylim([-2.5,2.5])
#plt.legend(fontsize='small')

ax = plt.subplot(4,4,11)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S2sim0.Mgs[:-1], S2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(S2sim1.Mgs[:-1], S2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[2])

plt.plot(S1sim0.Mgs[:-1], S1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.Mgs[:-1], S1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Solid Mg\#', fontsize=fontsize)
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,4,12)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S2sim0.phi[:-1], S2sim0.betas, linestyle=':', linewidth=linewidth, color=colors[1])
plt.plot(S2sim1.phi[:-1], S2sim1.betas, linestyle=':', linewidth=linewidth, color=colors[2])

plt.plot(S1sim0.phi[:-1], S1sim0.betas, linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.phi[:-1], S1sim1.betas, linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.ylim([-2.5,2.5])

ax = plt.subplot(4,4,13)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.p[ind_S1sim0[0][:-1]], S1sim0.betaf[ind_S1sim0[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1], label=r'MM3')
plt.plot(S1sim1.p[ind_S1sim1[0][:-1]], S1sim1.betaf[ind_S1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2], label=r'mBas')
plt.xlabel(r'$P$ (MPa)', fontsize=fontsize)
plt.ylabel(r'Liquid $\beta$', fontsize=fontsize)
plt.legend(fontsize='small')
plt.ylim([-10,10])

ax = plt.subplot(4,4,14)
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.t[ind_S1sim0[0][:-1]], S1sim0.betaf[ind_S1sim0[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1], label='MM3')
plt.plot(S1sim1.t[ind_S1sim1[0][:-1]], S1sim1.betaf[ind_S1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2], label='mBas')
plt.xlabel(r'$T$ ($^o$C)', fontsize=fontsize)
plt.ylim([-10,10])
#plt.legend(fontsize='small')

ax = plt.subplot(4,4,15)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.Mgf[ind_S1sim0[0][:-1]], S1sim0.betaf[ind_S1sim0[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.Mgf[ind_S1sim1[0][:-1]], S1sim1.betaf[ind_S1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid Mg\#', fontsize=fontsize)
plt.ylim([-10,10])

ax = plt.subplot(4,4,16)
# ax = plt.axes()
plt.grid(linestyle=':', linewidth=0.5)
plt.plot(S1sim0.phi[ind_S1sim0[0][:-1]], S1sim0.betaf[ind_S1sim0[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[1])
plt.plot(S1sim1.phi[ind_S1sim1[0][:-1]], S1sim1.betaf[ind_S1sim1[0][:-1]], linestyle='-', linewidth=linewidth, color=colors[2])
plt.xlabel(r'Liquid fraction $\phi$', fontsize=fontsize)
plt.ylim([-10,10])

plt.savefig('density_change_3_isentropic.pdf', bbox_inches = 'tight')


