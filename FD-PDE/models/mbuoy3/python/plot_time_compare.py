# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add path to vizMORBuoyancy
pathViz = './'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main - compare time series for n simulations
# ---------------------------------------
A = SimStruct()

A.input = 'compare'
A.output_path_dir = '../sims_full_ridge/Figures/'
A.path_dir = '../sims_full_ridge/'

log_file1 ='half_ridge_f0005'
log_file2 ='half_ridge_f0005_hc_cycles1'
# log_file3 ='half_ridge_f001_xmor3'
# log_file4 ='half_ridge_f0005'

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'

print('# OUTPUT: Time series '+A.input_dir)

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

# file 1
A.ts1, A.sol1 = vizB.parse_solver_log_file(A.input_dir+'log_out_'+log_file1+'.out')
A.flux1 = vizB.parse_outflux_log_file(A.input_dir+'log_out_'+log_file1+'.out')

# file 2
A.ts2, A.sol2 = vizB.parse_solver_log_file(A.input_dir+'log_out_'+log_file2+'.out')
A.flux2 = vizB.parse_outflux_log_file(A.input_dir+'log_out_'+log_file2+'.out')

# # file 3
# A.ts3, A.sol3 = vizB.parse_solver_log_file(A.input_dir+'log_out_'+log_file3+'.out')
# A.flux3 = vizB.parse_outflux_log_file(A.input_dir+'log_out_'+log_file3+'.out')

# # file 4
# A.ts4, A.sol4 = vizB.parse_solver_log_file(A.input_dir+'log_out_'+log_file4+'.out')
# A.flux4 = vizB.parse_outflux_log_file(A.input_dir+'log_out_'+log_file4+'.out')

# plot compare - solver residuals
fig = plt.figure(1,figsize=(10,10))

lbl1 = 'half ridge f=0.005 xmor=5 hc cycles=10'
lbl2 = 'half ridge f=0.005 xmor=5 hc cycles=1'
# lbl3 = 'half ridge f=0.01 xmor=3'
# lbl4 = 'half ridge f=0.005 xmor=5'

ax = plt.subplot(3,1,1)
ax.plot(np.arange(1,A.ts1,1), np.log10(A.sol1.HCres[1:-1]),'k', linewidth=0.5, label=lbl1)
ax.plot(np.arange(1,A.ts2,1), np.log10(A.sol2.HCres[1:-1]),'r', linewidth=0.5, label=lbl2)
# ax.plot(np.arange(1,A.ts3,1), np.log10(A.sol3.HCres[1:-1]),'b', linewidth=0.5, label=lbl3)
# ax.plot(np.arange(1,A.ts4,1), np.log10(A.sol4.HCres[1:-1]),'g', linewidth=0.5, label=lbl4)
plt.grid(True)
plt.legend()
ax.set_xlabel('Timestep')
ax.set_ylabel('log10(HC residual)')

ax = plt.subplot(3,1,2)
ax.plot(np.arange(1,A.ts1,1), np.log10(A.sol1.PVres[1:-1]),'k', linewidth=0.1, label=lbl1)
ax.plot(np.arange(1,A.ts2,1), np.log10(A.sol2.PVres[1:-1]),'r', linewidth=0.1, label=lbl2)
# ax.plot(np.arange(1,A.ts3,1), np.log10(A.sol3.PVres[1:-1]),'b', linewidth=0.1, label=lbl3)
# ax.plot(np.arange(1,A.ts4,1), np.log10(A.sol4.PVres[1:-1]),'g', linewidth=0.1, label=lbl4)
plt.grid(True)
plt.legend()
ax.set_xlabel('Timestep')
ax.set_ylabel('log10(PV residual)')

ax = plt.subplot(3,1,3)
ax.plot(np.arange(1,A.ts1,1), np.log10(A.sol1.dt[1:-1]),'k', linewidth=0.5, label=lbl1)
ax.plot(np.arange(1,A.ts2,1), np.log10(A.sol2.dt[1:-1]),'r', linewidth=0.5, label=lbl2)
# ax.plot(np.arange(1,A.ts3,1), np.log10(A.sol3.dt[1:-1]),'b', linewidth=0.5, label=lbl3)
# ax.plot(np.arange(1,A.ts4,1), np.log10(A.sol4.dt[1:-1]),'g', linewidth=0.5, label=lbl4)
plt.grid(True)
plt.legend()
ax.set_xlabel('Timestep')
ax.set_ylabel('log10(dt)')

plt.savefig(A.output_dir+'out_solver_residuals'+'.pdf', bbox_inches = 'tight')
plt.close()

# plot compare - outflux
fig = plt.figure(1,figsize=(10,10))

ax = plt.subplot(3,1,1)
ax.plot(A.flux1.t[1:-1], A.flux1.h[1:-1]/1000,'k', linewidth=0.5, label=lbl1)
ax.plot(A.flux2.t[1:-1], A.flux2.h[1:-1]/1000,'r', linewidth=0.5, label=lbl2)
# ax.plot(A.flux3.t[1:-1], A.flux3.h[1:-1]/1000,'b', linewidth=0.5, label=lbl3)
# ax.plot(A.flux4.t[1:-1], A.flux4.h[1:-1]/1000,'g', linewidth=0.5, label=lbl4)
plt.grid(True)
# plt.ylim([0, 10])
plt.legend()
ax.set_xlabel('Time [yr]')
ax.set_ylabel('Crustal thickness [km]')

ax = plt.subplot(3,1,2)
ax.plot(A.flux1.t[1:-1], A.flux1.F[1:-1],'k', linewidth=0.5, label=lbl1)
ax.plot(A.flux2.t[1:-1], A.flux2.F[1:-1],'r', linewidth=0.5, label=lbl2)
# ax.plot(A.flux3.t[1:-1], A.flux3.F[1:-1],'b', linewidth=0.5, label=lbl3)
# ax.plot(A.flux4.t[1:-1], A.flux4.F[1:-1],'g', linewidth=0.5, label=lbl4)
plt.grid(True)
plt.legend()
ax.set_xlabel('Time [yr]')
ax.set_ylabel('Flux out [kg/m/yr]')

ax = plt.subplot(3,1,3)
ax.plot(A.flux1.t[1:-1], A.flux1.C[1:-1],'k', linewidth=0.5, label=lbl1)
ax.plot(A.flux2.t[1:-1], A.flux2.C[1:-1],'r', linewidth=0.5, label=lbl2)
# ax.plot(A.flux3.t[1:-1], A.flux3.C[1:-1],'b', linewidth=0.5, label=lbl3)
# ax.plot(A.flux4.t[1:-1], A.flux4.C[1:-1],'g', linewidth=0.5, label=lbl4)
plt.grid(True)
plt.legend()
ax.set_xlabel('Time [yr]')
ax.set_ylabel('C out [wt. frac.]')

plt.savefig(A.output_dir+'out_outflux'+'.pdf', bbox_inches = 'tight')
plt.close()

