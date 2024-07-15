# Import modules
import os
import sys
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A  = SimStruct() # Do print(A.__dict__) to see structure of A
A2 = SimStruct() 
A3 = SimStruct() 

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
istep = 3000
A.dimensional = 1 # 0-nd, 1-dim
sim1 = 'run41_00_S_ref/'
sim2 = 'run41_01_SD_setup6_sigmabc1e-2/'
sim3 = 'run41_01_SD_setup6_sigmabc1e-1/'

A.input_dir = '/Users/apusok/Documents/morfault/'+sim1
A.output_dir = '/Users/apusok/Documents/morfault/Figures/'+sim1

# Create directories - default to check for files
vizB.make_dir(A.output_dir)

A2.dimensional = 1
A2.input_dir = '/Users/apusok/Documents/morfault/'+sim2

A3.dimensional = 1
A3.input_dir = '/Users/apusok/Documents/morfault/'+sim3

# ------------------
# sim1 'Timestep'+str(istep)
# ------------------
fdir = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
A.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
A.nx = A.grid.nx
A.nz = A.grid.nz

vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
# vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphiguess_ts'+str(istep)+'.py')

# Get data
# A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
# A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
# A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
# A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
# A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
# A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
# A.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
# A.P_res, A.Vsx_res, A.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
# A.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
# A.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
# A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
# A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
# A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
# A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
# A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
# A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
# A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
# A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)
# A.phisguess = vizB.parse_Element_file('out_xphiguess_ts'+str(istep),fdir)

# ------------------
# sim2 'Timestep'+str(istep)
# ------------------
fdir = A2.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A2.scal, A2.nd, A2.geoscal = vizB.parse_parameters_file('parameters',fdir)
A2.nd.istep, A2.nd.dt, A2.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
A2.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A2.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

A2.dx = A2.grid.xc[1]-A2.grid.xc[0]
A2.dz = A2.grid.zc[1]-A2.grid.zc[0]
A2.nx = A2.grid.nx
A2.nz = A2.grid.nz

vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A2.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
# vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphiguess_ts'+str(istep)+'.py')

# Get data
# A2.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
# A2.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
# A2.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
# A2.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
# A2.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
# A2.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
# A2.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A2.P, A2.Vsx, A2.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
# A2.P_res, A2.Vsx_res, A2.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
# A2.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
# A2.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
# A2.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
# A2.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
# A2.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
# A2.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# # A2.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
A2.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
# A2.Vfx, A2.Vfz, A2.Vx, A2.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
# A2.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
# A2.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
# A2.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A2.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)
# A2.phisguess = vizB.parse_Element_file('out_xphiguess_ts'+str(istep),fdir)

# ------------------
# sim3 'Timestep'+str(istep)
# ------------------
fdir = A3.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A3.scal, A3.nd, A3.geoscal = vizB.parse_parameters_file('parameters',fdir)
A3.nd.istep, A3.nd.dt, A3.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
A3.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A3.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

A3.dx = A3.grid.xc[1]-A3.grid.xc[0]
A3.dz = A3.grid.zc[1]-A3.grid.zc[0]
A3.nx = A3.grid.nx
A3.nz = A3.grid.nz

vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A3.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
# vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphiguess_ts'+str(istep)+'.py')

# Get data
# A3.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
# A3.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
# A3.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
# A3.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
# A3.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
# A3.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
# A3.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A3.P, A3.Vsx, A3.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
# A3.P_res, A3.Vsx_res, A3.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
# A3.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
# A3.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
# A3.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
# A3.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
# A3.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
# A3.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# # A3.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
A3.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
# A3.Vfx, A3.Vfz, A3.Vx, A3.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
# A3.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
# A3.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
# A3.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
# A3.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)
# A3.phisguess = vizB.parse_Element_file('out_xphiguess_ts'+str(istep),fdir)

# ------------------
# Comparison
# ------------------
scalx = vizB.get_scaling(A,'x',1,1)
scaleta = vizB.get_scaling(A,'eta',1,0)
scalrho = vizB.get_scaling(A,'rho',1,0)
lblz = vizB.get_label(A,'z',1)
lblx = vizB.get_label(A,'x',1)

topo1 = np.zeros(A.nx)
topo2 = np.zeros(A.nx)
topo3 = np.zeros(A.nx)

for i in range(0,A.nx):
  ztopo = 0.0
  for j in range(0,A.nz):
    if (A.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A.grid.zc[j]*scalx)
  topo1[i] = ztopo

for i in range(0,A2.nx):
  ztopo = 0.0
  for j in range(0,A2.nz):
    if (A2.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A2.grid.zc[j]*scalx)
  topo2[i] = ztopo

for i in range(0,A3.nx):
  ztopo = 0.0
  for j in range(0,A3.nz):
    if (A3.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A3.grid.zc[j]*scalx)
  topo3[i] = ztopo

# Topography using markers
x_mark = np.zeros(A.nx*2)
topo1_mark = np.zeros(A.nx*2)
topo2_mark = np.zeros(A.nx*2)
topo3_mark = np.zeros(A.nx*2)

for i in range(0,A.nx):
  x_mark[2*i  ] = A.grid.xv[i]
  x_mark[2*i+1] = A.grid.xc[i]

dx2 = (x_mark[1]-x_mark[0])*0.5

for i in range(0,A.nx*2):
  ztopo = 0.0
  for j in range(0,A.mark.n):
    if (A.mark.x[j]>=x_mark[i]-dx2) & (A.mark.x[j]<x_mark[i]+dx2) & (A.mark.id[j]==0):
      ztopo = min(ztopo,A.mark.z[j]*scalx)
  topo1_mark[i] = ztopo

for i in range(0,A.nx*2):
  ztopo = 0.0
  for j in range(0,A2.mark.n):
    if (A2.mark.x[j]>=x_mark[i]-dx2) & (A2.mark.x[j]<x_mark[i]+dx2) & (A2.mark.id[j]==0):
      ztopo = min(ztopo,A2.mark.z[j]*scalx)
  topo2_mark[i] = ztopo

for i in range(0,A.nx*2):
  ztopo = 0.0
  for j in range(0,A3.mark.n):
    if (A3.mark.x[j]>=x_mark[i]-dx2) & (A3.mark.x[j]<x_mark[i]+dx2) & (A3.mark.id[j]==0):
      ztopo = min(ztopo,A3.mark.z[j]*scalx)
  topo3_mark[i] = ztopo

# print(x_mark)
# print(topo1_mark)

# Plotting
fig = plt.figure(1,figsize=(10,3))

ax = plt.subplot(1,1,1)
# pl = ax.plot(A.grid.xc*scalx,topo1,'k-',label='previous')
# pl = ax.plot(A.grid.xc*scalx,topo2,'b--',label='Stokes')
# pl = ax.plot(A.grid.xc*scalx,topo3,'r--',label='Stokes-Darcy')
pl = ax.plot(x_mark*scalx,topo1_mark,'k-',label='amagmatic')
pl = ax.plot(x_mark*scalx,topo2_mark,'b--',label=r'$\sigma_{BC}=0.01$')
pl = ax.plot(x_mark*scalx,topo3_mark,'r--',label=r'$\sigma_{BC}=0.1$')
plt.grid(True)
ax.set_ylabel(lblz)
ax.set_title('Water-rock interface level')
ax.set_xlabel(lblx)
ax.legend()
ax.set_ylim(-20,0)

plt.savefig(A.output_dir+'out_compare_mark_run41_topo_ts'+str(istep)+'.png', bbox_inches = 'tight')
plt.close()