# Import modules
import os
import sys
import numpy as np
from numpy import linalg as la

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
Ar = SimStruct() 

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
istep = 10
A.dimensional = 1 # 0-nd, 1-dim
sim = 'run41_01_SD_setup2_phibc5e-3_sigmabc1e-3_test/'
A.input_dir = '/Users/apusok/Documents/morfault/'+sim
A.output_dir = '/Users/apusok/Documents/morfault/Figures/'+sim

# Create directories - default to check for files
vizB.make_dir(A.output_dir)

Ar.dimensional = 1
Ar.input_dir = A.input_dir

# ------------------
# 'Timestep'+str(istep)
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
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# vizB.correct_path_load_data(fdir+'/out_xphiprev_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphiprevcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphiguess_ts'+str(istep)+'.py')

# Get data
A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
A.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
A.P_res, A.Vsx_res, A.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
A.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
A.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# A.phisprev = vizB.parse_Element_file('out_xphiprev_ts'+str(istep),fdir)
# A.phicoeffprev = vizB.parse_Tcoeff_file('out_xphiprevcoeff_ts'+str(istep),fdir)
A.phisguess = vizB.parse_Element_file('out_xphiguess_ts'+str(istep),fdir)

# ------------------
# RESTART Timestep
# ------------------
fdir = Ar.input_dir+'Timestep'+str(istep)+'_r'
vizB.correct_path_load_data(fdir+'/parameters.py')
Ar.scal, Ar.nd, Ar.geoscal = vizB.parse_parameters_file('parameters',fdir)
Ar.nd.istep, Ar.nd.dt, Ar.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
Ar.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
Ar.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

Ar.dx = Ar.grid.xc[1]-Ar.grid.xc[0]
Ar.dz = Ar.grid.zc[1]-Ar.grid.zc[0]
Ar.nx = Ar.grid.nx
Ar.nz = Ar.grid.nz

vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
Ar.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# vizB.correct_path_load_data(fdir+'/out_xphiprev_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphiprevcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphiguess_ts'+str(istep)+'.py')

# Get data
Ar.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
Ar.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
Ar.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
Ar.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
Ar.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
Ar.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
Ar.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
Ar.P, Ar.Vsx, Ar.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
Ar.P_res, Ar.Vsx_res, Ar.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
Ar.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
Ar.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
Ar.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
Ar.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
Ar.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
Ar.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# Ar.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
Ar.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
Ar.Vfx, Ar.Vfz, Ar.Vx, Ar.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
Ar.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
Ar.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
Ar.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
Ar.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# Ar.phisprev = vizB.parse_Element_file('out_xphiprev_ts'+str(istep),fdir)
# Ar.phicoeffprev = vizB.parse_Tcoeff_file('out_xphiprevcoeff_ts'+str(istep),fdir)
Ar.phisguess = vizB.parse_Element_file('out_xphiguess_ts'+str(istep),fdir)

# ------------------
# Calculate norms 
# ------------------
print('\n>> '+A.input_dir)
print('>> Timestep'+str(istep)+'_r')
print('NORMS:')
print('|dt-dt_r| = ',A.nd.dt-Ar.nd.dt)
print('|P|2 = ',la.norm(A.P-Ar.P))
print('|Vsx|2 = ',la.norm(A.Vsx-Ar.Vsx))
print('|Vsz|2 = ',la.norm(A.Vsz-Ar.Vsz))
print('|T|2 = ',la.norm(A.T-Ar.T))
print('|phi|2 = ',la.norm(A.phis-Ar.phis))
print('|phicoeff-A|2 = ',la.norm(A.phicoeff.A-Ar.phicoeff.A))
print('|phicoeff-C|2 = ',la.norm(A.phicoeff.C-Ar.phicoeff.C))
print('|phicoeff-Bx|2 = ',la.norm(A.phicoeff.Bx-Ar.phicoeff.Bx))
print('|phicoeff-Bz|2 = ',la.norm(A.phicoeff.Bz-Ar.phicoeff.Bz))
print('|phicoeff-ux|2 = ',la.norm(A.phicoeff.ux-Ar.phicoeff.ux))
print('|phicoeff-uz|2 = ',la.norm(A.phicoeff.uz-Ar.phicoeff.uz))
print('|phiguess|2 = ',la.norm(A.phisguess-Ar.phisguess))

# print('|phi_prev|2 = ',la.norm(A.phisprev-Ar.phisprev))
# print('|phicoeff_prev-A|2 = ',la.norm(A.phicoeffprev.A-Ar.phicoeffprev.A))
# print('|phicoeff_prev-C|2 = ',la.norm(A.phicoeffprev.C-Ar.phicoeffprev.C))
# print('|phicoeff_prev-Bx|2 = ',la.norm(A.phicoeffprev.Bx-Ar.phicoeffprev.Bx))
# print('|phicoeff_prev-Bz|2 = ',la.norm(A.phicoeffprev.Bz-Ar.phicoeffprev.Bz))
# print('|phicoeff_prev-ux|2 = ',la.norm(A.phicoeffprev.ux-Ar.phicoeffprev.ux))
# print('|phicoeff_prev-uz|2 = ',la.norm(A.phicoeffprev.uz-Ar.phicoeffprev.uz))
