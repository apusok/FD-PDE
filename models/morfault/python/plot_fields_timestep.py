# Import modules
import os
import sys
import numpy as np

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
A.dimensional = 1 # 0-nd, 1-dim
sim = 'run52_V_var_age_eta1e18_Vext1/'
A.input_dir = '/Users/apusok/Documents/morfault/'+sim
A.output_dir= '/Users/apusok/Documents/morfault/Figures/'+sim

istep = 0

# Create directory
vizB.make_dir(A.output_dir)

# Read parameters file and get scaling params
fdir = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
A.lbl = vizB.create_labels()

# Read grid parameters
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
A.nx = A.grid.nx
A.nz = A.grid.nz

# plot entire domain
istart = 60
iend   = 140 #A.nx
jstart = 0
jend   = A.nz

print(A.input_dir)
print('  >> >> '+'Timestep'+str(istep))

# Load data
A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

# Load and plot markers
vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
# vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# Get data
A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
# A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
# A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
# A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
A.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
# A.P_res, A.Vsx_res, A.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
# A.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
# A.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
# A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
# A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
# A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
# A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
# A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# Center velocities and mass divergence
A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

# Plots
vizB.plot_phi_eps_div_eta_zeta_lam(A,istart,iend,jstart,jend,A.output_dir,'out_phi_eps_div_eta_zeta_lam_ts'+str(istep),istep,A.dimensional)

os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')