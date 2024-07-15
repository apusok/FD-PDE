# Import modules
import os
import sys
import numpy as np

# Add path to vizLABconvect
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
sim = 'run15_VEVP_03_AvgPh_Plith_etaVharm_etaK1e20_sigmatC4/'
iout = 0
A.input = '../'+sim
A.output_path_dir = '../Figures/'+sim
A.path_dir = './'

# # search timesteps in folder
# tdir = os.listdir(A.path_dir+A.input)
# if '.DS_Store' in tdir:
#   tdir.remove('.DS_Store')
# if 'log_out.out' in tdir:
#   tdir.remove('log_out.out')
# if 'model_input.opts' in tdir:
#   tdir.remove('model_input.opts')
# nt = len(tdir)

# # sort list in increasing tstep
# tdir.sort(key=sortTimesteps)
# time_list_v0 = np.zeros(nt)
# time_list = time_list_v0.astype(int)
# for ii in range(0,nt):
#   time_list[ii] = int(tdir[ii][8:])

# Create directories
A.input_dir = A.path_dir+A.input
A.output_dir = A.output_path_dir

# print('# INPUT: '+A.input_dir)
# if (A.dimensional):
#   print('# Dimensional output: yes')
# else:
#   print('# Dimensional output: no')

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

# # Read parameters file and get scaling params
# fdir = A.input_dir+'Timestep0'
# vizB.correct_path_load_data(fdir+'/parameters.py')
# A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
A.scal, A.nd, A.geoscal = vizB.create_scaling() # if read from params file is not done

# Create labels
A.lbl = vizB.create_labels()

# Read grid parameters - choose PV file timestep0
istep = 0
fdir  = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
A.nx = A.grid.nx
A.nz = A.grid.nz

# plot entire domain
istart = 0
iend   = A.nx
jstart = 0
jend   = A.nz

# Loop over timesteps
for istep in range(iout,100001,1):
  fdir  = A.input_dir+'Timestep'+str(istep)
  print(fdir)

  #   vizB.correct_path_load_data(fdir+'/parameters.py')
  #   A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

  # Load and plot markers
  vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
  A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)
  # vizB.plot_marker_id(A,istart,iend,jstart,jend,A.output_dir+'out_pic_ts'+str(istep),istep,A.dimensional)

  # Correct path for data
  vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
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
  vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

  # Get data
  A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
  A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
  A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
  A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
  A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
  A.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.P_res, A.Vsx_res, A.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
  A.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
  A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
  A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
  A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
  # A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
  A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
  A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
  A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

  # Center velocities and mass divergence
  A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

  # Plots
  vizB.plot_mark_eta_eps_tau(A,istart,iend,jstart,jend,A.output_dir+'out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)

  vizB.plot_T(A,istart,iend,jstart,jend,A.output_dir+'out_xT_ts'+str(istep),istep,A.dimensional)
  vizB.plot_MPhase(A,istart,iend,jstart,jend,A.output_dir+'out_xMPhase_ts'+str(istep),istep,A.dimensional)
  vizB.plot_P(A,istart,iend,jstart,jend,A.output_dir+'out_xP_ts'+str(istep),istep,A.dimensional,3)
  vizB.plot_PV(A,istart,iend,jstart,jend,A.output_dir+'out_xPV_ts'+str(istep),istep,A.dimensional,0)
  vizB.plot_PV(A,istart,iend,jstart,jend,A.output_dir+'out_resPV_ts'+str(istep),istep,A.dimensional,1)
  vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_dir+'out_xeps_ts'+str(istep),istep,A.dimensional,0)
  vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_dir+'out_xtau_ts'+str(istep),istep,A.dimensional,1)
  # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_dir+'out_xtauold_ts'+str(istep),istep,A.dimensional,2)
  # vizB.plot_PVcoeff(A,istart,iend,jstart,jend,A.output_dir+'out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
  vizB.plot_PVcoeff_Stokes(A,istart,iend,jstart,jend,A.output_dir+'out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
  vizB.plot_matProp(A,istart,iend,jstart,jend,A.output_dir+'out_matProp_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_Vel(A,istart,iend,jstart,jend,A.output_dir+'out_xVel_ts'+str(istep),istep,A.dimensional)
  vizB.plot_Tcoeff(A,istart,iend,jstart,jend,A.output_dir+'out_xTcoeff_ts'+str(istep),istep,A.dimensional)
  vizB.plot_plastic(A,istart,iend,jstart,jend,A.output_dir+'out_xplastic_ts'+str(istep),istep,A.dimensional)

  os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')

