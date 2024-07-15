# Import modules
import os
import sys
import numpy as np

# Add path to vizMORfault
#pathViz = '/home/sann3352/riftomat-morfault/models/morfault/python/'
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
sim = 'buoy00_age2_b1e-2/'

#path_in ='/data/magmox/sann3352/morfault2/'
#path_out='/data/magmox/sann3352/Figures_morfault2/'

path_in ='/Users/apusok/Documents/morfault2/'
path_out='/Users/apusok/Documents/morfault2/Figures/'

A.input = path_in+sim
A.output_path_dir = path_out+sim

# search timesteps in folder
tdir = os.listdir(A.input)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'model_input.opts' in tdir:
  tdir.remove('model_input.opts')
  
if 'submit_job.run' in tdir:
  tdir.remove('submit_job.run')
  
if 'egu23_sims' in tdir:
  tdir.remove('egu23_sims')

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '.out' in s:
    tdir.remove(s)

tdir_check = list.copy(tdir)
for s in tdir_check:
  if 'slurm' in s:
    tdir.remove(s)

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '_r' in s:
    tdir.remove(s)

nt = len(tdir)

# sort list in increasing tstep
tdir.sort(key=sortTimesteps)
time_list_v0 = np.zeros(nt)
time_list = time_list_v0.astype(int)
for ii in range(0,nt):
  time_list[ii] = int(tdir[ii][8:])

# Create directories - default to check for files
A.input_dir = A.input
A.output_dir = A.output_path_dir+'fields_3panels/'

vizB.make_dir(A.output_path_dir)
vizB.make_dir(A.output_dir)

# Read parameters file and get scaling params
istep = time_list[0]
fdir = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)

# Create labels
A.lbl = vizB.create_labels()

# Read grid parameters - do this operation only once
fdir  = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
A.nx = A.grid.nx
A.nz = A.grid.nz

# plot entire domain
istart = 60
iend   = 141
jstart = 0
jend   = A.nz

# check if time_list == number of output files
fout_list = os.listdir(A.output_dir)
if '.DS_Store' in fout_list:
  fout_list.remove('.DS_Store')
nout = len(fout_list)

flg_output = True
if (nt==nout):
  flg_output = False

if (flg_output):
  print('  >> '+A.output_dir+'  >> TRUE')

  # Loop over timesteps
  for istep1 in range(0,nt,1):
    istep = time_list[istep1]
    fdir  = A.input_dir+'Timestep'+str(istep)

    # check if timestep is output or not
    flg_output_ts = False
    for s in fout_list:
      if 'ts'+str(istep) in s:
        flg_output_ts = True

    if (flg_output_ts):
      continue

    print('  >> >> '+'Timestep'+str(istep))

    vizB.correct_path_load_data(fdir+'/parameters.py')
    A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

    # Load and plot markers
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
    A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
    A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
    A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
    A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
    A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
    A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

    # Center velocities and mass divergence
    A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)
    A.rheol = vizB.calc_VEVP_strain_rates(A)

    # Plots
    vizB.plot_3panels_phi_eps_div(A,istart,iend,jstart,jend,A.output_dir,'out_3panels'+str(istep),istep)

    os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')
