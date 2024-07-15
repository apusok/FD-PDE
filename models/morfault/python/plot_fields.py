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
sim = 'run41_01_SD_setup6_sigmabc1e-2/'
A.input = '/Users/apusok/Documents/morfault/'+sim
A.output_path_dir = '/Users/apusok/Documents/morfault/Figures/'+sim

# search timesteps in folder
tdir = os.listdir(A.input)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'model_input.opts' in tdir:
  tdir.remove('model_input.opts')

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '.out' in s:
    tdir.remove(s)

tdir_check = list.copy(tdir)
for s in tdir_check:
  if 'orig' in s:
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
# A.output_dir = A.output_path_dir+'fields2/'
A.output_dir = A.output_path_dir+'plastic2/'  

vizB.make_dir(A.output_path_dir)
vizB.make_dir(A.output_dir)

# Read parameters file and get scaling params
istep = time_list[0]
fdir = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
# A.scal, A.nd, A.geoscal = vizB.create_scaling() # if read from params file is not done

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
istart = 0
iend   = A.nx
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
  # for istep in time_list:
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
    # vizB.plot_marker_id(A,istart,iend,jstart,jend,A.output_path_dir+'pic/','out_pic_ts'+str(istep),istep,A.dimensional)

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
    # A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
    # A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
    A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
    A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
    A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
    A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
    A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
    A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

    # Center velocities and mass divergence
    A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

    # Calculate individual deformation strain rates
    # A.rheol = vizB.calc_dom_rheology_mechanism(A)

    # vizB.plot_mark_eps_phi(A,istart,iend,jstart,jend,A.output_dir,'out_mark_eps_phi_ts'+str(istep),istep,A.dimensional)

    # Plots: A.output_path_dir+'fields2/'
    # vizB.plot_mark_eta_eps_tau(A,istart,iend,jstart,jend,A.output_path_dir+'fields0/','out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_mark_eta_eps_tau2(A,istart,iend,jstart,jend,A.output_path_dir+'fields00/','out_mark_eta_eps_tau2_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_mark_eta_eps_tau_T_phi(A,istart,iend,jstart,jend,A.output_path_dir+'fields1/','out_mark_eta_eps_tau_T_phi_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_def_mechanisms(A,istart,iend,jstart,jend,A.output_path_dir+'def_mech/','out_def_mechanisms_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_mark_eps_phi(A,istart,iend,jstart,jend,A.output_dir,'out_mark_eps_phi_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_mark_divs_phi(A,istart,iend,jstart,jend,A.output_dir,'out_mark_divs_phi_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_mark_eps_phi_column3(A,istart,iend,jstart,jend,A.output_dir,'out_mark_eps_phi_ts'+str(istep),istep,A.dimensional)

    # vizB.plot_T(A,istart,iend,jstart,jend,A.output_path_dir+'T/','out_xT_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_MPhase(A,istart,iend,jstart,jend,A.output_path_dir+'MPhase/','out_xMPhase_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_P(A,istart,iend,jstart,jend,A.output_path_dir+'P/','out_xP_ts'+str(istep),istep,A.dimensional,3)
    # vizB.plot_PV(A,istart,iend,jstart,jend,A.output_path_dir+'PV/','out_xPV_ts'+str(istep),istep,A.dimensional,0)
    # vizB.plot_PV(A,istart,iend,jstart,jend,A.output_path_dir+'resPV/','out_resPV_ts'+str(istep),istep,A.dimensional,1)
    # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'eps/','out_xeps_ts'+str(istep),istep,A.dimensional,0)
    # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'tau/','out_xtau_ts'+str(istep),istep,A.dimensional,1)
    # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'tauold/','out_xtauold_ts'+str(istep),istep,A.dimensional,2)
    # vizB.plot_PVcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'PVcoeff/','out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_PVcoeff_Stokes(A,istart,iend,jstart,jend,A.output_path_dir+'PVcoeff/','out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_matProp(A,istart,iend,jstart,jend,A.output_path_dir+'matProp/','out_matProp_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_Vel(A,istart,iend,jstart,jend,A.output_path_dir+'Vel/','out_xVel_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_Tcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'Tcoeff/','out_xTcoeff_ts'+str(istep),istep,A.dimensional,0)
    # vizB.plot_Tcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'phicoeff/','out_xphicoeff_ts'+str(istep),istep,A.dimensional,1)
    # vizB.plot_plastic(A,istart,iend,jstart,jend,A.output_path_dir+'plastic/','out_xplastic_ts'+str(istep),istep,A.dimensional)
    vizB.plot_plastic_v2(A,istart,iend,jstart,jend,A.output_path_dir+'plastic2/','out_xplastic_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_individual_eps(A,istart,iend,jstart,jend,A.output_path_dir+'eps_ind/','out_individual_eps_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_phi(A,istart,iend,jstart,jend,A.output_path_dir+'phi/','out_xphi_ts'+str(istep),istep,A.dimensional)

    os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')