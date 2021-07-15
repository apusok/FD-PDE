# Import modules
import os
import sys
import numpy as np

# Add path to vizMORBuoyancy
pathViz = './'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
A.tout  = 1000
A.tstep = 200000
A.istep = 14741
A.dimensional = 1 # 0-nd, 1-dim
check_entire_dir = 1 

A.input = 'b000-hr000_xmor4_fextract0.1'
A.output_path_dir = '../Figures/'
A.path_dir = '../'

# search timesteps in folder
tdir = os.listdir(A.path_dir+A.input)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'log_out.out' in tdir:
  tdir.remove('log_out.out')
if 'model_half_ridge.opts' in tdir:
  tdir.remove('model_half_ridge.opts')
if 'submit_job.run' in tdir:
  tdir.remove('submit_job.run')
nt = len(tdir)

# sort list in increasing tstep
tdir.sort(key=sortTimesteps)

if (check_entire_dir):
  time_list_v0 = np.zeros(nt)
  time_list = time_list_v0.astype(int)
  for ii in range(0,nt):
    time_list[ii] = int(tdir[ii][8:])
else:
  time_list = range(A.istep,A.tstep+1,A.tout)

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'
A.output_dir_real = A.output_path_dir+'/'+A.input+'/debug_output/'

print('# INPUT: '+A.input_dir)
if (A.dimensional):
  print('# Dimensional output: yes')
else:
  print('# Dimensional output: no')

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir_real)
except OSError:
  pass

# Read parameters file and get scaling params
fdir = A.input_dir+'Timestep0'
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)

# Create labels
A.lbl = vizB.create_labels()

# Read grid parameters - choose PV file timestep0
vizB.correct_path_load_data(fdir+'/out_xPV_ts0.py')
A.grid = vizB.parse_grid_info('out_xPV_ts0',fdir)

# For easy access
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
A.nx = A.grid.nx
A.nz = A.grid.nz

# Get time data
# A.ts, A.sol = vizB.parse_solver_log_file(A.input_dir+'log_out_dum.out')

# plot entire domain
istart = 0
iend   = A.nx
jstart = 0
jend   = A.nz

# istart = 0
# iend   = 20 #A.nx
# jstart = A.nz-20 #0
# jend   = A.nz

# istart = 199-20
# iend   = 200+20 #A.nx
# jstart = A.nz-20 #0
# jend   = A.nz

# Loop over timesteps
for istep in time_list: #range(A.istep,A.tstep+1,A.tout):
  fdir  = A.input_dir+'Timestep'+str(istep)

  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

  # Correct path for data
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xHC_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xHCcoeff_ts'+str(istep)+'.py')

  # Get data
  A.P, A.Pc, A.Vsx, A.Vsz = vizB.parse_PV3_file('out_xPV_ts'+str(istep),fdir)
  A.H, A.C = vizB.parse_HC_file('out_xHC_ts'+str(istep),fdir)
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.HC_coeff = vizB.parse_HCcoeff_file('out_xHCcoeff_ts'+str(istep),fdir)

  if (istep > 0):
    # Correct path for data
    vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_resHC_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')

    # Get data
    A.PV_coeff = vizB.parse_PVcoeff3_file('out_xPVcoeff_ts'+str(istep),fdir)
    A.resP, A.resPc, A.resVsx, A.resVsz = vizB.parse_PV3_file('out_resPV_ts'+str(istep),fdir)
    A.resH, A.resC = vizB.parse_HC_file('out_resHC_ts'+str(istep),fdir)
    A.matProp = vizB.parse_matProps_file('out_matProp_ts'+str(istep),fdir)
  
  # Center velocities and mass divergence
  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
  A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)
  A.divmass = vizB.calc_divergence(A.Vx,A.Vz,A.dx,A.dz,A.nx,A.nz)

  # Plots
  if (A.dimensional):
    vizB.plot_porosity_contours(A,istart,iend,jstart,jend,A.output_dir_real+'out_porosity_contours_ts'+str(istep),istep,A.dimensional)
  
  # vizB.plot_temperature_slices(A,A.output_dir_real+'out_temp_slices_ts'+str(istep),istep,A.dimensional)

  # vizB.plot_PV3(0,A,istart,iend,jstart,jend,A.output_dir_real+'out_xPV_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_HC(0,A,istart,iend,jstart,jend,A.output_dir_real+'out_xHC_ts'+str(istep),istep,A.dimensional) # H,C
  vizB.plot_Enth(A,istart,iend,jstart,jend,A.output_dir_real+'out_xEnth_ts'+str(istep),istep,A.dimensional)
  vizB.plot_Vel(A,istart,iend,jstart,jend,A.output_dir_real+'out_xVel_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_HCcoeff(A,istart,iend,jstart,jend,A.output_dir_real+'out_xHCcoeff_ts'+str(istep),istep,A.dimensional)

  if (istep > 0):
    # vizB.plot_PVcoeff3(A,istart,iend,jstart,jend,A.output_dir_real+'out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
    # vizB.plot_PV3(1,A,istart,iend,jstart,jend,A.output_dir_real+'out_resPV_ts'+str(istep),istep,A.dimensional) # res PV
    # vizB.plot_HC(1,A,istart,iend,jstart,jend,A.output_dir_real+'out_resHC_ts'+str(istep),istep,A.dimensional) # res HC
    vizB.plot_matProp(A,istart,iend,jstart,jend,A.output_dir_real+'out_matProp_ts'+str(istep),istep,A.dimensional)

  os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')