# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import warnings
warnings.filterwarnings('ignore')

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

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

# Parameters
A.time_plot       = 1e6 # years
A.dimensional     = 1   # 0-nd, 1-dim

g        = 9.8
drho     = 500
phi0     = 0.2
DC       = 0.1
rho0     = 3300
kappa    = 1.0e-6
SEC_YEAR = 31536000

# Directories
A.output_path_dir = '../Figures/'
A.input_path_dir = '../'

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass
  
A.output_path_dir0 = A.output_path_dir+'porosity/'

try:
  os.mkdir(A.output_path_dir0)
except OSError:
  pass

def sortTimesteps(tdir):
  return int(tdir[8:])

# search sims in folder
sims = os.listdir(A.input_path_dir)
if '.DS_Store' in sims:
  sims.remove('.DS_Store')

#sims_check = list.copy(sims)
#for s in sims_check:
#  if 'b120' in s:
#    sims.remove(s)

sims = ['test']

sims.sort()
nsims = len(sims)

# iterate for all sims
for i in range(0,nsims):
  idir = sims[i]
  
  # get timesteps
  tdir = os.listdir(A.input_path_dir+idir)
  if '.DS_Store' in tdir:
    tdir.remove('.DS_Store')
  if 'model_half_ridge.opts' in tdir:
    tdir.remove('model_half_ridge.opts')
  if 'model_full_ridge.opts' in tdir:
    tdir.remove('model_full_ridge.opts')
  if 'submit_job.run' in tdir:
    tdir.remove('submit_job.run')
  if 'log_out.out' in tdir:
    tdir.remove('log_out.out')

  # remove slurm output
  tdir_check = list.copy(tdir)
  for s in tdir_check:
    if 'slurm' in s:
      tdir.remove(s)
      
  # remove restart timesteps
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
    
  #### -------------- POROSITY -------------- ###
  # create output dir
  A.output_dir = A.output_path_dir0+idir
  try:
    os.mkdir(A.output_dir)
  except OSError:
    pass

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
    for istep in time_list:
      fdir  = A.input_path_dir+idir+'/'+'Timestep'+str(istep)

      # check if timestep is output or not
      flg_output_ts = False
      for s in fout_list:
        if 'ts'+str(istep) in s:
          flg_output_ts = True

      if (flg_output_ts):
        continue

      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
      A.lbl = vizB.create_labels()

      vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')

      A.grid = vizB.parse_grid_info('out_xPV_ts'+str(istep),fdir)
      A.dx = A.grid.xc[1]-A.grid.xc[0]
      A.dz = A.grid.zc[1]-A.grid.zc[0]
      A.nx = A.grid.nx
      A.nz = A.grid.nz

      # Resolution of plots
      istart = 0
      iend   = A.nx
      jstart = 0
      jend   = A.nz

      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
      A.P, A.Pc, A.Vsx, A.Vsz = vizB.parse_PV3_file('out_xPV_ts'+str(istep),fdir)
      A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
      A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)

      A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
      A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
      A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)

      fmsg = A.output_dir+'/out_porosity_'+idir
      vizB.plot_porosity_half_ridge(A,istart,iend,jstart,jend,fmsg+'_ts'+str(istep),istep,A.dimensional)

      # remove
      os.system('rm -r '+A.input_path_dir+idir+'/'+'Timestep'+str(istep)+'/__pycache__')
  else:
    print('  >> '+A.output_dir+'  >> FALSE')

os.system('rm -r '+pathViz+'/__pycache__')

