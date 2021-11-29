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

# Add path to vizLABconvect
pathViz = '/Users/Adina/Bitbucket/riftomat-private/FD-PDE/models/LABconvect/python'

sys.path.append(pathViz)
import vizLABconvect as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

# Parameters
A.dimensional     = 0  # 0-nd, 1-dim

# Directories
sim_name = 'test'
A.input_path_dir = '../'+sim_name+'/'
A.output_path_dir = '../Figures/'+sim_name+'/'

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

def sortTimesteps(tdir):
  return int(tdir[8:])

# get timesteps
tdir = os.listdir(A.input_path_dir)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'LABconvect_input.opts' in tdir:
  tdir.remove('LABconvect_input.opts')
if 'submit_job.run' in tdir:
  tdir.remove('submit_job.run')

# remove slurm output
tdir_check = list.copy(tdir)
for s in tdir_check:
  if 'slurm' in s:
    tdir.remove(s)

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '.out' in s:
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

print('  >> '+sim_name)
# Loop over timesteps
for istep in time_list:
  fdir  = A.input_path_dir+'/'+'Timestep'+str(istep)
  print('  >> >> '+'Timestep'+str(istep))
  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
  A.lbl = vizB.create_labels()

  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
  if (istep==0):
    vizB.correct_path_load_data(fdir+'/out_xEnth_HS_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  if (istep>0):
    vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')

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
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  if (istep>0):
    A.matProp = vizB.parse_matProps_file('out_matProp_ts'+str(istep),fdir)

  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
  A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)

  if (istep==0):
    A.Enth = vizB.parse_Enth_file('out_xEnth_HS_ts'+str(istep),fdir)
    A.T_HS = A.Enth.T
    fmsg = A.output_path_dir+'/out_fields_'+sim_name+'_HS'
    vizB.plot_fields_model(A,istart,iend,jstart,jend,fmsg+'_ts'+str(istep),istep,A.dimensional)
    T_HS = A.Enth.T

  A.T_HS = T_HS

  # load default enthalpy
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)

  fmsg = A.output_path_dir+'/out_fields_'+sim_name
  if (A.dimensional==1):
    vizB.plot_fields_model(A,istart,iend,jstart,jend,fmsg+'_ts'+str(istep),istep,A.dimensional)
  else:
    vizB.plot_fields_model_nd(A,istart,iend,jstart,jend,fmsg+'_ts'+str(istep),istep,A.dimensional)

  # remove
  os.system('rm -r '+A.input_path_dir+'/'+'Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')

