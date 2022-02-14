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

##########################
# Add path to vizLABconvect
pathViz = '/Users/Adina/Bitbucket/riftomat-private/FD-PDE/models/LABconvect/python'
##########################

sys.path.append(pathViz)
import vizLABconvect as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
# Do print(A.__dict__) to see structure of A
A  = SimStruct()
A0 = SimStruct()

# Parameters
A.dimensional = 1  # 0-nd, 1-dim

##########################
# Directories
sim_name = 'test1'
A.input_path_dir = '../'+sim_name
A.output_path_dir = '../Figures/'+sim_name+'/'
##########################

# Create output directories if necessary
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

# sort timestep list in increasing tstep
tdir.sort(key=sortTimesteps)

time_list_v0 = np.zeros(nt)
time_list = time_list_v0.astype(int)
for ii in range(0,nt):
  time_list[ii] = int(tdir[ii][8:])

##########################
# Create 1D arrays - for time plots
C0mid = np.zeros(nt)
time_kyr = np.zeros(nt)

# Start figure for isotherm
fig1 = plt.figure(1,figsize=(6,4))
color = plt.cm.coolwarm(np.linspace(0,1,nt))
##########################

# Loop over timesteps
itime = 0
for istep in time_list:
  fdir   = A.input_path_dir+'/'+'Timestep'+str(istep)
  fdir0  = A.input_path_dir+'/'+'Timestep0'
  print('  >> >> '+'Timestep'+str(istep))

  # Load timestep data - no harm in loading all necessary data
  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
  A.lbl = vizB.create_labels()

  vizB.correct_path_load_data(fdir0+'/out_xEnth_HS_ts0.py')
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  if (istep>0):
    vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')

  # create some grid parameters
  A.grid = vizB.parse_grid_info('out_xPV_ts'+str(istep),fdir)
  A.dx = A.grid.xc[1]-A.grid.xc[0]
  A.dz = A.grid.zc[1]-A.grid.zc[0]
  A.nx = A.grid.nx
  A.nz = A.grid.nz

  # Parse (load) data
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
  A.P, A.Pc, A.Vsx, A.Vsz = vizB.parse_PV3_file('out_xPV_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  if (istep>0):
    A.matProp = vizB.parse_matProps_file('out_matProp_ts'+str(istep),fdir)

  # Interpolate center velocities
  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
  A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)

  # Load half-space initial data - need separate struct
  A0.Enth = vizB.parse_Enth_file('out_xEnth_HS_ts0',fdir0)
  A.Enth0 = A0.Enth

  # load default enthalpy
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)

  # get scaling variables
  scalx = vizB.get_scaling(A,'x',A.dimensional,1)
  scalv = vizB.get_scaling(A,'v',A.dimensional,1)
  scalt = vizB.get_scaling(A,'t',A.dimensional,1)

  ##########################
  # Extract variables of interest - they are dimensionless in the struct
  # Depth-averaged bulk composition - middle cross section
  mid_ind = int(A.nx/2)
  C0 = A.Enth.C[:,mid_ind] * A.scal.DC + A.scal.C0
#  if (itime==0):
#    print(C0,np.average(C0))
  C0mid[itime] = np.average(C0)
#  print(C0mid[itime])
  time_kyr[itime] = round(A.nd.t*scalt/1.0e3)

  # Plot T1300 isotherm (i.e. depth of isotherm)
  Tisotherm = 1300
  TCelsius  = A.Enth.T * A.scal.DT + A.scal.T0 - 273.15
  zT1300 = np.zeros(A.nx)
  for i in range(0,A.nx):
    ind_z = np.where(TCelsius[:,i]>Tisotherm) # we are interested the end index -1
    zT1300[i] = A.grid.zc[ind_z[0][-1]]*scalx

  # Plot depth of isotherm
  plt.plot(A.grid.xc*scalx,zT1300,color=color[itime])
  ##########################

  # increase time count
  itime += 1

  # remove
  os.system('rm -r '+A.input_path_dir+'/'+'Timestep'+str(istep)+'/__pycache__')

# Inspect data structure - better do this outside loop
# print(A.__dict__)
# print(A.Enth.__dict__)
# print(A.grid.__dict__)
# print(A.lbl.__dict__)

##########################
# Finish plots
plt.grid(True)
plt.xlabel('x [km]')
plt.ylabel('z [km]')
plt.title(r'Isotherm 1300$^o$C')
plt.savefig(A.output_path_dir+'isotherm1300.pdf', bbox_inches = 'tight')
plt.close()

# Plot time integrated data
fig = plt.figure(1,figsize=(5,4))
plt.plot(time_kyr,C0mid)
plt.grid(True)
plt.xlabel('Time [kyr]')
plt.ylabel('C0-mid average')
plt.savefig(A.output_path_dir+'C0mid.pdf', bbox_inches = 'tight')
plt.close()
##########################

os.system('rm -r '+pathViz+'/__pycache__')

