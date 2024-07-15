# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class SimStruct:
  pass

def sortTimesteps(tdir):
  return int(tdir[8:])

# ---------------------------------
def load_porosity_data(A):
  try: 
    A.input_dir = A.path_dir+A.input
    A.dimensional = 1

    # search timesteps in folder
    tdir = os.listdir(A.path_dir+A.input)
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
      if '_r' in s:
        tdir.remove(s)

    nt = len(tdir)
    A.nt = nt

    # sort list in increasing tstep
    tdir.sort(key=sortTimesteps)
    time_list_v0 = np.zeros(nt)
    time_list = time_list_v0.astype(int)
    for ii in range(0,nt):
      time_list[ii] = int(tdir[ii][8:])

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

    A.Vphi    = np.zeros(A.nt)
    A.phi_max = np.zeros(A.nt)
    A.t_kyr   = np.zeros(A.nt)
    A.z_max   = np.zeros(A.nt)
    
    scalx = vizB.get_scaling(A,'x',1,1)
    scalt = vizB.get_scaling(A,'t',1,1)

    # Loop over timesteps
    it = 0
    for istep in time_list:
      fdir  = A.input_dir+'Timestep'+str(istep)
      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

      # Correct path for data
      # vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
      
      # Get data
      # A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
      A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
      A.phi = 1.0 - A.phis
      # print(A.phi[60:100  ,40:60])

      # Magma parameters
      A.t_kyr[it]   = A.nd.t*scalt/1.0e3
      A.Vphi[it]    = np.sum(A.phi)*A.dx*A.dz*scalx**2
      A.phi_max[it] = np.max(A.phi)

      zdike = np.zeros(A.nx)
      for i in range(0,A.nx):
        zmax = -50.0
        for j in range(0,A.nz):
          if (A.phi[j,i]>1.0e-10):
            zmax = max(zmax,A.grid.zc[j]*scalx)
        zdike[i] = zmax

      A.z_max[it] = np.max(zdike)

      it += 1

      os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

    return A
  except OSError:
    print('Cannot open: '+A.path_dir+A.input)
    return 0.0

# ---------------------------------
# Main script
# ---------------------------------

path_dir = './'

sims = ['run35_setup5_age2Myr_sigmabc1e-1/',
        'run35_setup5_age2Myr_sigmabc1e-2/',
        'run35_setup5_age2Myr_sigmabc1e-3/']

lbl = ['0.1','0.01','0.001']

A0 = SimStruct()
A1 = SimStruct()
A2 = SimStruct()

A0.input = '../'+sims[0]
A0.path_dir = path_dir

A1.input = '../'+sims[1]
A1.path_dir = path_dir

A2.input = '../'+sims[2]
A2.path_dir = path_dir

A0 = load_porosity_data(A0)
A1 = load_porosity_data(A1)
A2 = load_porosity_data(A2)

fig = plt.figure(1,figsize=(10,8))

ax = plt.subplot(2,2,1)
pl0 = ax.plot(A0.t_kyr,A0.Vphi,label=r'$b$ = '+lbl[0])
pl1 = ax.plot(A1.t_kyr,A1.Vphi,label=r'$b$ = '+lbl[1])
pl2 = ax.plot(A2.t_kyr,A2.Vphi,label=r'$b$ = '+lbl[2])
plt.grid(True)
ax.set_ylabel('Vmagma')
ax.set_xlabel('Time (kyr)')
ax.legend()

ax = plt.subplot(2,2,1)
pl0 = ax.plot(A0.t_kyr,np.log10(A0.phi_max),label=r'$b$ = '+lbl[0])
pl1 = ax.plot(A1.t_kyr,np.log10(A1.phi_max),label=r'$b$ = '+lbl[1])
pl2 = ax.plot(A2.t_kyr,np.log10(A2.phi_max),label=r'$b$ = '+lbl[2])
plt.grid(True)
ax.set_ylabel(r'log$_{10}\phi$ max')
ax.set_xlabel('Time (kyr)')
ax.legend()

ax = plt.subplot(2,2,2)
pl0 = ax.plot(A0.t_kyr,A0.z_max,label=r'$b$ = '+lbl[0])
pl1 = ax.plot(A1.t_kyr,A1.z_max,label=r'$b$ = '+lbl[1])
pl2 = ax.plot(A2.t_kyr,A2.z_max,label=r'$b$ = '+lbl[2])
plt.grid(True)
ax.set_ylabel(r'z max (km)')
ax.set_xlabel('Time (kyr)')
ax.legend()

plt.savefig('plot_magma_constant_supply.png', bbox_inches = 'tight')
plt.close()