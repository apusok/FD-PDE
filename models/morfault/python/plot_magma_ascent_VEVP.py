# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
def load_data(A):
  try: 
    A.input_dir = A.input
    A.dimensional = 1

    # search timesteps in folder
    tdir = os.listdir(A.input)
    if '.DS_Store' in tdir:
      tdir.remove('.DS_Store')
    if 'model_input.opts' in tdir:
      tdir.remove('model_input.opts')
    if 'submit_job.run' in tdir:
      tdir.remove('submit_job.run')

    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if '.out' in s:
        tdir.remove(s)

    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if '_r' in s:
        tdir.remove(s)
    
    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if 'slurm' in s:
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

    # Time arrays
    A.nt_tkyr   = np.zeros(A.nt)
    A.nt_zmagma = np.zeros(A.nt)
    A.nt_xmagma = np.zeros(A.nt)
    A.nt_imagma = np.zeros(A.nt)
    A.nt_jmagma = np.zeros(A.nt)
    A.nt_phitip = np.zeros(A.nt)
    # A.nt_vmagma = np.zeros(A.nt-1)
    A.nt_dotlam = np.zeros(A.nt)
    A.nt_eta = np.zeros(A.nt)
    A.nt_zeta = np.zeros(A.nt)
    A.nt_delta = np.zeros(A.nt)

    A.nt_P = np.zeros(A.nt)
    A.nt_vsx = np.zeros(A.nt)
    A.nt_vsz = np.zeros(A.nt)
    A.nt_vfx = np.zeros(A.nt)
    A.nt_vfz = np.zeros(A.nt)
    A.nt_Plith = np.zeros(A.nt)
    A.nt_DP = np.zeros(A.nt)
    A.nt_DPdl = np.zeros(A.nt)
    A.nt_lam = np.zeros(A.nt)

    A.nt_tauII = np.zeros(A.nt)
    A.nt_tauxx = np.zeros(A.nt)
    A.nt_tauzz = np.zeros(A.nt)
    A.nt_tauxz = np.zeros(A.nt)
    A.nt_epsII = np.zeros(A.nt)
    A.nt_epsxx = np.zeros(A.nt)
    A.nt_epszz = np.zeros(A.nt)
    A.nt_epsxz = np.zeros(A.nt)

    A.nt_epsII_V = np.zeros(A.nt)
    A.nt_epsxx_V = np.zeros(A.nt)
    A.nt_epszz_V = np.zeros(A.nt)
    A.nt_epsxz_V = np.zeros(A.nt)

    A.nt_epsII_E = np.zeros(A.nt)
    A.nt_epsxx_E = np.zeros(A.nt)
    A.nt_epszz_E = np.zeros(A.nt)
    A.nt_epsxz_E = np.zeros(A.nt)

    A.nt_epsII_VP = np.zeros(A.nt)
    A.nt_epsxx_VP = np.zeros(A.nt)
    A.nt_epszz_VP = np.zeros(A.nt)
    A.nt_epsxz_VP = np.zeros(A.nt)

    A.nt_volC = np.zeros(A.nt)
    A.nt_volC_V = np.zeros(A.nt)
    A.nt_volC_E = np.zeros(A.nt)
    A.nt_volC_VP = np.zeros(A.nt)

    A.nt_eta_V = np.zeros(A.nt)
    A.nt_zeta_V = np.zeros(A.nt)
    A.nt_eta_E = np.zeros(A.nt)
    A.nt_zeta_E = np.zeros(A.nt)
    A.nt_eta_VE = np.zeros(A.nt)
    A.nt_zeta_VE = np.zeros(A.nt)
    
    scalx = vizB.get_scaling(A,'x',1,1)
    scalt = vizB.get_scaling(A,'t',1,1)
    scaleps = vizB.get_scaling(A,'eps',1,0)
    scal_v_ms = vizB.get_scaling(A,'v',1,0)
    scal_x_m = vizB.get_scaling(A,'x',1,0)
    scalP = vizB.get_scaling(A,'P',1,1)
    scalv = vizB.get_scaling(A,'v',1,1)
    scaleta = vizB.get_scaling(A,'eta',1,0)

    # Loop over timesteps
    it = 0
    for istep in time_list:
      fdir  = A.input_dir+'Timestep'+str(istep)
      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

      # Correct path for data
      vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
      
      # Get data
      A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
      A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
      A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
      A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
      A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
      A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
      A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
      A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
      A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
      A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
      A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
      A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

      A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)
      A.rheol = vizB.calc_VEVP_strain_rates(A)

      # Extract 
      A.phi = 1.0 - A.phis
      A.nt_tkyr[it]  = A.nd.t*scalt/1e3

      zdike = np.zeros(A.nx)
      jdike = np.zeros(A.nx)
      for i in range(0,A.nx):
        zmax = -50.0
        jj   = 0
        for j in range(0,A.nz):
          if (A.phi[j,i]>1.0e-10):
            zmax = max(zmax,A.grid.zc[j]*scalx)
            jj   = j
        zdike[i] = zmax
        jdike[i] = jj

      A.nt_zmagma[it] = np.max(zdike)
      for i in range(0,A.nx):
        if (A.nt_zmagma[it]==zdike[i]):
          A.nt_xmagma[it] = A.grid.xc[i]*scalx
          A.nt_imagma[it] = i
          A.nt_jmagma[it] = jdike[i]
      
      ii = int(A.nt_imagma[it])
      jj = int(A.nt_jmagma[it])
      
      # the output was done after porosity evolution, such that rheology was only calculated in cell below
      A.nt_phitip[it] = A.phi[jj,ii]
      A.nt_dotlam[it] = A.dotlam[jj-1,ii]*scal_v_ms/scal_x_m
      A.nt_eta[it] = A.matProp.eta[jj-1,ii]*scaleta
      A.nt_zeta[it] = A.matProp.zeta[jj-1,ii]*scaleta
      A.nt_delta[it] = np.sqrt(1e-7*(A.nt_zeta[it]+4/3*A.nt_eta[it])/1.0)

      A.nt_P[it] = A.P[jj-1,int(A.nt_imagma[it])]*scalP
      A.nt_vsx[it] = (A.Vsx[jj  ,ii]+A.Vsx[jj,ii+1])*0.5*scalv
      A.nt_vsz[it] = (A.Vsz[jj+1,ii]+A.Vsz[jj,ii  ])*0.5*scalv
      A.nt_vfx[it] = (A.Vfx[jj  ,ii]+A.Vfx[jj,ii+1])*0.5*scalv
      A.nt_vfz[it] = (A.Vfz[jj+1,ii]+A.Vfz[jj,ii  ])*0.5*scalv
      A.nt_Plith[it] = A.Plith[jj-1,ii]*scalP
      A.nt_DP[it] = A.DP[jj-1,ii]*scalP
      A.nt_DPdl[it] = A.matProp.DPdl[jj-1,ii]*scalP
      A.nt_lam[it] = A.lam[jj-1,ii]

      A.nt_tauII[it] = A.tau.II_center[jj-1,ii]*scalP
      A.nt_tauxx[it] = A.tau.xx_center[jj-1,ii]*scalP
      A.nt_tauzz[it] = A.tau.zz_center[jj-1,ii]*scalP
      A.nt_tauxz[it] = A.tau.xz_center[jj-1,ii]*scalP

      A.nt_epsII[it] = A.eps.II_center[jj-1,ii]*scaleps
      A.nt_epsxx[it] = A.eps.xx_center[jj-1,ii]*scaleps
      A.nt_epszz[it] = A.eps.zz_center[jj-1,ii]*scaleps
      A.nt_epsxz[it] = A.eps.xz_center[jj-1,ii]*scaleps

      A.nt_epsII_V[it] = A.rheol.epsV_II[jj-1,ii]*scaleps
      A.nt_epsxx_V[it] = A.rheol.epsV_xx[jj-1,ii]*scaleps
      A.nt_epszz_V[it] = A.rheol.epsV_zz[jj-1,ii]*scaleps
      A.nt_epsxz_V[it] = A.rheol.epsV_xz[jj-1,ii]*scaleps

      A.nt_epsII_E[it] = A.rheol.epsE_II[jj-1,ii]*scaleps
      A.nt_epsxx_E[it] = A.rheol.epsE_xx[jj-1,ii]*scaleps
      A.nt_epszz_E[it] = A.rheol.epsE_zz[jj-1,ii]*scaleps
      A.nt_epsxz_E[it] = A.rheol.epsE_xz[jj-1,ii]*scaleps

      A.nt_epsII_VP[it] = A.rheol.epsVP_II[jj-1,ii]*scaleps
      A.nt_epsxx_VP[it] = A.rheol.epsVP_xx[jj-1,ii]*scaleps
      A.nt_epszz_VP[it] = A.rheol.epsVP_zz[jj-1,ii]*scaleps
      A.nt_epsxz_VP[it] = A.rheol.epsVP_xz[jj-1,ii]*scaleps

      A.nt_volC[it] = A.divVs[jj-1,ii]*scaleps
      A.nt_volC_V[it] = A.rheol.volV[jj-1,ii]*scaleps
      A.nt_volC_E[it] = A.rheol.volE[jj-1,ii]*scaleps
      A.nt_volC_VP[it] = A.rheol.volVP[jj-1,ii]*scaleps

      A.nt_eta_V[it] = A.matProp.etaV[jj-1,ii]*scaleta
      A.nt_zeta_V[it] = A.matProp.zetaV[jj-1,ii]*scaleta
      A.nt_eta_E[it] = A.matProp.etaE[jj-1,ii]*scaleta
      A.nt_zeta_E[it] = A.matProp.zetaE[jj-1,ii]*scaleta

      A.nt_eta_VE[it] = 1.0/(1.0/A.nt_eta_V[it]+1.0/A.nt_eta_E[it])
      A.nt_zeta_VE[it] = 1.0/(1.0/A.nt_zeta_V[it]+1.0/A.nt_zeta_E[it])

      it += 1

      os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

      # A.nt_vmagma = (A.nt_zmagma[1:]-A.nt_zmagma[:-1])/(A.nt_tkyr[1:]-A.nt_tkyr[:-1])*100 #cm/yr
      
    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot1(A):
  try: 
    fig = plt.figure(1,figsize=(10,10))
    nplots = 5

    ax = plt.subplot(nplots,1,1)
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_phitip),'k-',linewidth=0.5)
    # ax.legend()
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\log_{10}\phi_{tip}$')

    ax = plt.subplot(nplots,1,2)
    pl = ax.plot(A.nt_tkyr,A.nt_zmagma,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'Depth magma tip (km)')

    ax = plt.subplot(nplots,1,3)
    pl = ax.plot(A.nt_tkyr,A.nt_delta/1e3,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\delta_c$ at tip (km)')

    ax = plt.subplot(nplots,1,4)
    pl = ax.plot(A.nt_tkyr,A.nt_vfz,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$V_\ell^z$ at tip (cm/yr)')

    ax = plt.subplot(nplots,1,5)
    pl = ax.plot(A.nt_tkyr,A.nt_dotlam,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\lambda}$ at tip (1/s)')

    plt.savefig(A.output_dir+'magma_tip_time_plot1.png', bbox_inches = 'tight')
    plt.close()
      
    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot2(A):
  try: 
    fig = plt.figure(1,figsize=(10,10))
    nplots = 5

    ax = plt.subplot(nplots,1,1)
    pl = ax.plot(A.nt_tkyr,A.nt_DP,'k-',linewidth=0.5)
    # ax.legend()
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\Delta P$ (MPa)')

    ax = plt.subplot(nplots,1,2)
    pl = ax.plot(A.nt_tkyr,A.nt_DPdl,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\Delta P_{dl}$ (MPa)')

    ax = plt.subplot(nplots,1,3)
    pl = ax.plot(A.nt_tkyr,A.nt_tauII,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\tau_{II}$ (MPa)')

    ax = plt.subplot(nplots,1,4)
    pl = ax.plot(A.nt_tkyr,A.nt_volC_V,'r-',linewidth=0.5,label='V')
    pl = ax.plot(A.nt_tkyr,A.nt_volC_E,'b-',linewidth=0.5,label='E')
    pl = ax.plot(A.nt_tkyr,A.nt_volC_VP,'g-',linewidth=0.5,label='VP')
    pl = ax.plot(A.nt_tkyr,A.nt_volC,'k-',linewidth=0.5,label='total')
    ax.legend(location='upper left')
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\mathcal{C}$ (1/s)')

    ax = plt.subplot(nplots,1,5)
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_V),'r-',linewidth=0.5,label='V')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_E),'b-',linewidth=0.5,label='E')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_VP),'g-',linewidth=0.5,label='VP')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII),'k-',linewidth=0.5, label='total')
    ax.legend(location='upper left')
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\epsilon}_{II}$ (1/s)')

    plt.savefig(A.output_dir+'magma_tip_time_plot2.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot3(A):
  try: 
    fig = plt.figure(1,figsize=(10,12))
    nplots = 5
    linewidth0 = 0.7
    fontsize0 = 14

    ax = plt.subplot(511)
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_phitip),'k-',linewidth=linewidth0)
    # ax.legend()
    # ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\log_{10}\phi_{tip}$', fontsize=fontsize0)

    ax = plt.subplot(512)
    pl = ax.plot(A.nt_tkyr,A.nt_tauII,'k-',linewidth=linewidth0)
    # ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\tau_{II}$ (MPa)', fontsize=fontsize0)

    # ax1 = plt.subplot(512, sharex=ax)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr,A.nt_DP,'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'$\Delta P$ (MPa)', color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    ax = plt.subplot(513)
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_V),'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_E),'b-',linewidth=linewidth0,label='E')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII_VP),'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr,np.log10(A.nt_epsII),'k-',linewidth=linewidth0, label='total')
    ax.legend(loc='upper right')
    # ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\log_{10}\dot{\epsilon}_{II}$ (1/s)', fontsize=fontsize0)

    ax = plt.subplot(514)
    pl = ax.plot(A.nt_tkyr,A.nt_volC_V,'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr,A.nt_volC_E,'b-',linewidth=linewidth0,label='E')
    pl = ax.plot(A.nt_tkyr,A.nt_volC_VP,'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr,A.nt_volC,'k-',linewidth=linewidth0,label='total')
    ax.legend(loc='upper right')
    # ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\mathcal{C}$ (1/s)', fontsize=fontsize0)

    ax = plt.subplot(515)
    pl = ax.plot(A.nt_tkyr,A.nt_dotlam,'k-',linewidth=linewidth0)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\lambda}$ at tip (1/s)', fontsize=fontsize0)

    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr,A.nt_zmagma,'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'Depth (km)',color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    plt.savefig(A.output_dir+'magma_tip_time_plot3.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
# Main script
# ---------------------------------

sim0 = 'run51_VEVP_var_age_eta1e18_Vext0/'
path_in ='/Users/apusok/Documents/morfault/'
path_out='/Users/apusok/Documents/morfault/Figures/'+sim0

read_data = 0

A = SimStruct()
A.input = path_in+sim0
A.output_dir= path_out
vizB.make_dir(A.output_dir)

fname_pickle = A.output_dir+'time_data_magma_ascent_tip.txt'

if (read_data):
  # Read raw data
  A = load_data(A)

  # Save data
  with open(fname_pickle,'wb') as fh:
      pickle.dump(A, fh)

else:
  # Read data
  pickle_off = open(fname_pickle,'rb')
  A = pickle.load(pickle_off)
  pickle_off.close()

# PLOTS
# plot1(A)
# plot2(A)
plot3(A)

# Print
# ind0 = np.where(A.nt_tkyr>50)
# print(A.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A.nt_tkyr>200)
# print(A.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A.nt_tkyr>500)
# print(A.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A.nt_tkyr>900)
# print(A.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A.nt_tkyr>1175)
# print(A.nt_zmagma[ind0[0][0]])
