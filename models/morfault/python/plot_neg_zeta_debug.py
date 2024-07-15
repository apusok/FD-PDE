# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

import warnings
warnings.filterwarnings('ignore')

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
sim = 'run37_flow_dike_dt5e1_etaK1e20/'
A.input = '../'+sim
A.output_path_dir = '../Figures/'+sim
A.path_dir = './'

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

# sort list in increasing tstep
tdir.sort(key=sortTimesteps)
time_list_v0 = np.zeros(nt)
time_list = time_list_v0.astype(int)
for ii in range(0,nt):
  time_list[ii] = int(tdir[ii][8:])

# Create directories - default to check for files
A.input_dir = A.path_dir+A.input
A.output_dir = A.output_path_dir+'fields2/' 

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
istart = 0
iend   = A.nx
jstart = 0
jend   = A.nz

print('  >> '+A.output_dir)

# Loop over timesteps
# for istep in time_list:
for istep1 in range(2336,2337,1):
  istep = time_list[istep1]

  fdir  = A.input_dir+'Timestep'+str(istep)
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
  A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
  # A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
  A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
  A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
  A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

  # Center velocities and mass divergence
  A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

  # print values
  i = int(A.nx/2)
  j = 54
  dim = 1
  scal_eta = vizB.get_scaling(A,'eta',dim,0)
  scal_P = vizB.get_scaling(A,'P',dim,1)
  scal_v = vizB.get_scaling(A,'v',dim,1)
  scal_x = vizB.get_scaling(A,'x',dim,1)
  scal_v_ms = vizB.get_scaling(A,'v',dim,0)
  scal_x_m = vizB.get_scaling(A,'x',dim,0)
  scal_t = vizB.get_scaling(A,'t',dim,1)
  scal_Kphi = vizB.get_scaling(A,'Kphi',dim,0)
  
  zetaVE = 1.0/(1.0/A.matProp.zetaV[j,i]+1.0/A.matProp.zetaE[j,i])
  phis = A.phis[j,i]
  curlyCp = A.divVs[j,i] - A.DPold[j,i]*phis/A.nd.dt/A.matProp.Z[j,i]

  print('\nDimensional:')
  print('zeta = ',A.matProp.zeta[j,i]*scal_eta,' Pa s')
  print('zeta_VE = ',zetaVE*scal_eta,' Pa s')
  print('phis = ',phis)
  print('DP = ',A.DP[j,i]*scal_P,' MPa')
  print('DP_old = ',A.DPold[j,i]*scal_P,' MPa')
  print('Z = ',A.matProp.Z[j,i]*scal_P,' MPa')
  print('divVs = ',A.divVs[j,i]*scal_v_ms/scal_x_m)
  print('curlyCp =',curlyCp*scal_v_ms/scal_x_m)
  print('dt = ',A.nd.dt*scal_t)

  print('\nNondimensional:')
  print('zeta = ',A.matProp.zeta[j,i])
  print('zeta_VE = ',zetaVE)
  print('phis = ',phis)
  print('phi = ',1.0-phis)
  print('DP = ',A.DP[j,i])
  print('DP_old = ',A.DPold[j,i])
  print('Z = ',A.matProp.Z[j,i])
  print('divVs = ',A.divVs[j,i])
  print('dt = ',A.nd.dt)
  print('curlyCp =',curlyCp)
  print('x =',A.grid.xc[i]*scal_x)
  print('z =',A.grid.zc[j]*scal_x)
  print('Kphi =',A.matProp.Kphi[j,i]*scal_Kphi)

  nplots = 11
  iplot = 0
  fig = plt.figure(1,figsize=(28,8))
  
  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.P[:,i]*scal_P,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$P$ [MPa]')
  ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.DP[:,i]*scal_P,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\Delta P$ [MPa]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.tau.II_center[:,i]*scal_P,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\tau_{II}$ [MPa]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.eps.II_center[:,i]*scal_v_ms/scal_x_m,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\dot{\epsilon}_{II}$ [1/s]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.divVs[:,i]*scal_v_ms/scal_x_m,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\nabla\cdot v_s$ [1/s]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot((A.divVs[:,i] - A.DPold[:,i]*A.phis[:,i]/A.nd.dt/A.matProp.Z[:,i])*scal_v_ms/scal_x_m,A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\mathcal{C}p$ [1/s]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(np.log10(A.matProp.eta[:,i]*scal_eta),A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'log10($\eta_{eff}$) [Pa s]')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(np.log10(A.matProp.zeta[:,i]*scal_eta),A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'log10($\zeta_{eff}$) [Pa s]')
  # ax.set_ylabel('z [km]')

  X = 1.0 - A.phis
  X[X<1e-20] = 1e-20
  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(np.log10(X[:,i]),A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'log10($\phi$)')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.lam[:,i],A.grid.zc*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$\lambda$')
  # ax.set_ylabel('z [km]')

  iplot +=1
  ax = plt.subplot(1,nplots,iplot)
  ax.plot(A.Vfz[:,i]*scal_v,A.grid.zv*scal_x,'k-')
  ax.grid(True)
  ax.set_xlabel(r'$v_\ell^z$ [cm/yr]')
  # ax.set_ylabel('z [km]')

  # ax = plt.subplot(1,nplots,2)
  # ax.plot(np.log10(1.0/(1.0/A.matProp.zetaV[:,i]+1.0/A.matProp.zetaE[:,i])*scal_eta),A.grid.zc*scal_x,'k-')
  # ax.grid(True)
  # ax.set_xlabel('log10(zetaVE) [Pa s]]')
  # # ax.set_ylabel('z [km]')

  # ax = plt.subplot(1,nplots,7)
  # ax.plot(A.matProp.Z[:,i]*scal_P,A.grid.zc*scal_x,'k-')
  # ax.grid(True)
  # ax.set_xlabel('Zphi [MPa]')
  # # ax.set_ylabel('z [km]')

  # ax = plt.subplot(1,nplots,9)
  # ax.plot(A.matProp.Kphi[:,i]*scal_Kphi,A.grid.zc*scal_x,'k-')
  # ax.grid(True)
  # ax.set_xlabel('Kphi [m2]')

  plt.savefig(A.output_path_dir+'profile_1D_z_ts'+str(istep)+'.png', bbox_inches = 'tight')
  plt.close()


  # Calculate individual deformation strain rates
  # A.rheol = vizB.calc_dom_rheology_mechanism(A)

  # vizB.plot_mark_eps_phi(A,istart,iend,jstart,jend,A.output_dir,'out_mark_eps_phi_ts'+str(istep),istep,A.dimensional)

  # # Plots: A.output_path_dir+'fields2/'
  # # vizB.plot_mark_eta_eps_tau(A,istart,iend,jstart,jend,A.output_path_dir+'fields0/','out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_mark_eta_eps_tau2(A,istart,iend,jstart,jend,A.output_path_dir+'fields00/','out_mark_eta_eps_tau2_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_mark_eta_eps_tau_T_phi(A,istart,iend,jstart,jend,A.output_path_dir+'fields1/','out_mark_eta_eps_tau_T_phi_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_def_mechanisms(A,istart,iend,jstart,jend,A.output_path_dir+'def_mech/','out_def_mechanisms_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_mark_eps_phi(A,istart,iend,jstart,jend,A.output_dir,'out_mark_eps_phi_ts'+str(istep),istep,A.dimensional)

  # # vizB.plot_T(A,istart,iend,jstart,jend,A.output_path_dir+'T/','out_xT_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_MPhase(A,istart,iend,jstart,jend,A.output_path_dir+'MPhase/','out_xMPhase_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_P(A,istart,iend,jstart,jend,A.output_path_dir+'P/','out_xP_ts'+str(istep),istep,A.dimensional,3)
  # vizB.plot_PV(A,istart,iend,jstart,jend,A.output_path_dir+'PV/','out_xPV_ts'+str(istep),istep,A.dimensional,0)
  # vizB.plot_PV(A,istart,iend,jstart,jend,A.output_path_dir+'resPV/','out_resPV_ts'+str(istep),istep,A.dimensional,1)
  # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'eps/','out_xeps_ts'+str(istep),istep,A.dimensional,0)
  # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'tau/','out_xtau_ts'+str(istep),istep,A.dimensional,1)
  # # vizB.plot_Tensor(A,istart,iend,jstart,jend,A.output_path_dir+'tauold/','out_xtauold_ts'+str(istep),istep,A.dimensional,2)
  # # vizB.plot_PVcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'PVcoeff/','out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_PVcoeff_Stokes(A,istart,iend,jstart,jend,A.output_path_dir+'PVcoeff/','out_xPVcoeff_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_matProp(A,istart,iend,jstart,jend,A.output_path_dir+'matProp/','out_matProp_ts'+str(istep),istep,A.dimensional)
  # vizB.plot_Vel(A,istart,iend,jstart,jend,A.output_path_dir+'Vel/','out_xVel_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_Tcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'Tcoeff/','out_xTcoeff_ts'+str(istep),istep,A.dimensional,0)
  # # vizB.plot_Tcoeff(A,istart,iend,jstart,jend,A.output_path_dir+'phicoeff/','out_xphicoeff_ts'+str(istep),istep,A.dimensional,1)
  # vizB.plot_plastic(A,istart,iend,jstart,jend,A.output_path_dir+'plastic/','out_xplastic_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_individual_eps(A,istart,iend,jstart,jend,A.output_path_dir+'eps_ind/','out_individual_eps_ts'+str(istep),istep,A.dimensional)
  # # vizB.plot_phi(A,istart,iend,jstart,jend,A.output_path_dir+'phi/','out_xphi_ts'+str(istep),istep,A.dimensional)

  os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')