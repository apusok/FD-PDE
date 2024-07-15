# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm

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
def get9pointind(i,j):
  vali = np.zeros(9)
  valj = np.zeros(9)
  vali[0] = i
  vali[1] = i-1
  vali[2] = i
  vali[3] = i+1
  vali[4] = i-1
  vali[5] = i+1
  vali[6] = i-1
  vali[7] = i
  vali[8] = i+1
  
  valj[0] = j
  valj[1] = j-1
  valj[2] = j-1
  valj[3] = j-1
  valj[4] = j
  valj[5] = j
  valj[6] = j+1
  valj[7] = j+1
  valj[8] = j+1

  return vali, valj

# ---------------------------------
def get9pointcoord(x,z,i,j,scal):
  valx = np.zeros(9)
  valz = np.zeros(9)
  valx[0] = x[i]*scal
  valx[1] = x[i-1]*scal
  valx[2] = x[i]*scal
  valx[3] = x[i+1]*scal
  valx[4] = x[i-1]*scal
  valx[5] = x[i+1]*scal
  valx[6] = x[i-1]*scal
  valx[7] = x[i]*scal
  valx[8] = x[i+1]*scal
  
  valz[0] = z[j]*scal
  valz[1] = z[j-1]*scal
  valz[2] = z[j-1]*scal
  valz[3] = z[j-1]*scal
  valz[4] = z[j]*scal
  valz[5] = z[j]*scal
  valz[6] = z[j+1]*scal
  valz[7] = z[j+1]*scal
  valz[8] = z[j+1]*scal

  return valx, valz

# ---------------------------------
def get9pointvalues(A,i,j,scal):
  val = np.zeros(9)
  if (i<0):
    val[0] = A[j  ,0]*scal
    val[1] = A[j-1,0]*scal
    val[2] = A[j-1,0]*scal
    val[3] = A[j-1,0]*scal
    val[4] = A[j  ,0]*scal
    val[5] = A[j  ,0]*scal
    val[6] = A[j+1,0]*scal
    val[7] = A[j+1,0]*scal
    val[8] = A[j+1,0]*scal
  else:
    val[0] = A[j  ,i  ]*scal
    val[1] = A[j-1,i-1]*scal
    val[2] = A[j-1,i  ]*scal
    val[3] = A[j-1,i+1]*scal
    val[4] = A[j  ,i-1]*scal
    val[5] = A[j  ,i+1]*scal
    val[6] = A[j+1,i-1]*scal
    val[7] = A[j+1,i  ]*scal
    val[8] = A[j+1,i+1]*scal

  return val

# ---------------------------------
def calc_width_magma(A,i,j):
  w = 0.0
  
  ind0 = 0
  for ii in range(i,A.nx-1):
    if (A.phi[j,ii]>0.0):
      ind0 += 1
      if (A.phi[j,ii+1]==0.0):
        break
  
  ind1 = 0
  for ii in range(i,0,-1):
    if (A.phi[j,ii]>0.0):
      ind1 += 1
      if (A.phi[j,ii-1]==0.0):
        break
  
  w = A.grid.xc[i+ind0]-A.grid.xc[i-ind1]
  return w

# ---------------------------------
def get9pointwidthmagma(A,i,j,scal):
  val = np.zeros(9)
  val[0] = calc_width_magma(A,i,j)*scal
  val[1] = calc_width_magma(A,i,j-1)*scal
  val[2] = val[1]
  val[3] = val[1]
  val[4] = val[0]
  val[5] = val[0]
  val[6] = calc_width_magma(A,i,j+1)*scal
  val[7] = val[6]
  val[8] = val[6]

  return val

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

    # Time arrays - for 9 points surrounding the tip of magma (ii,jj)
    npoints = 9
    A.nt_tkyr   = np.zeros(A.nt)
    A.nt_zmagma = np.zeros([npoints,A.nt])
    A.nt_xmagma = np.zeros([npoints,A.nt])
    A.nt_wmagma = np.zeros([npoints,A.nt])
    A.nt_imagma = np.zeros([npoints,A.nt])
    A.nt_jmagma = np.zeros([npoints,A.nt])
    A.nt_phi    = np.zeros([npoints,A.nt])

    A.nt_dotlam = np.zeros([npoints,A.nt])
    A.nt_eta = np.zeros([npoints,A.nt])
    A.nt_zeta = np.zeros([npoints,A.nt])
    A.nt_delta = np.zeros([npoints,A.nt])
    
    A.nt_T = np.zeros([npoints,A.nt])
    A.nt_Tiso = np.zeros([npoints,A.nt])
    A.nt_P = np.zeros([npoints,A.nt])
    A.nt_vsx = np.zeros([npoints,A.nt])
    A.nt_vsz = np.zeros([npoints,A.nt])
    A.nt_vfx = np.zeros([npoints,A.nt])
    A.nt_vfz = np.zeros([npoints,A.nt])
    A.nt_Plith = np.zeros([npoints,A.nt])
    A.nt_DP = np.zeros([npoints,A.nt])
    A.nt_DPdl = np.zeros([npoints,A.nt])
    A.nt_lam = np.zeros([npoints,A.nt])

    A.nt_tauII = np.zeros([npoints,A.nt])
    A.nt_tauxx = np.zeros([npoints,A.nt])
    A.nt_tauzz = np.zeros([npoints,A.nt])
    A.nt_tauxz = np.zeros([npoints,A.nt])
    A.nt_epsII = np.zeros([npoints,A.nt])
    A.nt_epsxx = np.zeros([npoints,A.nt])
    A.nt_epszz = np.zeros([npoints,A.nt])
    A.nt_epsxz = np.zeros([npoints,A.nt])

    A.nt_epsII_V = np.zeros([npoints,A.nt])
    A.nt_epsxx_V = np.zeros([npoints,A.nt])
    A.nt_epszz_V = np.zeros([npoints,A.nt])
    A.nt_epsxz_V = np.zeros([npoints,A.nt])

    A.nt_epsII_E = np.zeros([npoints,A.nt])
    A.nt_epsxx_E = np.zeros([npoints,A.nt])
    A.nt_epszz_E = np.zeros([npoints,A.nt])
    A.nt_epsxz_E = np.zeros([npoints,A.nt])

    A.nt_epsII_VP = np.zeros([npoints,A.nt])
    A.nt_epsxx_VP = np.zeros([npoints,A.nt])
    A.nt_epszz_VP = np.zeros([npoints,A.nt])
    A.nt_epsxz_VP = np.zeros([npoints,A.nt])

    A.nt_volC = np.zeros([npoints,A.nt])
    A.nt_volC_V = np.zeros([npoints,A.nt])
    A.nt_volC_E = np.zeros([npoints,A.nt])
    A.nt_volC_VP = np.zeros([npoints,A.nt])

    A.nt_eta_V = np.zeros([npoints,A.nt])
    A.nt_zeta_V = np.zeros([npoints,A.nt])
    A.nt_eta_E = np.zeros([npoints,A.nt])
    A.nt_zeta_E = np.zeros([npoints,A.nt])
    A.nt_eta_VE = np.zeros([npoints,A.nt])
    A.nt_zeta_VE = np.zeros([npoints,A.nt])
    
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
      vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
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
      A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
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
      X = 1.0 - A.phis
      X[X<0.0] = 0.0
      A.phi = X
      A.nt_tkyr[it]  = A.nd.t*scalt/1e3

      zdike = np.zeros(A.nx)
      jdike = np.zeros(A.nx)
      for i in range(0,A.nx):
        zmax = A.grid.zc[0]*scalx
        jj   = 0
        for j in range(0,A.nz):
          if (A.phi[j,i]>1.0e-10):
            zmax = max(zmax,A.grid.zc[j]*scalx)
            jj   = j
        zdike[i] = zmax
        jdike[i] = jj
      
      # save tip in location 0
      itip = 0
      A.nt_zmagma[itip,it] = np.max(zdike)
      for i in range(0,A.nx):
        if (A.nt_zmagma[itip,it]==zdike[i]):
          A.nt_xmagma[itip,it] = A.grid.xc[i]*scalx
          A.nt_imagma[itip,it] = i
          A.nt_jmagma[itip,it] = jdike[i]
      
      ii = int(A.nt_imagma[itip,it])
      jj = int(A.nt_jmagma[itip,it])
      
#      if (istep==0):
#        print(A.nt_zmagma[itip,it])
#        for j in range(0,A.nz):
#          print(A.grid.zc[j]*scalx,A.phi[j,ii])
      
      # save indices
      A.nt_imagma[:,it], A.nt_jmagma[:,it] = get9pointind(ii,jj)
      
      # save coord
      A.nt_xmagma[:,it], A.nt_zmagma[:,it] = get9pointcoord(A.grid.xc,A.grid.zc,ii,jj,scalx)
      A.nt_wmagma[:,it] = get9pointwidthmagma(A,ii,jj,scalx)

      # save values - A.nt_eta[:,it] = get9pointvalues(A,ii,jj,scal)
      A.nt_phi[:,it] = get9pointvalues(A.phi,ii,jj,1)
      A.nt_dotlam[:,it] = get9pointvalues(A.dotlam,ii,jj,scal_v_ms/scal_x_m)
      A.nt_eta[:,it] = get9pointvalues(A.matProp.eta,ii,jj,scaleta)
      A.nt_zeta[:,it] = get9pointvalues(A.matProp.zeta,ii,jj,scaleta)
      A.nt_delta[:,it] = np.sqrt(1e-7*(A.nt_zeta[:,it]+4/3*A.nt_eta[:,it])/1.0)

      X = vizB.scale_TC(A,'T','T',1,1)
      A.nt_T[:,it] = get9pointvalues(X,ii,jj,1)
      A.nt_Tiso[:,it] = get9pointvalues(X,-1,jj,1)
      
      A.nt_P[:,it] = get9pointvalues(A.P,ii,jj,scalP)
      A.nt_vsx[:,it] = get9pointvalues(A.Vscx,ii,jj,scalv)
      A.nt_vsz[:,it] = get9pointvalues(A.Vscz,ii,jj,scalv)

      A.Vfcx = (A.Vfx[:,:-1]+A.Vfx[:,1:])*0.5
      A.Vfcz = (A.Vfz[:-1,:]+A.Vfz[1:,:])*0.5
      
      A.nt_vfx[:,it] = get9pointvalues(A.Vfcx,ii,jj,scalv)
      A.nt_vfz[:,it] = get9pointvalues(A.Vfcz,ii,jj,scalv)
      
      A.nt_Plith[:,it] = get9pointvalues(A.Plith,ii,jj,scalP)
      A.nt_DP[:,it] = get9pointvalues(A.DP,ii,jj,scalP)
      A.nt_DPdl[:,it] = get9pointvalues(A.matProp.DPdl,ii,jj,scalP)
      A.nt_lam[:,it] = get9pointvalues(A.lam,ii,jj,1)

      A.nt_tauII[:,it] = get9pointvalues(A.tau.II_center,ii,jj,scalP)
      A.nt_tauxx[:,it] = get9pointvalues(A.tau.xx_center,ii,jj,scalP)
      A.nt_tauzz[:,it] = get9pointvalues(A.tau.zz_center,ii,jj,scalP)
      A.nt_tauxz[:,it] = get9pointvalues(A.tau.xz_center,ii,jj,scalP)

      A.nt_epsII[:,it] = get9pointvalues(A.eps.II_center,ii,jj,scaleps)
      A.nt_epsxx[:,it] = get9pointvalues(A.eps.xx_center,ii,jj,scaleps)
      A.nt_epszz[:,it] = get9pointvalues(A.eps.zz_center,ii,jj,scaleps)
      A.nt_epsxz[:,it] = get9pointvalues(A.eps.xz_center,ii,jj,scaleps)

      A.nt_epsII_V[:,it] = get9pointvalues(A.rheol.epsV_II,ii,jj,scaleps)
      A.nt_epsxx_V[:,it] = get9pointvalues(A.rheol.epsV_xx,ii,jj,scaleps)
      A.nt_epszz_V[:,it] = get9pointvalues(A.rheol.epsV_zz,ii,jj,scaleps)
      A.nt_epsxz_V[:,it] = get9pointvalues(A.rheol.epsV_xz,ii,jj,scaleps)

      A.nt_epsII_E[:,it] = get9pointvalues(A.rheol.epsE_II,ii,jj,scaleps)
      A.nt_epsxx_E[:,it] = get9pointvalues(A.rheol.epsE_xx,ii,jj,scaleps)
      A.nt_epszz_E[:,it] = get9pointvalues(A.rheol.epsE_zz,ii,jj,scaleps)
      A.nt_epsxz_E[:,it] = get9pointvalues(A.rheol.epsE_xz,ii,jj,scaleps)

      A.nt_epsII_VP[:,it] = get9pointvalues(A.rheol.epsVP_II,ii,jj,scaleps)
      A.nt_epsxx_VP[:,it] = get9pointvalues(A.rheol.epsVP_xx,ii,jj,scaleps)
      A.nt_epszz_VP[:,it] = get9pointvalues(A.rheol.epsVP_zz,ii,jj,scaleps)
      A.nt_epsxz_VP[:,it] = get9pointvalues(A.rheol.epsVP_xz,ii,jj,scaleps)

      A.nt_volC[:,it] = get9pointvalues(A.divVs,ii,jj,scaleps)
      A.nt_volC_V[:,it] = get9pointvalues(A.rheol.volV,ii,jj,scaleps)
      A.nt_volC_E[:,it] = get9pointvalues(A.rheol.volE,ii,jj,scaleps)
      A.nt_volC_VP[:,it] = get9pointvalues(A.rheol.volVP,ii,jj,scaleps)

      A.nt_eta_V[:,it] = get9pointvalues(A.matProp.etaV,ii,jj,scaleta)
      A.nt_zeta_V[:,it] = get9pointvalues(A.matProp.zetaV,ii,jj,scaleta)
      A.nt_eta_E[:,it] = get9pointvalues(A.matProp.etaE,ii,jj,scaleta)
      A.nt_zeta_E[:,it] = get9pointvalues(A.matProp.zetaE,ii,jj,scaleta)

      A.nt_eta_VE[:,it] = 1.0/(1.0/A.nt_eta_V[:,it]+1.0/A.nt_eta_E[:,it])
      A.nt_zeta_VE[:,it] = 1.0/(1.0/A.nt_zeta_V[:,it]+1.0/A.nt_zeta_E[:,it])

      it += 1

      os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')
    
#    a0 = make_interp_spline(A.nt_tkyr,A.nt_epsII_E[0,:])
#    A.nt_epsII_E_avg0 = a0(A.nt_tkyr)
#    a1 = make_interp_spline(A.nt_tkyr,A.nt_epsII_V[0,:])
#    A.nt_epsII_V_avg0 = a1(A.nt_tkyr)
#    a2 = make_interp_spline(A.nt_tkyr,A.nt_DP[2,:])
#    A.nt_DP_avg2 = a2(A.nt_tkyr)

#    deg = 2
#    a0 = np.polyfit(A.nt_tkyr,A.nt_epsII_E[0,:],deg)
#    A.nt_epsII_E_avg0 = np.poly1d(a0)(A.nt_tkyr)
#    a1 = np.polyfit(A.nt_tkyr,A.nt_epsII_V[0,:],deg)
#    A.nt_epsII_V_avg0 = np.poly1d(a1)(A.nt_tkyr)
#    a2 = np.polyfit(A.nt_tkyr,A.nt_DP[2,:],deg)
#    A.nt_DP_avg2 = np.poly1d(a2)(A.nt_tkyr)

    smooth = 0.05
    A.nt_epsII_E_avg0 = sm.nonparametric.lowess(A.nt_epsII_E[0,:],A.nt_tkyr,frac=smooth)
    A.nt_epsII_V_avg0 = sm.nonparametric.lowess(A.nt_epsII_V[0,:],A.nt_tkyr,frac=smooth)
    A.nt_DP_avg2 = sm.nonparametric.lowess(A.nt_DP[2,:],A.nt_tkyr,frac=smooth)
    
    ind0 = np.where(A.nt_tkyr>=A.tend)
    
#    # BDTZ delineator: eps^E>eps^V
#    indbd0 = np.where(A.nt_epsII_E[0,1:]>A.nt_epsII_V[0,1:]) # first index
#
#    # brittle delineator: DP(2)>0 and min(phi(1))
#    indphi = np.where((A.nt_phi[1,:ind0[0][0]]>1e-8) & (A.nt_DP[2,:ind0[0][0]]>0.0))
#    min_phitip = np.min(A.nt_phi[1,indphi[0][:]])
#    t_phitip = A.nt_tkyr[indphi[0][:]]
#    indd1 = np.where(A.nt_phi[1,indphi[0][:]]==min_phitip) # first index
#    indt1 = np.where(A.nt_tkyr==t_phitip[indd1[0][0]])

    # BDTZ delineator: eps^E>eps^V
    indbd0 = np.where(A.nt_epsII_E_avg0[1:,1]>A.nt_epsII_V_avg0[1:,1]) # first index

    # brittle delineator: DP(2)>0 and min(phi(1))
    indphi = np.where((A.nt_phi[1,2:ind0[0][0]]>1e-8) & (A.nt_DP_avg2[2:ind0[0][0],1]>0.0))
    min_phitip = np.min(A.nt_phi[1,indphi[0][:]])
    t_phitip = A.nt_tkyr[indphi[0][:]]
    indd1 = np.where(A.nt_phi[1,indphi[0][:]]==min_phitip) # first index
    indt1 = np.where(A.nt_tkyr==t_phitip[indd1[0][0]])
    
#    print(indphi[0])
#    print(indt1[0])
    
    A.ind_BTDZ = indbd0[0][0]
    A.ind_brittle = indt1[0][0] #indd1[0][0]

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot1(A,itip,tend):
  try: 
    fig = plt.figure(1,figsize=(10,10))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)
    
    ax = plt.subplot(511)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[itip,:ind0[0][0]]),'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\log_{10}\phi$')
    ax.grid(True)

    ax = plt.subplot(512)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\lambda}$ (1/s)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_zmagma[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'Depth (km)',color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    ax = plt.subplot(513)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_delta[itip,:ind0[0][0]]/1e3,'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\delta_c$ at tip (km)')
    ax.grid(True)

    ax = plt.subplot(514)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[itip,:ind0[0][0]],'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$V_\ell^z$ at tip (cm/yr)')
    ax.grid(True)

    ax = plt.subplot(515)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauxx[itip,:ind0[0][0]],'r-',linewidth=linewidth0,label='xx')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauzz[itip,:ind0[0][0]],'b-',linewidth=linewidth0,label='zz')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauxz[itip,:ind0[0][0]],'g-',linewidth=linewidth0,label='xz')
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$\tau_{ij}$ (MPa)', fontsize=fontsize0)

    plt.savefig(A.output_dir+'magma_tip_time_plot1_loc'+str(itip)+'.png', bbox_inches = 'tight')
    plt.close()
      
    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot2(A,itip,tend):
  try: 
    fig = plt.figure(1,figsize=(10,10))
    linewidth0 = 0.7
    fontsize0 = 14
    nplots= 5
    ind0 = np.where(A.nt_tkyr>=tend)

    ax = plt.subplot(nplots,1,1)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[itip,:ind0[0][0]],'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\Delta P$ (MPa)')
    ax.grid(True)

    ax = plt.subplot(nplots,1,2)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DPdl[itip,:ind0[0][0]],'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\Delta P_{dl}$ (MPa)')
    ax.grid(True)

    ax = plt.subplot(nplots,1,3)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[itip,:ind0[0][0]],'k-',linewidth=0.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\tau_{II}$ (MPa)')
    ax.grid(True)

    ax = plt.subplot(nplots,1,4)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_V[itip,:ind0[0][0]],'r-',linewidth=0.5,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_E[itip,:ind0[0][0]],'b-',linewidth=0.5,label='E')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_VP[itip,:ind0[0][0]],'g-',linewidth=0.5,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC[itip,:ind0[0][0]],'k-',linewidth=0.5,label='total')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\mathcal{C}$ (1/s)')
    ax.grid(True)

    ax = plt.subplot(nplots,1,5)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_V[itip,:ind0[0][0]]),'r-',linewidth=0.5,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_E[itip,:ind0[0][0]]),'b-',linewidth=0.5,label='E')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[itip,:ind0[0][0]]),'g-',linewidth=0.5,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII[itip,:ind0[0][0]]),'k-',linewidth=0.5, label='total')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\epsilon}_{II}$ (1/s)')
    ax.grid(True)

    plt.savefig(A.output_dir+'magma_tip_time_plot2_loc'+str(itip)+'.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot3(A,itip,tend):
  try: 
    fig = plt.figure(1,figsize=(10,12))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)

    ax = plt.subplot(511)
    ax.grid(True)
    if (itip>=5):
      pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_phi[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
      ax.set_ylabel(r'$\phi$', fontsize=fontsize0)
    else:
      pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[itip,:ind0[0][0]]),'k-',linewidth=linewidth0)
      ax.set_ylabel(r'$\log_{10}\phi$', fontsize=fontsize0)

    ax = plt.subplot(512)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_ylabel(r'$\tau_{II}$ (MPa)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'$\Delta P$ (MPa)', color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    ax = plt.subplot(513)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_V[itip,:ind0[0][0]]),'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_E[itip,:ind0[0][0]]),'b-',linewidth=linewidth0,label='E')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[itip,:ind0[0][0]]),'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII[itip,:ind0[0][0]]),'k-',linewidth=linewidth0, label='total')
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$\log_{10}\dot{\epsilon}_{II}$ (1/s)', fontsize=fontsize0)

    ax = plt.subplot(514)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_V[itip,:ind0[0][0]],'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_E[itip,:ind0[0][0]],'b-',linewidth=linewidth0,label='E')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_VP[itip,:ind0[0][0]],'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC[itip,:ind0[0][0]],'k-',linewidth=linewidth0,label='total')
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$\mathcal{C}$ (1/s)', fontsize=fontsize0)

    ax = plt.subplot(515)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$\dot{\lambda}$ (1/s)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_zmagma[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'Depth (km)',color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    plt.savefig(A.output_dir+'magma_tip_time_plot3_loc'+str(itip)+'.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot4(A,itip,tend):
  try:
    fig = plt.figure(1,figsize=(10,12))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)

    ax = plt.subplot(511)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_wmagma[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_ylabel(r'Magma width (km)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_xmagma[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'x-dir magma (km)', color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    ax = plt.subplot(512)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_T[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_ylabel(r'$T$ ($^o$C)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_Tiso[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'$T_{iso}$ ($^o$C)', color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax = plt.subplot(513)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_ylabel(r'$V_\ell^z$ (cm/yr)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfx[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'$V_\ell^x$ (cm/yr)', color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')
    
    if ((itip==0) | (itip==4) | (itip==5)):
      itip0 = 4
      itip1 = 0
      itip2 = 5
    if ((itip==1) | (itip==2) | (itip==3)):
      itip0 = 1
      itip1 = 2
      itip2 = 3
    if ((itip==6) | (itip==7) | (itip==8)):
      itip0 = 6
      itip1 = 7
      itip2 = 8
    
    dPdx0 = (A.nt_P[itip1,:]-A.nt_P[itip0,:])/A.dx
    dPdx1 = (A.nt_P[itip2,:]-A.nt_P[itip1,:])/A.dx
    
    if ((itip==0) | (itip==2) | (itip==7)):
      itip0 = 2
      itip1 = 0
      itip2 = 7
    if ((itip==1) | (itip==4) | (itip==6)):
      itip0 = 1
      itip1 = 4
      itip2 = 6
    if ((itip==3) | (itip==5) | (itip==8)):
      itip0 = 3
      itip1 = 5
      itip2 = 8
    
    dPdz0 = (A.nt_P[itip1,:]-A.nt_P[itip0,:])/A.dz
    dPdz1 = (A.nt_P[itip2,:]-A.nt_P[itip1,:])/A.dz
    
    ax = plt.subplot(514) #dP/dx, dP/dz
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],dPdx0[:ind0[0][0]],'k-',linewidth=linewidth0, label='dP/dx (left)')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],dPdx1[:ind0[0][0]],'k--',linewidth=linewidth0, label='dP/dx (right)')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],dPdz0[:ind0[0][0]],'b-',linewidth=linewidth0, label='dP/dz (down)')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],dPdz1[:ind0[0][0]],'b--',linewidth=linewidth0, label='dP/dz (up)')
    ax.set_ylabel(r'$\nabla P$ (MPa/km)', fontsize=fontsize0)
    ax.legend(loc='lower left')
    
    ax = plt.subplot(515)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_P[itip,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel(r'$P$ (MPa)', fontsize=fontsize0)
    ax1 = ax.twinx()
    pl = ax1.plot(A.nt_tkyr[:ind0[0][0]],A.nt_Plith[itip,:ind0[0][0]],'b-',linewidth=linewidth0)
    ax1.set_ylabel(r'$P_{lith}$ (MPa)',color='b', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='b')

    plt.savefig(A.output_dir+'magma_tip_time_plot4_loc'+str(itip)+'.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot5(A,tend):
  try:
    fig = plt.figure(1,figsize=(10,12))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)
    
#    # delineators - check multiple criteria
#    indbd0 = np.where(A.nt_epsII_E[0,:]>A.nt_epsII_V[0,:]) # first index
#    indbd1 = np.where(A.nt_epsII_E[1,:]>A.nt_epsII_V[1,:])
#    indbd2 = np.where(A.nt_epsII_E[2,:]>A.nt_epsII_V[2,:])
#
#    indphi = np.where(A.nt_phi[1,:ind0[0][0]]>1e-8)
#    min_phitip = np.min(A.nt_phi[1,indphi[0][:]])
#
##    print(A.nt_epsII_VP[1,:ind0[0][0]])
#    indd0 = np.where(A.nt_epsII_VP[1,:ind0[0][0]-10]>1e-20) # last index here
#    indd1 = np.where(A.nt_phi[1,:]==min_phitip) # first index

    xt0 = A.nt_tkyr[A.ind_BTDZ]
    xt1 = A.nt_tkyr[A.ind_brittle]
        
    ax = plt.subplot(511)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[0,:ind0[0][0]]),'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[1,:ind0[0][0]]),'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[2,:ind0[0][0]]),'b-',linewidth=linewidth0, label='loc 2')
    
#    pl = ax.plot(A.nt_tkyr[indbd0[0][0]],np.log10(A.nt_phi[0,indbd0[0][0]]),'ko')
#    pl = ax.plot(A.nt_tkyr[indbd1[0][0]],np.log10(A.nt_phi[1,indbd1[0][0]]),'ro')
#    pl = ax.plot(A.nt_tkyr[indbd2[0][0]],np.log10(A.nt_phi[2,indbd2[0][0]]),'bo')
#
#    pl = ax.plot(A.nt_tkyr[indd0[0][-1]],np.log10(A.nt_phi[0,indd0[0][-1]]),'rd')
#    pl = ax.plot(A.nt_tkyr[indd1[0][0]],np.log10(A.nt_phi[1,indd1[0][0]]),'ro')
    
    # delineators
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\log_{10}\phi$', fontsize=fontsize0)

    ax = plt.subplot(512)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[1,:ind0[0][0]],'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\tau_{II}$ (MPa)', fontsize=fontsize0)
    # delineators
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(513)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[1,:ind0[0][0]],'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\Delta P$ (MPa)', fontsize=fontsize0)
    # delineators
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(514)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[1,:ind0[0][0]],'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$V_\ell^z$ (cm/yr)', fontsize=fontsize0)
    # delineators
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(515)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[0,:ind0[0][0]]),'k-',linewidth=linewidth0,label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[1,:ind0[0][0]]),'r-',linewidth=linewidth0,label='loc 1')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[2,:ind0[0][0]]),'b-',linewidth=linewidth0,label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\log_{10}\dot{\epsilon}_{II}^{VP}$ (1/s)', fontsize=fontsize0)
    # delineators
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    
#    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
#    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[1,:ind0[0][0]],'r-',linewidth=linewidth0, label='loc 1')
#    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
#    ax.legend(loc='upper left')
#    ax.set_xlabel('Time (kyr)')
#    ax.set_ylabel(r'$\dot{\lambda}$ (1/s)', fontsize=fontsize0)

    plt.savefig(A.output_dir+'magma_tip_time_plot5.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot_all(A,tend):
  try:
    fig = plt.figure(1,figsize=(18,12))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)
    
    xt0 = A.nt_tkyr[A.ind_BTDZ]
    xt1 = A.nt_tkyr[A.ind_brittle]

    ax = plt.subplot(4,2,1)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[0,:ind0[0][0]]),'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_phi[2,:ind0[0][0]]),'b-',linewidth=linewidth0, label='loc 2')
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\log_{10}\phi$', fontsize=fontsize0)
    
    ax = plt.subplot(4,2,2)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_dotlam[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\dot{\lambda}$ (1/s)', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(4,2,3)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_tauII[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\tau_{II}$ (MPa)', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    
    ax = plt.subplot(4,2,4)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_V[0,:ind0[0][0]]),'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_E[0,:ind0[0][0]]),'b-',linewidth=linewidth0,label='E')

    pl = ax.plot(A.nt_epsII_V_avg0[:ind0[0][0],0],np.log10(A.nt_epsII_V_avg0[:ind0[0][0],1]),'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_epsII_E_avg0[:ind0[0][0],0],np.log10(A.nt_epsII_E_avg0[:ind0[0][0],1]),'b-',linewidth=linewidth0,label='E')
    
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII_VP[0,:ind0[0][0]]),'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],np.log10(A.nt_epsII[0,:ind0[0][0]]),'k-',linewidth=linewidth0, label='total')
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$\log_{10}\dot{\epsilon}_{II}$ (1/s) loc=0', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(4,2,5)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_DP[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    pl = ax.plot(A.nt_DP_avg2[:ind0[0][0],0],A.nt_DP_avg2[:ind0[0][0],1],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\Delta P$ (MPa)', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    
    ax = plt.subplot(4,2,6)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_V[2,:ind0[0][0]],'r-',linewidth=linewidth0,label='V')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_E[2,:ind0[0][0]],'b-',linewidth=linewidth0,label='E')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC_VP[2,:ind0[0][0]],'g-',linewidth=linewidth0,label='VP')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_volC[2,:ind0[0][0]],'k-',linewidth=linewidth0,label='total')
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$\mathcal{C}$ (1/s) loc=2', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    ax = plt.subplot(4,2,7)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[0,:ind0[0][0]],'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_vfz[2,:ind0[0][0]],'b-',linewidth=linewidth0, label='loc 2')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$V_\ell^z$ (cm/yr)', fontsize=fontsize0)
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')
    ax.set_xlabel('Time (kyr)')

    ax = plt.subplot(4,2,8)
    ax.grid(True)
    pl = ax.plot(A.nt_tkyr[:ind0[0][0]],A.nt_zmagma[0,:ind0[0][0]],'k-',linewidth=linewidth0)
    ax.set_ylabel(r'Depth (km)', fontsize=fontsize0)
    ax.set_xlabel('Time (kyr)')
    plt.axvline(x = xt0, color = 'k', linestyle='--')
    plt.axvline(x = xt1, color = 'k', linestyle='--')

    plt.savefig(A.output_dir+'magma_tip_time_plot_all.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
def plot6(A,tend):
  try:
    fig = plt.figure(1,figsize=(10,12))
    linewidth0 = 0.7
    fontsize0 = 14
    ind0 = np.where(A.nt_tkyr>=tend)
    
    xmin = 0
    xmax = 200

#    dt = (A.nt_tkyr[1]-A.nt_tkyr[0])
    dt = (A.nt_tkyr[1]-A.nt_tkyr[0])*1e3*(365*24*3600)
    fs = 1.0/dt

    ax = plt.subplot(511)
    ax.grid(True)
    (f, S) = scipy.signal.periodogram(A.nt_phi[0,:ind0[0][0]],fs,scaling='density')
    (f1, S1) = scipy.signal.periodogram(A.nt_phi[1,:ind0[0][0]],fs,scaling='density')
    (f2, S2) = scipy.signal.periodogram(A.nt_phi[2,:ind0[0][0]],fs,scaling='density')

#    pl = ax.plot(f,S,'k-',linewidth=linewidth0, label='loc 0')
#    pl = ax.plot(f1,S1,'r-',linewidth=linewidth0, label='loc 1')
#    pl = ax.plot(f2,S2,'b-',linewidth=linewidth0, label='loc 2')
    
    pl = ax.plot(1.0/f/(365*24*3600)/1e3,S,'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(1.0/f1/(365*24*3600)/1e3,S1,'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(1.0/f2/(365*24*3600)/1e3,S2,'b-',linewidth=linewidth0, label='loc 2')
    
    ax.set_xlim(xmin,xmax)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,ymax/2)
    ax.legend(loc='upper right')
    ax.set_ylabel(r'psd $\phi$', fontsize=fontsize0)
#    ax.set_xlabel(r'Frequency (Hz)', fontsize=fontsize0)
#    ax.set_xlabel(r'Period (kyr)', fontsize=fontsize0)

    ax = plt.subplot(512)
    ax.grid(True)
    (f0, S0) = scipy.signal.periodogram(A.nt_tauII[0,:ind0[0][0]],fs,scaling='density')
    (f1, S1) = scipy.signal.periodogram(A.nt_tauII[1,:ind0[0][0]],fs,scaling='density')
    (f2, S2) = scipy.signal.periodogram(A.nt_tauII[2,:ind0[0][0]],fs,scaling='density')
    
    pl = ax.plot(1.0/f0/(365*24*3600)/1e3,S0,'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(1.0/f1/(365*24*3600)/1e3,S1,'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(1.0/f2/(365*24*3600)/1e3,S2,'b-',linewidth=linewidth0, label='loc 2')
    ax.set_xlim(xmin,xmax)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,ymax/2)
    ax.legend(loc='upper right')
    ax.set_ylabel(r'psd $\tau_{II}$', fontsize=fontsize0)

    ax = plt.subplot(513)
    ax.grid(True)
    (f0, S0) = scipy.signal.periodogram(A.nt_DP[0,:ind0[0][0]],fs,scaling='density')
    (f1, S1) = scipy.signal.periodogram(A.nt_DP[1,:ind0[0][0]],fs,scaling='density')
    (f2, S2) = scipy.signal.periodogram(A.nt_DP[2,:ind0[0][0]],fs,scaling='density')
    
    pl = ax.plot(1.0/f0/(365*24*3600)/1e3,S0,'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(1.0/f1/(365*24*3600)/1e3,S1,'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(1.0/f2/(365*24*3600)/1e3,S2,'b-',linewidth=linewidth0, label='loc 2')
    ax.set_xlim(xmin,xmax)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,ymax/2)
    ax.legend(loc='upper right')
    ax.set_ylabel(r'psd $\Delta P$', fontsize=fontsize0)

    ax = plt.subplot(514)
    ax.grid(True)
    (f0, S0) = scipy.signal.periodogram(A.nt_vfz[0,:ind0[0][0]],fs,scaling='density')
    (f1, S1) = scipy.signal.periodogram(A.nt_vfz[1,:ind0[0][0]],fs,scaling='density')
    (f2, S2) = scipy.signal.periodogram(A.nt_vfz[2,:ind0[0][0]],fs,scaling='density')
    
    pl = ax.plot(1.0/f0/(365*24*3600)/1e3,S0,'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(1.0/f1/(365*24*3600)/1e3,S1,'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(1.0/f2/(365*24*3600)/1e3,S2,'b-',linewidth=linewidth0, label='loc 2')
    ax.set_xlim(xmin,xmax)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,ymax/2)
    ax.legend(loc='upper right')
    ax.set_ylabel(r'psd $v_\ell^z$', fontsize=fontsize0)

    ax = plt.subplot(515)
    ax.grid(True)
    (f0, S0) = scipy.signal.periodogram(A.nt_dotlam[0,:ind0[0][0]],fs,scaling='density')
    (f1, S1) = scipy.signal.periodogram(A.nt_dotlam[1,:ind0[0][0]],fs,scaling='density')
    (f2, S2) = scipy.signal.periodogram(A.nt_dotlam[2,:ind0[0][0]],fs,scaling='density')
    
    pl = ax.plot(1.0/f0/(365*24*3600)/1e3,S0,'k-',linewidth=linewidth0, label='loc 0')
    pl = ax.plot(1.0/f1/(365*24*3600)/1e3,S1,'r-',linewidth=linewidth0, label='loc 1')
    pl = ax.plot(1.0/f2/(365*24*3600)/1e3,S2,'b-',linewidth=linewidth0, label='loc 2')
    ax.set_xlim(xmin,xmax)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,ymax/2)
    ax.legend(loc='upper right')
    ax.set_ylabel(r'psd $\dot{\lambda}$', fontsize=fontsize0)
    ax.set_xlabel(r'Period (kyr)', fontsize=fontsize0)

    plt.savefig(A.output_dir+'magma_tip_time_plot6.png', bbox_inches = 'tight')
    plt.close()

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0
# ---------------------------------
# Main script
# ---------------------------------

tend = 1000
sim0 = 'run51_VEVP_var_age_eta1e18_Vext0/'
path_in ='/Users/apusok/Documents/morfault/'
path_out='/Users/apusok/Documents/morfault/Figures/'+sim0

#sim0 = 'c00_age10/'
#tend = 535 # kyr

#sim0 = 'c00_age20/'
#tend = 850 # kyr

#sim0 = 'c00_age40/'
#tend = 960 # kyr

#sim0 = 'c00_age60/'
#tend = 1430 # kyr

#sim0 = 'c00_age80/'
#tend = 1100 # kyr

#sim0 = 'c00_age100/'
#tend = 1110 # kyr

#sim0 = 'c00_age40_phimin1e-3/'
#tend = 1000 # kyr

# sim0 = 'c00_age40_sigma1e-3/'
# tend = 1950 # kyr

# path_in ='/data/magmox/sann3352/cider_lab/'
# path_out='/data/magmox/sann3352/Figures_cider_lab/'+sim0

read_data = 1

A = SimStruct()
A.input = path_in+sim0
A.output_dir= path_out
vizB.make_dir(A.output_dir)

A.tend = tend
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

# LOCATION PLOTS - range(0,9) but most info is 0,1,2
for itip in range(0,3):
  plot1(A,itip,tend)
  plot2(A,itip,tend)
  plot3(A,itip,tend)
  plot4(A,itip,tend)

# PLOTS
plot5(A,tend)
plot6(A,tend)
plot_all(A,tend)

print('Ductile-BDTZ')
print('t = '+str(A.nt_tkyr[A.ind_BTDZ])+' kyr, z = '+str(A.nt_zmagma[0][A.ind_BTDZ])+' km')

print('BDTZ-Diking')
print('t = '+str(A.nt_tkyr[A.ind_brittle])+' kyr, z = '+str(A.nt_zmagma[0][A.ind_brittle])+' km')

## Print
#t1 = 715
#t2 = 1100
#ind0 = np.where(A.nt_tkyr>=t1)
#ind1 = np.where(A.nt_tkyr>=t2)
#hdike = (A.nt_zmagma[ind1[0][0]]-A.nt_zmagma[ind0[0][0]])*1e3
#vdike = hdike/(t2-t1)/1e3
#print('Dike height (m): ',hdike)
#print('Dike velocity (m/yr): ',vdike)
#print('Dike T start (deg C): ',A.nt_T[ind0[0][0]])
#print('Dike z start (m): ',A.nt_zmagma[ind0[0][0]]*1e3)
