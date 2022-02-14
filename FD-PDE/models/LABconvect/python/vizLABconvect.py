# ---------------------------------
# Load modules
# ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

from matplotlib import rc
import importlib
import os

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class EmptyStruct:
  pass

# ---------------------------------
# Definitions
# ---------------------------------
def correct_path_load_data(fname):
  try: # try to open file .py file
    fname_new = fname[:-3]+'_new.py'

    # Copy new file
    f = open(fname, 'r')
    f_new = open(fname_new, 'w')
    line_prev = ''

    for line in f:
      if 'def _PETScBinaryFilePrefix():' in line_prev:
        line = '  return "'+fname[:-3]+'"\n'

      if '  filename = ' in line:
        line = '  filename = "'+fname[:-3]+'.pbin"\n'
      
      if '  print(\'Filename:' in line:
        line = '  print(\'Filename: '+fname[:-3]+'.pbin\')\n'

      f_new.write(line)
      line_prev = line

    f.close()
    f_new.close()

    # remove new file
    os.system('cp '+fname_new+' '+fname)
    os.system('rm '+fname_new)

  except OSError:
    print('Cannot open:', fname)

# ---------------------------------
def parse_parameters_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Sort data
    scal = EmptyStruct()
    scal.SEC_YEAR = 31536000

    scal.x = data['scalx'][0]
    scal.v = data['scalv'][0]
    scal.t = data['scalt'][0]
    scal.K = data['scalK'][0]
    scal.P = data['scalP'][0]
    scal.eta = data['scaleta'][0]
    scal.rho = data['scalrho'][0]
    scal.H = data['scalH'][0]
    scal.Gamma = data['scalGamma'][0]
    scal.C0 = data['C0'][0]
    scal.DC = data['DC'][0]
    scal.T0 = data['T0'][0]
    scal.DT = data['DT'][0]

    nd = EmptyStruct()
    nd.L = data['L'][0]
    nd.H = data['H'][0]
    nd.xmin = data['xmin'][0]
    nd.zmin = data['zmin'][0]
    nd.age = data['age'][0]
    nd.visc_ratio = data['visc_ratio'][0]
    nd.eta_min = data['eta_min'][0]
    nd.eta_max = data['eta_max'][0]
    nd.istep = data['istep'][0]
    nd.t = data['t'][0]
    nd.dt = data['dt'][0]
    nd.tmax = data['tmax'][0]
    nd.dtmax = data['dtmax'][0]

    nd.delta = data['delta'][0]
    nd.alpha_s = data['alpha_s'][0]
    nd.beta_s = data['beta_s'][0]
    nd.A = data['A'][0]
    nd.S = data['S'][0]
    nd.PeT = data['PeT'][0]
    nd.PeC = data['PeC'][0]
    nd.thetaS = data['thetaS'][0]
    nd.G = data['G'][0]
    nd.RM = data['RM'][0]

    # geoscal
    geoscal = EmptyStruct()
    geoscal.x  = 1e-3 # km
    geoscal.P  = 1e-6 # MPa
    geoscal.v  = 1.0e2*scal.SEC_YEAR # cm/yr
    geoscal.T  = 273.15 # deg C
    geoscal.Gamma= 1000*scal.SEC_YEAR # g/m3/yr
    geoscal.t  = 1.0/scal.SEC_YEAR

    return scal, nd, geoscal
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_time_info_parameters_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    istep = data['istep'][0]
    t = data['t'][0]
    dt = data['dt'][0]

    return istep, dt, t
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def create_labels():
  try: 
    # Create data object
    lbl     = EmptyStruct()
    lbl.nd  = EmptyStruct()
    lbl.dim = EmptyStruct()

    # Units and labels - nondimensional
    lbl.nd.x = r'x/H [-]'
    lbl.nd.z = r'z/H [-]'
    lbl.nd.P = r'$P$ [-]'
    lbl.nd.Pc = r'$P_c$ [-]'
    lbl.nd.vs= r'$V_s$ [-]'
    lbl.nd.vsx= r'$V_s^x$ [-]'
    lbl.nd.vsz= r'$V_s^z$ [-]'
    lbl.nd.vf= r'$V_f$ [-]'
    lbl.nd.vfx= r'$V_f^x$ [-]'
    lbl.nd.vfz= r'$V_f^z$ [-]'
    lbl.nd.v = r'$V$ [-]'
    lbl.nd.vx= r'$V^x$ [-]'
    lbl.nd.vz= r'$V^z$ [-]'
    lbl.nd.C = r'$\Theta$ [-]'
    lbl.nd.H = r'$H$ [-]'
    lbl.nd.phi = r'$\phi$ [-]'
    lbl.nd.T = r'$\tilde{\theta}$ [-]'
    lbl.nd.TP = r'$\theta$ [-]'
    lbl.nd.Cf = r'$\Theta_f$ [-]'
    lbl.nd.Cs = r'$\Theta_s$ [-]'
    lbl.nd.Plith = r'$P_{lith}$ [-]'
    lbl.nd.resP = r'res $P$ [-]'
    lbl.nd.resPc = r'res $P_c$ [-]'
    lbl.nd.resvsx= r'res $V_s^x$ [-]'
    lbl.nd.resvsz= r'res $V_s^z$ [-]'
    lbl.nd.resC = r'res $\Theta$ [-]'
    lbl.nd.resH = r'res $H$ [-]'
    lbl.nd.eta = r'$\eta$ [-]'
    lbl.nd.zeta = r'$\zeta$ [-]'
    lbl.nd.K = r'$K$ [-]'
    lbl.nd.rho = r'$\rho$ [-]'
    lbl.nd.rhof = r'$\rho_f$ [-]'
    lbl.nd.rhos = r'$\rho_s$ [-]'
    lbl.nd.Gamma = r'$\Gamma$ [-]'
    lbl.nd.divmass = r'$\nabla\cdot(v)$ [-]'

    # Units and labels - dimensional
    lbl.dim.P = r'$P$ [MPa]'
    lbl.dim.Pc = r'$P_c$ [MPa]'
    lbl.dim.x = r'x [km]'
    lbl.dim.z = r'z [km]'      
    lbl.dim.vs= r'$V_s$ [cm/yr]'
    lbl.dim.vsx= r'$V_s^x$ [cm/yr]'
    lbl.dim.vsz= r'$V_s^z$ [cm/yr]'
    lbl.dim.vf= r'$V_f$ [cm/yr]'
    lbl.dim.vfx= r'$V_f^x$ [cm/yr]'
    lbl.dim.vfz= r'$V_f^z$ [cm/yr]'
    lbl.dim.v = r'$V$ [cm/yr]'
    lbl.dim.vx= r'$V^x$ [cm/yr]'
    lbl.dim.vz= r'$V^z$ [cm/yr]'
    lbl.dim.H = r'$H$ [J/kg]'
    lbl.dim.C = r'$C$ [wt. frac.]'
    lbl.dim.Cf = r'$C_f$ [wt. frac.]'
    lbl.dim.Cs = r'$C_s$ [wt. frac.]'
    lbl.dim.T = r'$T$ $[^oC]$'
    lbl.dim.TP = r'$T$ potential $[^oC]$'
    lbl.dim.Plith = r'$P_{lith}$ [MPa]'
    lbl.dim.phi = r'$\phi$ '
    lbl.dim.resP = r'res $P$ [MPa]'
    lbl.dim.resPc = r'res $P_c$ [MPa]'
    lbl.dim.resvsx= r'res $V_s^x$ [cm/yr]'
    lbl.dim.resvsz= r'res $V_s^z$ [cm/yr]'
    lbl.dim.resC = r'res $C$ [wt. frac.]'
    lbl.dim.resH = r'res $H$ [J/kg]'
    lbl.dim.eta = r'$\eta$ [Pa.s]'
    lbl.dim.zeta = r'$\zeta$ [Pa.s]'
    lbl.dim.K = r'$K$ [m2]'
    lbl.dim.rho = r'$\rho$ [kg/m3]'
    lbl.dim.rhof = r'$\rho_f$ [kg/m3]'
    lbl.dim.rhos = r'$\rho_s$ [kg/m3]'
    lbl.dim.Gamma = r'$\Gamma$ [g/m$^3$/yr]'
    lbl.dim.divmass = r'$\nabla\cdot v$ [/s]'

    return lbl
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------------
def parse_solver_log_file(fname):
  tstep = 0
  try: # try to open directory

    finished_sim = 0
    # parse number of timesteps
    f = open(fname, 'r')
    i0=0
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
      if '# Runtime:' in line: 
        finished_sim = 1
    f.close()
    tstep = i0

    if (finished_sim==0): 
      tstep -= 1

    # Convergence 
    sol = EmptyStruct()
    sol.HCres = np.zeros(tstep+1)
    sol.PVres = np.zeros(tstep+1)
    sol.HCres_diverged = np.nan*np.zeros(tstep+1)
    sol.PVres_diverged = np.nan*np.zeros(tstep+1)
    sol.dt = np.zeros(tstep+1)
    sol.runtime = np.zeros(tstep+1)

    # Parse output and save norm info
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
      
      if (i0<tstep):
        if 'SNES Function norm' in line:
          snes_norm = float(line[23:41])

        if 'Nonlinear hc_ solve converged' in line:
          sol.HCres[i0+1] = snes_norm 
        if 'Nonlinear pv_ solve converged' in line:
          sol.PVres[i0+1] = snes_norm
        if 'Nonlinear hc_ solve did not converge' in line:
          sol.HCres[i0+1] = snes_norm
          sol.HCres_diverged[i0+1] = snes_norm
        if 'Nonlinear pv_ solve did not converge' in line:
          sol.PVres[i0+1] = snes_norm
          sol.PVres_diverged[i0+1] = snes_norm

        if '# TIME:' in line:
          sol.dt[i0+1] = float(line[44:62])
        
        if '# Timestep runtime:' in line:
          # print(line)
          ij = line.find('(')
          sol.runtime[i0+1] = float(line[20:ij-1])

        line_prev = line

    f.close()

    return tstep, sol
  except OSError:
    print('Cannot open:', fname)
    return tstep, 0

# ---------------------------------
def parse_grid_info(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    grid = EmptyStruct()
    grid.nx = data['Nx'][0]
    grid.nz = data['Ny'][0]
    grid.xc = data['x1d_cell']
    grid.zc = data['y1d_cell']
    grid.xv = data['x1d_vertex']
    grid.zv = data['y1d_vertex']

    return grid
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_PV3_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    vxr= data['X_face_x']
    vzr= data['X_face_y']
    pr = data['X_cell']
    dof = 2

    pi  = pr[0::dof]
    pci = pr[1::dof]

    # Reshape data in 2D
    P  = pi.reshape(nz,nx)
    Pc = pci.reshape(nz,nx)
    Vx = vxr.reshape(nz,nx+1)
    Vz = vzr.reshape(nz+1,nx)

    return P,Pc,Vx,Vz
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_HC_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    HC = data['X_cell']
    dof = 2
    Hr = HC[0::dof]
    Cr = HC[1::dof]

    # Reshape data in 2D
    H = Hr.reshape(nz,nx)
    C = Cr.reshape(nz,nx)

    return H,C
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_Vel_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    Velx = data['X_face_x']
    Velz = data['X_face_y']
    dof = 2
    vfxr = Velx[0::dof]
    vbxr = Velx[1::dof]
    vfzr = Velz[0::dof]
    vbzr = Velz[1::dof]

    # Reshape data in 2D
    Vfx = vfxr.reshape(nz,nx+1)
    Vfz = vfzr.reshape(nz+1,nx)
    Vbx = vbxr.reshape(nz,nx+1)
    Vbz = vbzr.reshape(nz+1,nx)

    return Vfx,Vfz,Vbx,Vbz
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_Enth_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    En_data = data['X_cell']
    dof_en = 5+3*2

    Hr  = En_data[0::dof_en]
    Tr  = En_data[1::dof_en]
    TPr = En_data[2::dof_en]
    phir= En_data[3::dof_en]
    Pr  = En_data[4::dof_en]
    Cr  = En_data[5::dof_en]
    Csr = En_data[7::dof_en]
    Cfr = En_data[9::dof_en]

    # Reshape data in 2D
    Enth = EmptyStruct()
    Enth.H  = Hr.reshape(nz,nx)
    Enth.T  = Tr.reshape(nz,nx)
    Enth.TP = TPr.reshape(nz,nx)
    Enth.phi= phir.reshape(nz,nx)
    Enth.P  = Pr.reshape(nz,nx)
    Enth.C  = Cr.reshape(nz,nx)
    Enth.Cs = Csr.reshape(nz,nx)
    Enth.Cf = Cfr.reshape(nz,nx)

    return Enth
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_matProps_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    Matprops_data = data['X_cell']
    dof = 7

    etar  = Matprops_data[0::dof]
    zetar = Matprops_data[1::dof]
    Kr    = Matprops_data[2::dof]
    rhor  = Matprops_data[3::dof]
    rhofr = Matprops_data[4::dof]
    rhosr = Matprops_data[5::dof]
    Gammar= Matprops_data[6::dof]

    # Reshape data in 2D
    matProp = EmptyStruct()
    matProp.eta  = etar.reshape(nz,nx)
    matProp.zeta = zetar.reshape(nz,nx)
    matProp.K    = Kr.reshape(nz,nx)
    matProp.rho  = rhor.reshape(nz,nx)
    matProp.rhof = rhofr.reshape(nz,nx)
    matProp.rhos = rhosr.reshape(nz,nx)
    matProp.Gamma= Gammar.reshape(nz,nx)

    return matProp
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_PVcoeff3_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    data_v = data['X_vertex']
    data_fx= data['X_face_x']
    data_fz= data['X_face_y']
    data_c = data['X_cell']

    # Separate coefficients
    dof = 4
    Cr = data_c[0::dof]
    Ar = data_c[1::dof]
    D1r= data_c[2::dof]
    DCr= data_c[3::dof]

    Bxr = data_fx[0::dof]
    D2xr= data_fx[1::dof]
    D3xr= data_fx[2::dof]
    D4xr= data_fx[3::dof]

    Bzr = data_fz[0::dof]
    D2zr= data_fz[1::dof]
    D3zr= data_fz[2::dof]
    D4zr= data_fz[3::dof]

    # Reshape data in 2D
    coeff = EmptyStruct()
    coeff.A_cor = data_v.reshape(nz+1,nx+1)
    coeff.C = Cr.reshape(nz,nx)
    coeff.A = Ar.reshape(nz,nx)
    coeff.D1= D1r.reshape(nz,nx)
    coeff.DC= DCr.reshape(nz,nx)

    coeff.Bx  = Bxr.reshape(nz,nx+1)
    coeff.D2x = D2xr.reshape(nz,nx+1)
    coeff.D3x = D3xr.reshape(nz,nx+1)
    coeff.D4x = D4xr.reshape(nz,nx+1)

    coeff.Bz  = Bzr.reshape(nz+1,nx)
    coeff.D2z = D2zr.reshape(nz+1,nx)
    coeff.D3z = D3zr.reshape(nz+1,nx)
    coeff.D4z = D4zr.reshape(nz+1,nx)

    return coeff
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_HCcoeff_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    data_fx= data['X_face_x']
    data_fz= data['X_face_y']
    data_c = data['X_cell']

    # Separate coefficients
    dof = 6
    A1r = data_c[0::dof]
    B1r = data_c[1::dof]
    D1r = data_c[2::dof]
    A2r = data_c[3::dof]
    B2r = data_c[4::dof]
    D2r = data_c[5::dof]

    dof = 5
    C1xr = data_fx[0::dof]
    C2xr = data_fx[1::dof]
    vxr  = data_fx[2::dof]
    vfxr = data_fx[3::dof]
    vsxr = data_fx[4::dof]

    C1zr = data_fz[0::dof]
    C2zr = data_fz[1::dof]
    vzr  = data_fz[2::dof]
    vfzr = data_fz[3::dof]
    vszr = data_fz[4::dof]

    # Reshape data in 2D
    coeff = EmptyStruct()
    coeff.A1 = A1r.reshape(nz,nx)
    coeff.B1 = B1r.reshape(nz,nx)
    coeff.D1 = D1r.reshape(nz,nx)
    coeff.A2 = A2r.reshape(nz,nx)
    coeff.B2 = B2r.reshape(nz,nx)
    coeff.D2 = D2r.reshape(nz,nx)

    coeff.C1x = C1xr.reshape(nz,nx+1)
    coeff.C2x = C2xr.reshape(nz,nx+1)
    coeff.vx  = vxr.reshape(nz,nx+1)
    coeff.vfx = vfxr.reshape(nz,nx+1)
    coeff.vsx = vsxr.reshape(nz,nx+1)

    coeff.C1z = C1zr.reshape(nz+1,nx)
    coeff.C2z = C2zr.reshape(nz+1,nx)
    coeff.vz  = vzr.reshape(nz+1,nx)
    coeff.vfz = vfzr.reshape(nz+1,nx)
    coeff.vsz = vszr.reshape(nz+1,nx)

    return coeff
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def calc_center_velocities(vx,vz,nx,nz):

  vxc  = np.zeros([nz,nx])
  vzc  = np.zeros([nz,nx])

  for i in range(0,nx):
    for j in range(0,nz):
      vxc[j][i]  = 0.5 * (vx[j][i+1] + vx[j][i])
      vzc[j][i]  = 0.5 * (vz[j+1][i] + vz[j][i])

  return vxc, vzc

# ---------------------------------
def calc_divergence(vx,vz,dx,dz,nx,nz):

  div  = np.zeros([nz,nx])

  for i in range(0,nx):
    for j in range(0,nz):
      div[j][i]  = (vx[j][i+1] - vx[j][i])/dx + (vz[j+1][i] - vz[j][i])/dz

  return div

# ---------------------------------
def plot_solver_residuals(A,fname):

  fig = plt.figure(1,figsize=(10,10))

  ax = plt.subplot(3,1,1)
  ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.HCres[1:-1]), linewidth=0.1)
  ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.HCres_diverged[1:-1]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(HC residual)')

  ax = plt.subplot(3,1,2)
  pl = ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.PVres[1:-1]), linewidth=0.1)
  ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.PVres_diverged[1:-1]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(PV residual)')

  ax = plt.subplot(3,1,3)
  pl = ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.dt[1:-1]), linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(dt)')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_standard(fig,ax,X,extent,title,lblx,lblz):
  im = ax.imshow(X,extent=extent,cmap='viridis',origin='lower')
  fig.colorbar(im,ax=ax, shrink=0.80)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(title)
  return im

# ---------------------------------
def get_scaling(A,lbl,dim,extra):
  if (dim):
    if (extra):
      exec("global scal; scal = A.scal.%s*A.geoscal.%s" % (lbl,lbl))
    else:
      exec("global scal; scal = A.scal.%s" % (lbl))
    global scal
  else: 
    scal = 1.0
  return scal

# ---------------------------------
def scale_TC(A,lbl1,lbl2,dim,extra):
  if (dim):
    if (extra):
      exec("global TC; TC = A.%s * A.scal.D%s + A.scal.%s0 - A.geoscal.%s" % (lbl1,lbl2,lbl2,lbl2))
    else:
      exec("global TC; TC = A.%s * A.scal.D%s + A.scal.%s0" % (lbl1,lbl2,lbl2))
  else: 
    exec("global TC; TC = A.%s" % (lbl1))
  global TC
  return TC

# ---------------------------------
def get_label(A,lbl_i,dim):
  if (dim):
    exec("global lbl_out; lbl_out = A.lbl.dim.%s" % (lbl_i))
  else:
    exec("global lbl_out; lbl_out = A.lbl.nd.%s" % (lbl_i))
  global lbl_out
  return lbl_out

# ---------------------------------
def plot_PV3(iplot,A,istart,iend,jstart,jend,fname,istep,dim):
  fig = plt.figure(1,figsize=(14,8))

  # return scaling params and labels
  scalP = get_scaling(A,'P',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalx = get_scaling(A,'x',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  if (iplot==0):
    X1 = A.P*scalP
    X2 = A.Vsx*scalv
    X3 = A.Vsz*scalv
    X4 = A.Pc*scalP
    lbl1 = get_label(A,'P',dim)
    lbl2 = get_label(A,'vsx',dim)
    lbl3 = get_label(A,'vsz',dim)
    lbl4 = get_label(A,'Pc',dim)

  if (iplot==1):
    X1 = A.resP*scalP
    X2 = A.resVsx*scalv
    X3 = A.resVsz*scalv
    X4 = A.resPc*scalP
    lbl1 = get_label(A,'resP',dim)
    lbl2 = get_label(A,'resvsx',dim)
    lbl3 = get_label(A,'resvsz',dim)
    lbl4 = get_label(A,'resPc',dim)

  X1 = X1[jstart:jend  ,istart:iend  ]
  X2 = X2[jstart:jend  ,istart:iend+1]
  X3 = X3[jstart:jend+1,istart:iend  ]
  X4 = X4[jstart:jend  ,istart:iend  ]

  extentP =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,2,1)
  plot_standard(fig,ax,X1,extentP,lbl1+' tstep = '+str(istep),lblx,lblz)

  # # add streamlines for pressure plot
  # if (iplot==0):
  #   xa = A.grid.xc[::4]*scalx
  #   stream_points = []
  #   for xi in xa:
  #     stream_points.append([xi,A.grid.zc[0]*scalx])
  #   ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, A.Vscx*scalv, A.Vscz*scalv,color='k',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  ax = plt.subplot(2,2,2)
  plot_standard(fig,ax,X4,extentP,lbl4,lblx,lblz)

  ax = plt.subplot(2,2,3)
  plot_standard(fig,ax,X2,extentVx,lbl2,lblx,lblz)

  ax = plt.subplot(2,2,4)
  plot_standard(fig,ax,X3,extentVz,lbl3,lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HC(iplot,A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(7,8))

  # return scaling params and labels
  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  # H,C
  if (iplot==0):
    scalH = get_scaling(A,'H',dim,0)
    X1 = A.H*scalH
    X2 = scale_TC(A,'C','C',dim,0)
    lbl1 = get_label(A,'H',dim)
    lbl2 = get_label(A,'C',dim)
  
  # res H,C
  if (iplot==1):
    scalH = get_scaling(A,'H',dim,0)
    X1 = A.resH*scalH
    X2 = scale_TC(A,'resC','C',dim,0)
    lbl1 = get_label(A,'resH',dim)
    lbl2 = get_label(A,'resC',dim)

  X1 = X1[jstart:jend,istart:iend]
  X2 = X2[jstart:jend,istart:iend]

  ax = plt.subplot(2,1,1)
  plot_standard(fig,ax,X1,extentC,lbl1+' tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(2,1,2)
  plot_standard(fig,ax,X2,extentC,lbl2,lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Enth(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,14))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  ax = plt.subplot(4,2,1)
  scal = get_scaling(A,'H',dim,0)
  lbl  = get_label(A,'H',dim)
  plot_standard(fig,ax,A.Enth.H[jstart:jend,istart:iend]*scal,extentC,lbl+' tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(4,2,2)
  X = scale_TC(A,'Enth.C','C',dim,0)
  lbl = get_label(A,'C',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,3)
  X = scale_TC(A,'Enth.T','T',dim,1)
  lbl = get_label(A,'T',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,4)
  X = scale_TC(A,'Enth.Cf','C',dim,0)
  lbl = get_label(A,'Cf',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,5)
  X = scale_TC(A,'Enth.TP','T',dim,1)
  lbl = get_label(A,'TP',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,6)
  X = scale_TC(A,'Enth.Cs','C',dim,0)
  lbl = get_label(A,'Cs',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,7)
  scal = get_scaling(A,'P',dim,1)
  lbl  = get_label(A,'Plith',dim)
  plot_standard(fig,ax,A.Enth.P[jstart:jend,istart:iend]*scal,extentC,lbl,lblx,lblz)

  ax = plt.subplot(4,2,8)
  lbl  = get_label(A,'phi',dim)
  plot_standard(fig,ax,A.Enth.phi[jstart:jend,istart:iend],extentC,lbl,lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Vel(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,7))

  scalv = get_scaling(A,'v',dim,1)
  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentVx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,2,1)
  lbl = get_label(A,'vfx',dim)
  plot_standard(fig,ax,A.Vfx[jstart:jend  ,istart:iend+1]*scalv,extentVx,lbl+' tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(2,2,2)
  lbl = get_label(A,'vfz',dim)
  plot_standard(fig,ax,A.Vfz[jstart:jend+1,istart:iend  ]*scalv,extentVz,lbl,lblx,lblz)

  ax = plt.subplot(2,2,3)
  lbl = get_label(A,'vx',dim)
  plot_standard(fig,ax,A.Vx[jstart:jend  ,istart:iend+1]*scalv,extentVx,lbl,lblx,lblz)

  ax = plt.subplot(2,2,4)
  lbl = get_label(A,'vz',dim)
  plot_standard(fig,ax,A.Vz[jstart:jend+1,istart:iend  ]*scalv,extentVz,lbl,lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff3(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,14))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]
  extentN =[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(4,2,1)
  plot_standard(fig,ax,A.PV_coeff.A_cor[jstart:jend+1,istart:iend+1],extentN,'A corner tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(4,2,2)
  plot_standard(fig,ax,A.PV_coeff.A[jstart:jend  ,istart:iend  ],extentE,'A center',lblx,lblz)

  ax = plt.subplot(4,2,3)
  plot_standard(fig,ax,A.PV_coeff.C[jstart:jend  ,istart:iend  ],extentE,'C center',lblx,lblz)

  ax = plt.subplot(4,2,5)
  plot_standard(fig,ax,A.PV_coeff.D1[jstart:jend  ,istart:iend  ],extentE,'D1 center',lblx,lblz)

  ax = plt.subplot(4,2,6)
  plot_standard(fig,ax,A.PV_coeff.DC[jstart:jend  ,istart:iend  ],extentE,'DC center',lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_center.pdf', bbox_inches = 'tight')
  plt.close()

  # part 2
  fig = plt.figure(1,figsize=(14,14))
  ax = plt.subplot(4,2,1)
  plot_standard(fig,ax,A.PV_coeff.Bx[jstart:jend  ,istart:iend+1],extentFx,'Bx face',lblx,lblz)

  ax = plt.subplot(4,2,2)
  plot_standard(fig,ax,A.PV_coeff.Bz[jstart:jend+1,istart:iend  ],extentFz,'Bz face',lblx,lblz)

  ax = plt.subplot(4,2,3)
  plot_standard(fig,ax,A.PV_coeff.D2x[jstart:jend  ,istart:iend+1],extentFx,'D2x face',lblx,lblz)

  ax = plt.subplot(4,2,4)
  plot_standard(fig,ax,A.PV_coeff.D2z[jstart:jend+1,istart:iend  ],extentFz,'D2z face',lblx,lblz)

  ax = plt.subplot(4,2,5)
  plot_standard(fig,ax,A.PV_coeff.D3x[jstart:jend  ,istart:iend+1],extentFx,'D3x face',lblx,lblz)

  ax = plt.subplot(4,2,6)
  plot_standard(fig,ax,A.PV_coeff.D3z[jstart:jend+1,istart:iend  ],extentFz,'D3z face',lblx,lblz)

  ax = plt.subplot(4,2,7)
  plot_standard(fig,ax,A.PV_coeff.D4x[jstart:jend  ,istart:iend+1],extentFx,'D4x face',lblx,lblz)

  ax = plt.subplot(4,2,8)
  plot_standard(fig,ax,A.PV_coeff.D4z[jstart:jend+1,istart:iend  ],extentFz,'D4z face',lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_face.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HCcoeff(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,17))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(5,2,1)
  plot_standard(fig,ax,A.HC_coeff.A1[jstart:jend  ,istart:iend  ],extentE,'A1 center tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(5,2,2)
  plot_standard(fig,ax,A.HC_coeff.A2[jstart:jend  ,istart:iend  ],extentE,'A2 center',lblx,lblz)

  ax = plt.subplot(5,2,3)
  plot_standard(fig,ax,A.HC_coeff.B1[jstart:jend  ,istart:iend  ],extentE,'B1 center',lblx,lblz)

  ax = plt.subplot(5,2,4)
  plot_standard(fig,ax,A.HC_coeff.B2[jstart:jend  ,istart:iend  ],extentE,'B2 center',lblx,lblz)

  ax = plt.subplot(5,2,5)
  plot_standard(fig,ax,A.HC_coeff.D1[jstart:jend  ,istart:iend  ],extentE,'D1 center',lblx,lblz)

  ax = plt.subplot(5,2,6)
  plot_standard(fig,ax,A.HC_coeff.D2[jstart:jend  ,istart:iend  ],extentE,'D2 center',lblx,lblz)

  ax = plt.subplot(5,2,7)
  plot_standard(fig,ax,A.HC_coeff.C1x[jstart:jend  ,istart:iend+1],extentFx,'C1x face',lblx,lblz)

  ax = plt.subplot(5,2,8)
  plot_standard(fig,ax,A.HC_coeff.C1z[jstart:jend+1,istart:iend  ],extentFz,'C1z face',lblx,lblz)

  ax = plt.subplot(5,2,9)
  plot_standard(fig,ax,A.HC_coeff.C2x[jstart:jend  ,istart:iend+1],extentFx,'C2x face',lblx,lblz)

  ax = plt.subplot(5,2,10)
  plot_standard(fig,ax,A.HC_coeff.C2z[jstart:jend+1,istart:iend  ],extentFz,'C2z face',lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_part1.pdf', bbox_inches = 'tight')
  plt.close()

  # part 2
  fig = plt.figure(1,figsize=(14,17))
  ax = plt.subplot(5,2,1)
  plot_standard(fig,ax,A.HC_coeff.vx[jstart:jend  ,istart:iend+1],extentFx,'vx face tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(5,2,2)
  plot_standard(fig,ax,A.HC_coeff.vz[jstart:jend+1,istart:iend  ],extentFz,'vz face',lblx,lblz)

  ax = plt.subplot(5,2,3)
  plot_standard(fig,ax,A.HC_coeff.vfx[jstart:jend  ,istart:iend+1],extentFx,'vfx face',lblx,lblz)

  ax = plt.subplot(5,2,4)
  plot_standard(fig,ax,A.HC_coeff.vfz[jstart:jend+1,istart:iend  ],extentFz,'vfz face',lblx,lblz)

  ax = plt.subplot(5,2,5)
  plot_standard(fig,ax,A.HC_coeff.vsx[jstart:jend  ,istart:iend+1],extentFx,'vsx face',lblx,lblz)

  ax = plt.subplot(5,2,6)
  plot_standard(fig,ax,A.HC_coeff.vsz[jstart:jend+1,istart:iend  ],extentFz,'vsz face',lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_part2.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_matProp(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,14))
  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  ax = plt.subplot(4,2,1)
  scal = get_scaling(A,'eta',dim,0)
  lbl  = get_label(A,'eta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.eta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl+' tstep = '+str(istep),lblx,lblz)

  ax = plt.subplot(4,2,2)
  scal = get_scaling(A,'eta',dim,0)
  lbl  = get_label(A,'zeta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.zeta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl,lblx,lblz)

  ax = plt.subplot(4,2,3)
  scal = get_scaling(A,'K',dim,0)
  lbl  = get_label(A,'K',dim)
  plot_standard(fig,ax,A.matProp.K[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz)

  ax = plt.subplot(4,2,4)
  scal = get_scaling(A,'rho',dim,0)
  lbl  = get_label(A,'rho',dim)
  im = plot_standard(fig,ax,A.matProp.rho[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz)
  # im.set_clim(2750,3000)

  ax = plt.subplot(4,2,5)
  scal = get_scaling(A,'rho',dim,0)
  lbl  = get_label(A,'rhof',dim)
  plot_standard(fig,ax,A.matProp.rhof[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz)

  ax = plt.subplot(4,2,6)
  scal = get_scaling(A,'rho',dim,0)
  lbl  = get_label(A,'rhos',dim)
  plot_standard(fig,ax,A.matProp.rhos[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz)

  ax = plt.subplot(4,2,7)
  scal = get_scaling(A,'Gamma',dim,1)
  lbl  = get_label(A,'Gamma',dim)
  plot_standard(fig,ax,A.matProp.Gamma[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz)

  ax = plt.subplot(4,2,8)
  scal = get_scaling(A,'t',dim,0)
  lbl  = get_label(A,'divmass',dim)
  plot_standard(fig,ax,A.divmass[jstart:jend  ,istart:iend  ]/scal,extentE,lbl,lblx,lblz)

  plt.tight_layout() 
  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_fields_model(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 20})
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  fig = plt.figure(1,figsize=(32,20))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # 1. POROSITY
  ax = plt.subplot(2,2,1)
  cmap1 = plt.cm.get_cmap('inferno', 20)
  A.Enth.phi[A.Enth.phi==0] = 1e-10
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)

  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)
  cbar.ax.set_title(r'log$_{10}\phi$')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)

  nind_factor = int(A.nz/100)
#  nind = 4*nind_factor
  nind = 4
  # solid streamlines
#  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
#        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv,
  color='lightgrey', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T_HS = A.T_HS*A.scal.DT+A.scal.T0-A.geoscal.T

  T = scale_TC(A,'Enth.T','T',dim,1)
  DT = T-T_HS

  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  # 2. Temperature
  ax = plt.subplot(2,2,2)
  lblT = get_label(A,'T',dim)
  cmap1 = plt.cm.get_cmap('RdBu_r')
  im = ax.imshow(DT[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-50,50)
#   im.set_clim(0,1500)
  lblT = r'$\Delta T$ [$^o$C]'

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  # plot solid velocity field
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='k', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # temperature contour
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblT)

  # 3. Bulk composition
  ax = plt.subplot(2,2,3)
  lblC = get_label(A,'C',dim)
  cmap1 = 'gist_earth_r'
  X = scale_TC(A,'Enth.C','C',dim,0)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0.85,0.88)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblC)

  # 4. Density
  ax = plt.subplot(2,2,4)
  cmap1 = 'terrain'
  lbl  = get_label(A,'rho',dim)
  scal_rho = get_scaling(A,'rho',dim,0)
  rho0 = 3300
  if (istep>0):
    X = A.matProp.rho
  else:
    X = np.ones_like(T)*rho0/scal_rho

  XX = X*scal_rho-rho0
  im = ax.imshow(XX[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-100.0,50.0)
  lblrho = r'$\Delta\rho$ [kg/m3]'

  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblrho)

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_fields_model_nd(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 20})
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  fig = plt.figure(1,figsize=(32,20))
  scalx = 1.0
  scalv = 1.0
  scalt = 1.0

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # 1. POROSITY
  ax = plt.subplot(2,2,1)
  cmap1 = plt.cm.get_cmap('inferno', 20)
  A.Enth.phi[A.Enth.phi==0] = 1e-10
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)

  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)
  cbar.ax.set_title(r'log$_{10}\phi$')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)

  nind_factor = int(A.nz/100)
#  nind = 4*nind_factor
  nind = 4
  # solid streamlines
#  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
#        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv,
  color='lightgrey', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t
  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t,2)))

  # 2. Temperature
  ax = plt.subplot(2,2,2)
  lblT = get_label(A,'T',dim)
  cmap1 = plt.cm.get_cmap('RdBu_r')
  DT = A.Enth.T-A.T_HS
  im = ax.imshow(DT[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
#    im.set_clim(dt0,dt1)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  # plot solid velocity field
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv,
      color='k', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # temperature contour
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblT)

  # 3. Bulk composition
  ax = plt.subplot(2,2,3)
  lblC = get_label(A,'C',dim)
  cmap1 = 'gist_earth_r'
  X = A.Enth.C
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0.0,(0.88-A.scal.C0)/A.scal.DC)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblC)

  # 4. Density
  ax = plt.subplot(2,2,4)
  cmap1 = 'terrain'
  rho0 = 3300/500
  if (istep>0):
    X = A.matProp.rho
  else:
    X = np.ones_like(A.Enth.T)*rho0

  im = ax.imshow(X[jstart:jend  ,istart:iend  ]-rho0,extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-200.0/500,0.0)
  lblrho = r'$\Delta\rho$ [-]'
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblrho)

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()
