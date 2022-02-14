# ---------------------------------
# Load modules
# ---------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
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
    nd.xsill = data['xsill'][0]
    nd.U0 = data['U0'][0]
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

# ---------------------------------------
def parse_outflux_log_file(fname):
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

    # variables - outflow
    outf = EmptyStruct()
    outf.t = np.zeros(tstep+1)
    outf.F = np.zeros(tstep+1)
    outf.C = np.zeros(tstep+1)
    outf.h = np.zeros(tstep+1)
    outf.vfz_max = np.zeros(tstep+1)
    outf.phi_max = np.zeros(tstep+1)
    outf.tstep = np.zeros(tstep+1)
    outf.Asym = np.zeros(tstep+1)

    # Parse output and save norm info
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
        indt = line.find(':')
        outf.tstep[i0] = float(line[10:indt])
      
      # if (i0<tstep):
      if 'xMOR FLUXES:' in line:
        outf.t[i0] = float(line[19:37])
        outf.F[i0] = float(line[82:101])
        outf.C[i0] = float(line[47:66])
        outf.h[i0] = float(line[121:140])

      if 'vfz_max' in line:
        outf.vfz_max[i0] = float(line[54:72])
        outf.phi_max[i0] = float(line[25:43])

      if 'SILL FLUXES:' in line:
        outf.t[i0] = float(line[19:37])
        outf.F[i0] = float(line[82:101])
        outf.C[i0] = float(line[47:66])
        outf.h[i0] = float(line[121:140])

      if 'Asymmetry (full ridge)' in line:
        outf.Asym[i0] = float(line[30:49])

    f.close()

    return outf
  except OSError:
    print('Cannot open:', fname)
    return 0

# ---------------------------------------
def parse_multiple_outflux_log_file(dir,fname_list):

  # use first fname as reference
  fname = dir+'/'+fname_list[0]
  outf= parse_outflux_log_file(fname)

  for ii in range(1,len(fname_list)):
    fname = dir+'/'+fname_list[ii]
    flux = parse_outflux_log_file(fname)
    
    # append to previous list
    tstep_first = flux.tstep[0]
    indi = np.where(outf.tstep == tstep_first)

    if (len(indi[0])==0):
      outf.tstep = np.append(outf.tstep[:-1],flux.tstep)
      outf.t = np.append(outf.t[:-1],flux.t)
      outf.F = np.append(outf.F[:-1],flux.F)
      outf.C = np.append(outf.C[:-1],flux.C)
      outf.h = np.append(outf.h[:-1],flux.h)
      outf.vfz_max = np.append(outf.vfz_max[:-1],flux.vfz_max)
      outf.phi_max = np.append(outf.phi_max[:-1],flux.phi_max)
      outf.Asym = np.append(outf.Asym[:-1],flux.Asym)
    else:
      outf.tstep = np.append(outf.tstep[:indi[0][0]],flux.tstep)
      outf.t = np.append(outf.t[:indi[0][0]],flux.t)
      outf.F = np.append(outf.F[:indi[0][0]],flux.F)
      outf.C = np.append(outf.C[:indi[0][0]],flux.C)
      outf.h = np.append(outf.h[:indi[0][0]],flux.h)
      outf.vfz_max = np.append(outf.vfz_max[:indi[0][0]],flux.vfz_max)
      outf.phi_max = np.append(outf.phi_max[:indi[0][0]],flux.phi_max)
      outf.Asym = np.append(outf.Asym[:indi[0][0]],flux.Asym)

  return outf

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
def plot_extract_outflux(A,fname):

  fig = plt.figure(1,figsize=(10,10))

  ax = plt.subplot(3,1,1)
  ax.plot(A.flux.t[1:-1], A.flux.h[1:-1]/1000, label = 's.s. value = '+str(A.flux.h[-1]/1000))
  plt.grid(True)
  # plt.ylim([0, 10])
  ax.legend()
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('Crustal thickness [km]')

  ax = plt.subplot(3,1,2)
  pl = ax.plot(A.flux.t[1:-1], A.flux.F[1:-1], label = 's.s. value = '+str(A.flux.F[-1]))
  plt.grid(True)
  ax.legend()
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('Flux out [kg/m/yr]')

  ax = plt.subplot(3,1,3)
  pl = ax.plot(A.flux.t[1:-1], A.flux.C[1:-1], label = 's.s. value = '+str(A.flux.C[-1]))
  plt.grid(True)
  ax.legend()
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('C out [wt. frac.]')

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
def plot_porosity_contours(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(14,5))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  # 1. porosity
  ax = plt.subplot(1,2,1)
  # cmap1='ocean_r'
  cmap1 = plt.cm.get_cmap('twilight', 20)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  # im.set_clim(0,0.002)
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, A.Vscx*scalv, A.Vscz*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)

  nind_factor = int(A.nz/100)
  nind = 4*nind_factor
  if (istep>0):
    Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, Vfx*scalv, Vfz*scalv,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4,clip_on=False)
  ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi)
  # ax.set_title(A.lbl.nd.phi+' tstep = '+str(istep)+' time = '+str(t)+' [yr]')
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  # 2. Vertical solid velocity
  ax = plt.subplot(1,2,2)
  #  cmap1 = 'viridis'
  # cmap1 = 'binary'
  cmap1 = 'RdBu'
  im = ax.imshow(A.Vsz[jstart:jend+1,istart:iend  ]*scalv,extent=extentVz,cmap=cmap1,origin='lower')
  im.set_clim(-6.0,6.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  
  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  
  zdist = A.grid.zv[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl = get_label(A,'vsz',dim)
  ax.set_title(lbl)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_temperature_slices(A,fname,istep,dim):

  fig = plt.figure(1,figsize=(4,4))
  scalx = get_scaling(A,'x',dim,1)
  lblz = get_label(A,'z',dim)
  lblT = get_label(A,'T',dim)
  T = scale_TC(A,'T','T',dim,1)

  ax = plt.subplot(1,1,1)
  pl = ax.plot(T[:,1], A.grid.zc*scalx, label = 'x = '+str(A.grid.xc[1]*scalx))
  pl = ax.plot(T[:,50], A.grid.zc*scalx, label = 'x = '+str(A.grid.xc[50]*scalx))
  pl = ax.plot(T[:,-1], A.grid.zc*scalx, label = 'x = '+str(A.grid.xc[-1]*scalx))
  ax.legend()
  plt.grid(True)

  ax.set_xlabel(lblT)
  ax.set_ylabel(lblz)

  plt.savefig(fname+'_dim'+str(dim)+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_solid_stream(A,fname,istep,ext,dim):

  fig = plt.figure(1,figsize=(14,5))

  scalv = get_scaling(A,'v',dim,1)
  scalx = get_scaling(A,'x',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc)*scalx, max(A.grid.xc)*scalx, min(A.grid.zc)*scalx, max(A.grid.zc)*scalx]
  extentVz=[min(A.grid.xc)*scalx, max(A.grid.xc)*scalx, min(A.grid.zv)*scalx, max(A.grid.zv)*scalx]

  # 1. porosity
  ax = plt.subplot(1,2,1)
  lbl = get_label(A,'phi',dim)
  im = ax.imshow(np.log10(A.Enth.phi),extent=extentE,cmap='ocean_r',origin='lower',label=lbl)
  # im.set_clim(0,0.02)
  im.set_clim(-5,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  xa = A.grid.xc[::4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[0]*scalx])

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, A.Vscx*scalv, A.Vscz*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)
  nind_factor = int(A.nz/100)
  nind = 4*nind_factor
  # if (istep>0):
  #   Q  = ax.quiver(A.grid.xc[::nind]*scalx, A.grid.zc[::nind]*scalx, Vfx[::nind,::nind]*scalv, Vfz[::nind,::nind]*scalv, 
  #     color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, Vfx*scalv, Vfz*scalv,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # solid streamlines
  stream = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, A.Vscx*scalv, A.Vscz*scalv,
        color='lightgrey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'T','T',dim,1)
  ts = ax.contour(A.grid.xc*scalx, A.grid.zc*scalx, T, levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc*scalx, A.grid.zc*scalx, A.phi, levels=[1e-8,], colors = ('k',),linewidths=(0.7,), extend='both')

  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(' time = %0.1f [yr]' % t)

  plt.savefig(fname+'.'+ext, bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def extract_plot_vzmax_timeseries(A,input_dir,time_list,tstart,tend,fname,flag,fullridge):

  nt = len(time_list)

  # init data
  Wmax_data = EmptyStruct()
  wmaxi = np.zeros(nt)
  ti    = np.zeros(nt)
  tstepi= np.zeros(nt)
  cnt = 0

  # loop timesteps
  for istep in time_list: #range(A.istep,A.tstep+1,A.tout):
    fdir  = input_dir+'Timestep'+str(istep)
    
    # get data
    correct_path_load_data(fdir+'/parameters.py')
    A.scal, A.nd, A.geoscal = parse_parameters_file('parameters',fdir)
    A.nd.istep, A.nd.dt, A.nd.t = parse_time_info_parameters_file('parameters',fdir)
    
    correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
    A.grid = parse_grid_info('out_xPV_ts'+str(istep),fdir)
    A.P, A.Pc, A.Vsx, A.Vsz = parse_PV3_file('out_xPV_ts'+str(istep),fdir)
    
    correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
    A.Enth = parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
    
    A.dx = A.grid.xc[1]-A.grid.xc[0]
    A.dz = A.grid.zc[1]-A.grid.zc[0]
    A.nx = A.grid.nx
    A.nz = A.grid.nz
    
    scalv = get_scaling(A,'v',A.dimensional,1)
    scalt = get_scaling(A,'t',A.dimensional,1)

    # - extract wmax for every timestep
    ti[cnt]     = A.nd.t
    if (fullridge==0):
      wmaxi[cnt]  = np.max(A.Vsz[0:int(A.nz/2),0:int(A.nx/2)])
    else:
      nmor = int(A.nx/2)
      ndist = int(A.nx/4)
      wmaxi[cnt]  = np.max(A.Vsz[0:int(A.nz/2),nmor-ndist:nmor+ndist])
    tstepi[cnt] = istep
    cnt += 1

    os.system('rm -r '+fdir+'/__pycache__')

  # steady state value
  time_myr = ti*scalt/1e6
  wmax_end = wmaxi[-1]*scalv
  #ind = np.where((time_myr>=tstart) & (time_myr<=tend))
  wmax_avg = np.average(wmaxi[np.where((time_myr>=tstart) & (time_myr<=tend))]*scalv)

  # - plot wmax vs time (new dir)
  if (flag):
    # fig = plt.figure(1,figsize=(10,8))
    fig = plt.figure(1,figsize=(10,5))
    ax = plt.subplot(1,1,1)
    ax.plot(time_myr,wmaxi*scalv,'k', label = 'Wmax ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(wmax_avg,4))+', Wmax ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(wmax_end,4)))
    # ax.plot(tstepi,wmaxi*scalv,'k', label = 's.s. value = '+str(wmax_end))
    plt.grid(True)
    ax.legend()
    ax.set_xlabel('Time [Myr]')
    # ax.set_xlabel('Timestep')
    ax.set_ylabel('Wmax [cm/yr]')
    plt.savefig(fname+'.pdf', bbox_inches = 'tight')
    plt.close()

  # save data
  Wmax_data.nsteps = nt
  Wmax_data.time_nd = ti
  Wmax_data.vzmax_nd = wmaxi
  Wmax_data.tstep = tstepi
  Wmax_data.time_myr = time_myr
  Wmax_data.vzmax_cmyr = wmaxi*scalv
  Wmax_data.tstart_myr = tstart
  Wmax_data.tend_myr = tend
  Wmax_data.wmax_end = wmax_end
  Wmax_data.wmax_avg = wmax_end

  Wmax_data.scalv = scalv
  Wmax_data.scalt = scalt

  # return wmax_avg, wmax_end, time_myr[-1]
  return Wmax_data

# ---------------------------------
def extract_plot_phi0_h0_F0_timeseries(A,input_dir,time_list,tstart,tend,fname,flag):

  nt = len(time_list)
  axis_data = EmptyStruct()

  # init data
  phi0i = np.zeros(nt)
  F0i   = np.zeros(nt)
  h0i   = np.zeros(nt)
  ti    = np.zeros(nt)
  tstepi= np.zeros(nt)
  cnt = 0

  # loop timesteps
  for istep in time_list: #range(A.istep,A.tstep+1,A.tout):
    fdir  = input_dir+'Timestep'+str(istep)
    
    # get data
    correct_path_load_data(fdir+'/parameters.py')
    A.scal, A.nd, A.geoscal = parse_parameters_file('parameters',fdir)
    A.nd.istep, A.nd.dt, A.nd.t = parse_time_info_parameters_file('parameters',fdir)
    
    correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
    A.grid = parse_grid_info('out_xPV_ts'+str(istep),fdir)
    
    correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
    A.Enth = parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
    
    A.dx = A.grid.xc[1]-A.grid.xc[0]
    A.dz = A.grid.zc[1]-A.grid.zc[0]
    A.nx = A.grid.nx
    A.nz = A.grid.nz
    scalt = get_scaling(A,'t',A.dimensional,1)
    scalx = get_scaling(A,'x',A.dimensional,1)

    # - extract max values beneath MOR axis for every timestep
    ind_mor = np.where(A.grid.xc*scalx<=4)

    ti[cnt]     = A.nd.t
    phi0i[cnt]  = np.max(A.Enth.phi[:,0]) #:ind_mor[-1]+1])
    F0i[cnt]    = np.max(A.Enth.Cs[:,0]) #:ind_mor[-1]+1])

    if (istep>0):
      indj = np.where(A.Enth.phi[:,0]>0.0)
      h0i[cnt] = (A.grid.zc[indj[-1][-1]] - A.grid.zc[indj[0][0]])*scalx

    tstepi[cnt] = istep
    cnt += 1

    os.system('rm -r '+fdir+'/__pycache__')

  # steady state value
  time_myr = ti*scalt/1e6
  phi0_end = phi0i[-1]
  phi0_avg = np.average(phi0i[np.where((time_myr>=tstart) & (time_myr<=tend))])

  F0_end = F0i[-1]
  F0_avg = np.average(F0i[np.where((time_myr>=tstart) & (time_myr<=tend))])

  h0_end = h0i[-1]
  h0_avg = np.average(h0i[np.where((time_myr>=tstart) & (time_myr<=tend))])

  # - plot values vs time (new dir)
  if (flag):
    fig = plt.figure(1,figsize=(10,10))
    ax = plt.subplot(3,1,1)
    ax.plot(time_myr,phi0i,'k', label = r'$\phi_0$ ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(phi0_avg,4))+', $\phi_0$ ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(phi0_end,4)))
    plt.grid(True)
    ax.legend()
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$\phi_0$')

    ax = plt.subplot(3,1,2)
    ax.plot(time_myr,F0i,'k', label = r'$F_0$ ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(F0_avg,4))+', $F_0$ ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(F0_end,4)))
    plt.grid(True)
    ax.legend()
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$F_0$')

    ax = plt.subplot(3,1,3)
    ax.plot(time_myr,h0i,'k', label = r'$h_0$ ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(h0_avg,4))+', $h_0$ ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(h0_end,4)))
    plt.grid(True)
    ax.legend()
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$h_0$ [km]')

    plt.savefig(fname+'.pdf', bbox_inches = 'tight')
    plt.close()

  # save data
  axis_data.nsteps = nt
  axis_data.time_nd = ti
  axis_data.tstep = tstepi
  axis_data.time_myr = time_myr

  axis_data.phi0 = phi0i
  axis_data.F0   = F0i
  axis_data.h0_km = h0i

  axis_data.phi0_end = phi0_end
  axis_data.phi0_avg = phi0_avg
  axis_data.F0_end = F0_end
  axis_data.F0_avg = F0_avg
  axis_data.h0_km_end = h0_end
  axis_data.h0_km_avg = h0_avg

  axis_data.tstart_myr = tstart
  axis_data.tend_myr = tend
  axis_data.scalx = scalx
  axis_data.scalt = scalt

  # return phi0_avg, phi0_end, F0_avg, F0_end, h0_avg, h0_end, time_myr[-1]
  return axis_data

# ---------------------------------
def plot_crustal_thickness(A,tstart,tend,fname):

  fig = plt.figure(1,figsize=(10,5))
  ax = plt.subplot(1,1,1)

  time_myr = A.flux.t[1:-1]/1e6
  crust_thick = A.flux.h[1:-1]/1000
  crust_end = crust_thick[-1]
  crust_avg = np.average(crust_thick[np.where((time_myr>=tstart) & (time_myr<=tend))])

  ax.plot(time_myr, crust_thick, 'k', label = 'H ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(crust_avg,4))+', H ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(crust_end,4)))
  plt.grid(True)
  # plt.ylim([0, 10])
  ax.legend()
  ax.set_xlabel('Time [Myr]')
  ax.set_ylabel('Crustal thickness [km]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_convergence_steady_state(A,tstart,tend,fname):

  fig = plt.figure(1,figsize=(10,5))
  ax = plt.subplot(1,1,1)

  time_myr = A.flux.t[1:-1]/1e6
  crust_thick = A.flux.h[1:-1]/1000
  dHdt = np.abs((crust_thick[1:]-crust_thick[0:-1])/(time_myr[1:]-time_myr[0:-1]))
  dH = np.abs(crust_thick[1:]-crust_thick[0:-1])

  # ax.plot(np.log10(dHdt), 'k')
  ax.plot(np.log10(dH), 'k')
  plt.grid(True)
  # ax.legend()
  ax.set_xlabel('tstep [-]')
  # ax.set_ylabel(r'log10$\left|\frac{H-Hprev}{t-tprev}\right|$ [km/Myr]')
  ax.set_ylabel(r'log10$\left|H-Hprev\right|$ [km]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

  # ---------------------------------
def plot_asymmetry(A,tstart,tend,fname):

  fig = plt.figure(1,figsize=(10,5))
  ax = plt.subplot(1,1,1)

  time_myr = A.flux.t[1:-1]/1e6
  Asym = A.flux.Asym[1:-1]
  Asym_end = Asym[-1]
  Asym_avg = np.average(Asym[np.where((time_myr>=tstart) & (time_myr<=tend))])

  ax.plot(time_myr, Asym, 'k', label = 'H ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(Asym_avg,4))+', H ('+str(np.around(time_myr[-1],2))+' Myr) = '+str(np.around(Asym_end,4)))
  plt.grid(True)
  # plt.ylim([0, 10])
  ax.legend()
  ax.set_xlabel('Time [Myr]')
  ax.set_ylabel('Asymmetry parameter')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_hq(A,istart,iend,jstart,jend,fname,istep,dim):
  
  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))

  fig = plt.figure(1,figsize=(10,8))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

#  cmap1 ='ocean_r'
#  cmap1 = 'terrain_r' #
#  cmap1 = 'tab20c'
  cmap1 = plt.cm.get_cmap('twilight', 20)
  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  # im.set_clim(0,0.002)
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

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
  nind = 4*nind_factor
  if (istep>0):
    Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, Vfx*scalv, Vfz*scalv,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4,clip_on=False)
  ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi)
  # ax.set_title(A.lbl.nd.phi+' tstep = '+str(istep)+' time = '+str(t)+' [yr]')
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_vz_hq(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(10,8))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

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
  nind = 4*nind_factor

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

#  cmap1 = 'viridis'
  # cmap1 = 'binary'
  cmap1 = 'RdBu'

  # Vertical solid velocity
  ax = plt.subplot(1,1,1)
  im = ax.imshow(A.Vsz[jstart:jend+1,istart:iend  ]*scalv,extent=extentVz,cmap=cmap1,origin='lower')
  # im.set_clim(0.0,6.0)
  im.set_clim(-6.0,6.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zv[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl = get_label(A,'vsz',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_rho_hq(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  fig = plt.figure(1,figsize=(10,8))
  ax = plt.subplot(1,1,1)

  scal = get_scaling(A,'rho',dim,0)
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  rho0 = 3000

  lbl  = get_label(A,'rho',dim)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'terrain'

  if (iplot==1):
    X = A.matProp.rhof
  elif (iplot==2):
    X = A.matProp.rhos
  else:
    X = A.matProp.rho

  im = ax.imshow(X[jstart:jend+1,istart:iend  ]*scal-rho0,extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-100.0,0.0)
#  im.set_clim(-100.0,100.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = r'$\Delta\rho_f$ [kg/m3]'
  elif (iplot==2):
    lbl = r'$\Delta\rho_s$ [kg/m3]'
  else:
    lbl = r'$\Delta\rho$ [kg/m3]'
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_comp_hq(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  fig = plt.figure(1,figsize=(10,8))
  ax = plt.subplot(1,1,1)

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'gist_earth_r'

  if (iplot==1):
    X = scale_TC(A,'Enth.Cf','C',dim,0)
  elif (iplot==2):
    X = scale_TC(A,'Enth.Cs','C',dim,0)
  else:
    X = scale_TC(A,'Enth.C','C',dim,0)

  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  if (iplot==1):
    im.set_clim(0.75,0.78)
  elif (iplot==2):
    im.set_clim(0.85,0.88)
  else:
    im.set_clim(0.85,0.88)

  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = get_label(A,'Cf',dim)
  elif (iplot==2):
    lbl = get_label(A,'Cs',dim)
  else:
    lbl = get_label(A,'C',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_gamma_hq(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(10,8))
  ax = plt.subplot(1,1,1)

  scal = get_scaling(A,'Gamma',dim,1)
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

#  cmap1 = 'gist_earth_r'
#  cmap1 = plt.cm.get_cmap('twilight', 20)
  cmap1 = plt.cm.get_cmap('binary', 20)

  X = A.matProp.Gamma*scal
  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0,2.5)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl  = get_label(A,'Gamma',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_visc_hq(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  fig = plt.figure(1,figsize=(10,8))
  ax = plt.subplot(1,1,1)

  scal = get_scaling(A,'eta',dim,0)
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'CMRmap_r'
#  cmap1 = 'viridis'

  if (iplot==0):
    X = np.log10(A.matProp.eta*scal)
  else:
    X = np.log10(A.matProp.zeta*scal)

  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(15,25)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=4, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==0):
    lbl  = get_label(A,'eta',dim)
  else:
    lbl  = get_label(A,'zeta',dim)
  ax.set_title('log10 '+lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_crustal_thickness(A,tstart,tend,fname):

  fig = plt.figure(1,figsize=(4,1.5))
  ax = plt.subplot(1,1,1)
  plt.grid(linestyle = ':', linewidth = 0.5)

  time_myr = A.flux.t[1:-1]/1e6
  crust_thick = A.flux.h[1:-1]/1000
  crust_avg = np.average(crust_thick[np.where((time_myr>=tstart) & (time_myr<=tend))])

  ax.plot(time_myr, crust_thick, 'k', label = 'H ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(crust_avg,4))+' km')
  ax.plot(time_myr[-1], crust_thick[-1], 'ro')
  plt.grid(True)
  plt.ylim([0, 20])
  ax.legend()
  ax.set_xlabel('Time [Myr]')
  ax.set_ylabel('Crustal thickness [km]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_porosity(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(6,4))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  cmap1 = plt.cm.get_cmap('twilight', 20)
  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
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
  nind = 4*nind_factor
  # if (istep>0):
    # Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
      # color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_vz(A,istart,iend,jstart,jend,fname,istep,dim):

  fig = plt.figure(1,figsize=(6,4))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

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
  nind = 4*nind_factor

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0

  cmap1 = 'RdBu'
  divnorm=colors.TwoSlopeNorm(vmin=-2.0, vcenter=0., vmax=8)

  # Vertical solid velocity
  ax = plt.subplot(1,1,1)
  im = ax.imshow(A.Vsz[jstart:jend+1,istart:iend  ]*scalv,extent=extentVz,cmap=cmap1,origin='lower', norm=divnorm)
  # im.set_clim(-6.0,6.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(r'$V_s^z$ [cm/yr]')

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zv[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl = get_label(A,'vsz',dim)
  # ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_comp(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  fig = plt.figure(1,figsize=(6,4))
  ax = plt.subplot(1,1,1)

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'gist_earth_r'

  if (iplot==1):
    X = scale_TC(A,'Enth.Cf','C',dim,0)
  elif (iplot==2):
    X = scale_TC(A,'Enth.Cs','C',dim,0)
  else:
    X = scale_TC(A,'Enth.C','C',dim,0)

  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  if (iplot==1):
    im.set_clim(0.75,0.78)
  elif (iplot==2):
    im.set_clim(0.85,0.88)
  else:
    im.set_clim(0.85,0.88)

  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(r'$C$ [wt. frac.]')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = get_label(A,'Cf',dim)
  elif (iplot==2):
    lbl = get_label(A,'Cs',dim)
  else:
    lbl = get_label(A,'C',dim)
  # ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_rho(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  fig = plt.figure(1,figsize=(6,4))
  ax = plt.subplot(1,1,1)

  scal = get_scaling(A,'rho',dim,0)
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  rho0 = 3000

  lbl  = get_label(A,'rho',dim)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

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
  nind = 4*nind_factor
  if (istep>0):
    Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='k', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'terrain'

  if (iplot==1):
    X = A.matProp.rhof
  elif (iplot==2):
    X = A.matProp.rhos
  else:
    X = A.matProp.rho

  im = ax.imshow(X[jstart:jend+1,istart:iend  ]*scal-rho0,extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-100.0,0.0)
#  im.set_clim(-100.0,100.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(r'$\rho-\rho_0$ [kg/m$^3$]')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = r'$\Delta\rho_f$ [kg/m3]'
  elif (iplot==2):
    lbl = r'$\Delta\rho_s$ [kg/m3]'
  else:
    lbl = r'$\Delta\rho$ [kg/m3]'
  # ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_crustal_thickness(A,tstart,tend,fname):

  fig = plt.figure(1,figsize=(4,1.5))
  ax = plt.subplot(1,1,1)
  plt.grid(linestyle = ':', linewidth = 0.5)

  time_myr = A.flux.t[1:-1]/1e6
  crust_thick = A.flux.h[1:-1]/1000
  crust_avg = np.average(crust_thick[np.where((time_myr>=tstart) & (time_myr<=tend))])

  ax.plot(time_myr, crust_thick, 'k', label = 'H ('+str(tstart)+'-'+str(tend)+' Myr) = '+str(np.around(crust_avg,4))+' km')
  ax.plot(time_myr[-1], crust_thick[-1], 'ro')
  plt.grid(True)
  plt.ylim([0, 20])
  ax.legend()
  ax.set_xlabel('Time [Myr]')
  ax.set_ylabel('Crustal thickness [km]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_porosity_scaled_domain(A,istart,iend,jstart,jend,fname,istep,dim):

  xsize = 6*((iend-istart)/200)
  zsize = 4*((jend-jstart)/100)
  fig = plt.figure(1,figsize=(xsize,zsize))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  cmap1 = plt.cm.get_cmap('twilight', 20)
  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(r'log$_{10}\phi$')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  xa2 = A.grid.xc[istart:iend]*scalx
  stream_points2 = []
  for xi in xa2:
    stream_points2.append([xi,A.grid.zc[int(jend/2)]*scalx])

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)
  nind_factor = int(A.nz/100)
  nind = 4*nind_factor
  # if (istep>0):
  #   Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
  #     color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vfcx[jstart:jend  ,istart:iend  ]*scalv, A.Vfcz[jstart:jend  ,istart:iend  ]*scalv,
        color='orange',linewidth=0.3, start_points=stream_points2, density=20.0, minlength=0.5, arrowstyle='-')
  stream.lines.set_linestyle(':')

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  if (A.grid.xv[istart]==0.0):
    ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_reference_comp_scaled_domain(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  xsize = 6*((iend-istart)/200)
  zsize = 4*((jend-jstart)/100)
  fig = plt.figure(1,figsize=(xsize,zsize))
  ax = plt.subplot(1,1,1)

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'gist_earth_r'

  if (iplot==1):
    X = scale_TC(A,'Enth.Cf','C',dim,0)
  elif (iplot==2):
    X = scale_TC(A,'Enth.Cs','C',dim,0)
  else:
    X = scale_TC(A,'Enth.C','C',dim,0)

  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  if (iplot==1):
    im.set_clim(0.75,0.78)
  elif (iplot==2):
    im.set_clim(0.85,0.88)
  else:
    im.set_clim(0.85,0.88)

  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(r'$C$ [wt. frac.]')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  # solid streamlines
  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])
  
  xa = A.grid.zc[jstart:jend:4]*scalx
  stream_points2 = []
  for xi in xa:
    stream_points2.append([xi,A.grid.zc[jstart]*scalx])

  if (A.grid.xv[istart]==0.0):
    stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.4, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  else:
    stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.4, density=0.5, minlength=0.5, arrowstyle='-')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='none', markersize=6, clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = get_label(A,'Cf',dim)
  elif (iplot==2):
    lbl = get_label(A,'Cs',dim)
  else:
    lbl = get_label(A,'C',dim)
  # ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')x

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})

  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  # transform zero porosity
  A.Enth.phi[A.Enth.phi==0] = 1e-10

  # 1. porosity
  ax = plt.subplot(1,2,1)
  # cmap1='ocean_r'
  # cmap1 = plt.cm.get_cmap('twilight', 20)
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  # im.set_clim(0,0.002)
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)
  cbar.ax.set_title(r'log$_{10}\phi$')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, A.Vscx*scalv, A.Vscz*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)

  nind_factor = int(A.nz/100)
  nind = 4*nind_factor
  # if (istep>0):
  #   Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
  #     color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, Vfx*scalv, Vfz*scalv,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  # nind1 = 4*nind_factor
  # nind2 = 8*nind_factor
  # #normalize vectors
  # r = np.power(np.add(np.power(A.Vscx,2), np.power(A.Vscz,2)),0.5)
  # Vscx = A.Vscx/r
  # Vscz = A.Vscz/r
  # Q  = ax.quiver(A.grid.xc[istart:iend:nind1]*scalx, A.grid.zc[jstart:jend:nind2]*scalx, Vscx[jstart:jend:nind2,istart:iend:nind1]*scalv, Vscz[jstart:jend:nind2,istart:iend:nind1]*scalv, 
  #     color='grey', units='width', pivot='mid', width=0.0008, headaxislength=3, minlength=0)

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=14)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi)
  # ax.set_title(A.lbl.nd.phi+' tstep = '+str(istep)+' time = '+str(t)+' [yr]')
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_full_ridge_white(A,istart,iend,jstart,jend,fname,istep,dim):
  
  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  cmap1 = plt.cm.get_cmap('twilight', 20)
  ax = plt.subplot(1,2,1)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  # im.set_clim(0,0.002)
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)
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
  nind = 4*nind_factor
  # if (istep>0):
  #   Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
  #     color='orange', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*scalx,A.grid.zc*scalx, Vfx*scalv, Vfz*scalv,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  nind = 6*nind_factor
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='silver', units='width', pivot='mid', width=0.0008, headaxislength=3, minlength=0)

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=14)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title(A.lbl.nd.phi)
  # ax.set_title(A.lbl.nd.phi+' tstep = '+str(istep)+' time = '+str(t)+' [yr]')
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_vorticity_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})

  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentN =[min(A.grid.xv[istart:iend  ])*scalx, max(A.grid.xv[istart:iend  ])*scalx, min(A.grid.zv[jstart:jend  ])*scalx, max(A.grid.zv[jstart:jend  ])*scalx]

  ax = plt.subplot(1,2,1)
  cmap1='RdBu_r'
  
  dx = A.grid.xv[1]-A.grid.xv[0]
  dz = A.grid.zv[1]-A.grid.zv[0]
  dVzdx = (A.Vsz[jstart:jend+1,istart:iend-1]-A.Vsz[jstart:jend+1,istart+1:iend])/dx
  dVxdz = (A.Vsx[jstart:jend-1,istart:iend+1]-A.Vsx[jstart+1:jend,istart:iend+1])/dz

  Vorticity = (dVzdx[1:-1,:]-dVxdz[:,1:-1])
  im = ax.imshow(Vorticity,extent=extentN,cmap=cmap1,origin='lower')
  im.set_clim(-0.0001,0.0001)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)
  cbar.ax.set_title(r'$\omega$')

  # # vorticity contours
  # levels = [-0.0002, -0.0001, 0, 0.0001, 0.0002,]
  # om = ax.contour(A.grid.xv[istart:iend-1]*scalx, A.grid.zv[jstart:jend-1]*scalx, Vorticity, levels=levels,linewidths=(0.8,), extend='both')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=14)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.5,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\omega$ time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_compare_b100_b120(A,A2,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})

  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  istart = 0
  iend   = 200
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]
  
  istart2 = 0
  iend2   = 200
  extentE2 =[min(A2.grid.xc[istart2:iend2  ])*scalx, max(A2.grid.xc[istart2:iend2  ])*scalx, min(A2.grid.zc[jstart:jend  ])*scalx, max(A2.grid.zc[jstart:jend  ])*scalx]

  # transform zero porosity
  A.Enth.phi[A.Enth.phi==0] = 1e-10
  A2.Enth.phi[A2.Enth.phi==0] = 1e-10

  # 1. porosity
  ax = plt.subplot(1,2,1)
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)
  cbar.ax.set_title(r'log$_{10}\phi$')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=14)

  # plot right
  cmap2 = plt.cm.get_cmap('inferno', 20)
  im2 = ax.imshow(np.log10(A2.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE2,cmap=cmap2,origin='lower')
  im2.set_clim(-4,-1)
  xa = A2.grid.xc[istart2:iend2:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A2.grid.zc[jstart]*scalx])

  # solid streamlines
  stream2 = ax.streamplot(A2.grid.xc[istart2:iend2  ]*scalx,A2.grid.zc[jstart:jend  ]*scalx, A2.Vscx[jstart:jend  ,istart2:iend2  ]*scalv, A2.Vscz[jstart:jend  ,istart2:iend2  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  T2 = scale_TC(A2,'Enth.T','T',dim,1)
  ts2 = ax.contour(A2.grid.xc[istart:iend  ]*scalx, A2.grid.zc[jstart:jend  ]*scalx, T2[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts2, fmt=fmt, fontsize=14)

  t = A.nd.t*scalt
  t2 = A2.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  xdist2 = A.nd.U0*scalv*t2*1.0e-5

  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0

  zdist = A.grid.zc[-1]*scalx
  ax.plot(xdist2,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  ax.text(-220, 3.0, r'$U_0$ = 4 cm/yr', fontsize=20, weight='bold') 
  ax.text(-125.0, 3.0, 'A. Passive flow', fontsize=20)  
  ax.text(50.0, 3.0, r'B. Active flow ($\phi$-C buoyancy)', fontsize=20)  

  ax.text(-195.0, -95, str(np.around(t/1.0e6,1))+' Myr', fontsize=24,color='white') 
  ax.text(165.0, -95, str(np.around(t2/1.0e6,1))+' Myr', fontsize=24,color='white') 

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_vz_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

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
  nind = 4*nind_factor

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'RdBu'

  # Vertical solid velocity
  ax = plt.subplot(1,2,1)
  im = ax.imshow(A.Vsz[jstart:jend+1,istart:iend  ]*scalv,extent=extentVz,cmap=cmap1,origin='lower')
  # im.set_clim(0.0,6.0)
  im.set_clim(-6.0,6.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zv[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl = get_label(A,'vsz',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_rho_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))  #14,5
  ax = plt.subplot(1,2,1)

  scal = get_scaling(A,'rho',dim,0)
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  rho0 = 3000

  lbl  = get_label(A,'rho',dim)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  if (iplot==1):
    # cmap1 = 'terrain_r'
    cmap1 = plt.cm.get_cmap('terrain_r', 40)
    # only where phi non-zero
    mask = np.zeros(A.matProp.rhof.shape, dtype=bool)
    mask[np.where(A.Enth.phi<1e-10)] = True
    X = np.ma.array(A.matProp.rhof, mask=mask)
    # X = A.matProp.rhof
    rho0 = 2500
  elif (iplot==2):
    cmap1 = 'terrain'
    X = A.matProp.rhos
  else:
    cmap1 = 'terrain'
    X = A.matProp.rho

  im = ax.imshow(X[jstart:jend+1,istart:iend  ]*scal-rho0,extent=extentE,cmap=cmap1,origin='lower')
  if (iplot==1):
    im.set_clim(0.0,500.0)
  elif (iplot==2):
    im.set_clim(-100.0,0.0)
  else:
    im.set_clim(-125.0,0.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])
  
  # solid streamlines
  # stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = r'$\Delta\rho_f$ [kg/m3]'
  elif (iplot==2):
    lbl = r'$\Delta\rho_s$ [kg/m3]'
  else:
    lbl = r'$\Delta\rho$ [kg/m3]'
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_comp_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim,iplot):

  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))  #14,5
  ax = plt.subplot(1,2,1)

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # MOR marker [km]
  t = A.nd.t*scalt
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'gist_earth_r'

  if (iplot==1):
    X = scale_TC(A,'Enth.Cf','C',dim,0)
    mask = np.zeros(X.shape, dtype=bool)
    mask[np.where(A.Enth.phi<1e-10)] = True
    X = np.ma.array(X, mask=mask)
  elif (iplot==2):
    X = scale_TC(A,'Enth.Cs','C',dim,0)
  else:
    X = scale_TC(A,'Enth.C','C',dim,0)

  im = ax.imshow(X[jstart:jend+1,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  if (iplot==1):
    im.set_clim(0.75,0.78)
  elif (iplot==2):
    im.set_clim(0.85,0.88)
  else:
    im.set_clim(0.85,0.88)

  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  if (iplot==1):
    lbl = get_label(A,'Cf',dim)
  elif (iplot==2):
    lbl = get_label(A,'Cs',dim)
  else:
    lbl = get_label(A,'C',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_vx_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentVx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend+1])*scalx, max(A.grid.zc[jstart:jend+1])*scalx]
  # extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

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
  nind = 4*nind_factor

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  cmap1 = 'RdBu'

  # Vertical solid velocity
  ax = plt.subplot(1,2,1)
  im = ax.imshow(A.Vsx[jstart:jend+1,istart:iend  ]*scalv,extent=extentVx,cmap=cmap1,origin='lower')
  # im.set_clim(0.0,6.0)
  im.set_clim(-10.0,10.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zv[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  lbl = get_label(A,'vsx',dim)
  ax.set_title(lbl+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_temperature_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 16})
  fig = plt.figure(1,figsize=(50,10))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  xa = A.grid.xc[istart:iend:4]*scalx
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[jstart]*scalx])

  t = A.nd.t*scalt
  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else:
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0
    # xdist = A.grid.xc[0]*scalx

  T0 = 273.15
  Tp = 1648
  A_nd  = 2.450000000000e-02
  zmin_nd  = -1.0
  Tm  = (Tp-T0)*np.exp(-A_nd*zmin_nd)+T0
  kappa = 1.0e-6
  xmor = 4.0

  cmap1 = 'RdBu_r'
  T = scale_TC(A,'Enth.T','T',dim,1)

  T_HS = np.zeros_like(T)
  for i in range(0,len(A.grid.xc)):
    for j in range(0,len(A.grid.zc)):
      ix = A.grid.xc[i]*scalx
      iz = A.grid.zc[j]*scalx
      age = np.abs(ix-xmor)*1.0e3/(A.nd.U0*scalv*1.0e-2)*A.scal.SEC_YEAR
      if (age<=0.0):
        T_HS[j,i] = Tm-T0
      else:
        T_HS[j,i] = np.abs((Tm-T0)*math.erf(iz*1.0e3/(2.0*np.sqrt(kappa*age))))

  X = T-T_HS
  # Temperature deviations from the halfspace
  ax = plt.subplot(1,2,1)
  im = ax.imshow(X[jstart:jend,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-80.0,80.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)
  cbar.ax.set_title(r'$\Delta T$ (K)')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.Enth.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)
  nind_factor = int(A.nz/100)
  nind = 4*nind_factor
  if (istep>0):
    Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, Vfx[jstart:jend:nind,istart:iend:nind]*scalv, Vfz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='k', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('time = '+str(round(t/1.0e3,0))+' [kyr]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_fields_full_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 20})

  fig = plt.figure(1,figsize=(50,25))  #14,5
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  # 1. POROSITY
  ax = plt.subplot(4,1,1)
  # transform zero porosity
  cmap1 = plt.cm.get_cmap('inferno', 20)
  A.Enth.phi[A.Enth.phi==0] = 1e-10
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.90)
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
  nind = 4*nind_factor
  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=5.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  # 2. Temperature
  ax = plt.subplot(4,1,2)
  lblT = get_label(A,'T',dim)
  cmap1 = plt.cm.get_cmap('RdBu_r')
  im = ax.imshow(T[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0,1400)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  
  # plot solid velocity field
  nind = 10
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='k', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # temperature contour
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)
  cbar = fig.colorbar(im,ax=ax, shrink=0.90)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblT)

  # 3. Bulk composition
  ax = plt.subplot(4,1,3)
  lblC = get_label(A,'C',dim)
  cmap1 = 'gist_earth_r'
  X = scale_TC(A,'Enth.C','C',dim,0)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0.85,0.88)
  cbar = fig.colorbar(im,ax=ax, shrink=0.90)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblC)

  # 4. Density
  ax = plt.subplot(4,1,4)
  cmap1 = 'terrain'
  lbl  = get_label(A,'rho',dim)
  scal_rho = get_scaling(A,'rho',dim,0)
  rho0 = 3000
  if (istep>0):
    X = A.matProp.rho
  else:
    X = np.ones_like(T)*rho0/scal_rho
  im = ax.imshow(X[jstart:jend  ,istart:iend  ]*scal_rho-rho0,extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-200.0,0.0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.90)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\Delta\rho$ [kg/m3]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_fields_half_ridge(A,istart,iend,jstart,jend,fname,istep,dim):

  matplotlib.rcParams.update({'font.size': 20})
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  fig = plt.figure(1,figsize=(40,25))
  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  scalt = get_scaling(A,'t',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  # 1. POROSITY
  ax = plt.subplot(4,1,1)
  cmap1 = plt.cm.get_cmap('inferno', 20)
  A.Enth.phi[A.Enth.phi==0] = 1e-10
  im = ax.imshow(np.log10(A.Enth.phi[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-4,-1)

  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax_cb)
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
  nind = 4*nind_factor
  # solid streamlines
  stream = ax.streamplot(A.grid.xc[istart:iend  ]*scalx,A.grid.zc[jstart:jend  ]*scalx, A.Vscx[jstart:jend  ,istart:iend  ]*scalv, A.Vscz[jstart:jend  ,istart:iend  ]*scalv,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  levels = [250, 500, 750, 1000, 1200, 1300,]
  fmt = r'%0.0f $^o$C'
  T = scale_TC(A,'Enth.T','T',dim,1)
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  t = A.nd.t*scalt

  # MOR marker [km]
  xdist = A.nd.U0*scalv*t*1.0e-5
  if (A.grid.xc[0]<0.0):
    xdist_full = -xdist
  else: 
    xdist_full = xdist

  if (istep==0):
    xdist = 0.0

  zdist = A.grid.zc[-1]*scalx
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(A.lbl.nd.phi+' time = '+str(round(t/1.0e3,0))+' [kyr]')

  # 2. Temperature
  ax = plt.subplot(4,1,2)
  lblT = get_label(A,'T',dim)
  cmap1 = plt.cm.get_cmap('RdBu_r')
  im = ax.imshow(T[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0,1400)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
  
  # plot solid velocity field
  nind = 20
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv, 
      color='k', units='width', pivot='mid', width=0.001, headaxislength=3, minlength=0)

  # temperature contour
  ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, T[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=20)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax_cb)

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblT)

  # 3. Bulk composition
  ax = plt.subplot(4,1,3)
  lblC = get_label(A,'C',dim)
  cmap1 = 'gist_earth_r'
  X = scale_TC(A,'Enth.C','C',dim,0)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(0.85,0.88)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax_cb)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  # ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(lblC)

  # 4. Density
  ax = plt.subplot(4,1,4)
  cmap1 = 'terrain'
  lbl  = get_label(A,'rho',dim)
  scal_rho = get_scaling(A,'rho',dim,0)
  rho0 = 3000
  if (istep>0):
    X = A.matProp.rho
  else:
    X = np.ones_like(T)*rho0/scal_rho
  im = ax.imshow(X[jstart:jend  ,istart:iend  ]*scal_rho-rho0,extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-200.0,0.0)
  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  cbar = fig.colorbar(im,ax=ax_cb)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, A.Enth.phi[jstart:jend  ,istart:iend  ], levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  if (xdist<=A.grid.xc[iend-1]*scalx) & (xdist>=A.grid.xc[istart]*scalx):
    ax.plot(xdist,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)
    ax.plot(xdist_full,zdist,'v',color='orange',linewidth=0.1,mfc='orange', markersize=10,clip_on=False)

  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\Delta\rho$ [kg/m3]')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()