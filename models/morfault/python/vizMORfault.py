# ---------------------------------
# Load modules
# ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

from matplotlib import rc
from matplotlib import colors
import importlib
import os
import math
from cmcrameri import cm

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class EmptyStruct:
  pass

# ---------------------------------
# Definitions
# ---------------------------------
def make_dir(output_dir):
  try:
    os.mkdir(output_dir)
  except OSError:
    pass

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
def correct_path_marker_data(fname):
  try: # try to open file .xmf file
    fname_new = fname[:-4]+'_new.xmf'

    # Copy new file
    f = open(fname, 'r')
    f_new = open(fname_new, 'w')
    line_prev = ''

    for line in f:
      if '<DataItem' in line_prev:
        try:
          line2 = line[line.index('/')+1:]
          line = line2
        except:
          line = line
      f_new.write(line)
      line_prev = line

    f.close()
    f_new.close()

    # remove new file
    os.system('cp '+fname_new+' '+fname)
    os.system('rm '+fname_new)

  except OSError:
    print('Cannot open:', fname)

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

    # parse number of PV and T SNES iterations
    iPV=0
    iT =0
    iphi = 0
    PVsolver = 0
    Tsolver  = 0
    phisolver  = 0
    f = open(fname, 'r')
    for line in f:
      if '# (PV) Mechanics' in line: 
        PVsolver = 1
        Tsolver  = 0
        phisolver = 0
      if '# (T) Energy' in line: 
        PVsolver = 0
        Tsolver  = 1
        phisolver = 0
      if '# (phi) Porosity' in line: 
        PVsolver = 0
        Tsolver  = 0
        phisolver = 1
      if 'SNES Function norm' in line:
        if (PVsolver==1):
          iPV+=1
        if (Tsolver==1):
          iT+=1
        if (phisolver==1):
          iphi+=1
    f.close()

    # Convergence 
    sol = EmptyStruct()
    sol.iPV = iPV
    sol.iT  = iT
    sol.iphi  = iphi
    sol.tstep = tstep
    sol.runtime  = np.zeros(sol.tstep)
    sol.dt  = np.zeros(sol.tstep)

    sol.PV_ts_res  = np.zeros(sol.tstep)
    sol.PV_n_it  = np.zeros(sol.tstep)
    sol.PV_it_res  = np.zeros(iPV)
    sol.PV_iter = np.zeros(iPV)
    sol.PV_ts_diverged = np.zeros(sol.tstep)

    sol.T_ts_res  = np.zeros(sol.tstep)
    sol.T_n_it  = np.zeros(sol.tstep)
    sol.T_it_res  = np.zeros(iT)
    sol.T_iter = np.zeros(iT)
    sol.T_ts_diverged = np.zeros(sol.tstep)

    sol.phi_ts_res  = np.zeros(sol.tstep)
    sol.phi_n_it  = np.zeros(sol.tstep)
    sol.phi_it_res  = np.zeros(iphi)
    sol.phi_iter = np.zeros(iphi)
    sol.phi_ts_diverged = np.zeros(sol.tstep)

    # Parse output and save residual info
    i0 =-1
    iPV=0
    iT =0
    iphi=0
    PVsolver = 0
    Tsolver  = 0
    phisolver  = 0
    f = open(fname, 'r')
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
        niPV=0
        niT =0
        niphi=0
      if '# Timestep runtime:' in line:
        ij = line.find('(')
        sol.runtime[i0]  = float(line[20:ij-1])
      if '# TIME:' in line:
        sol.dt[i0] = float(line[44:62])
      if '# (PV) Mechanics' in line: 
        PVsolver = 1
        Tsolver  = 0
        phisolver = 0
      if '# (T) Energy' in line: 
        PVsolver = 0
        Tsolver  = 1
        phisolver = 0
      if '# (phi) Porosity' in line: 
        PVsolver = 0
        Tsolver  = 0
        phisolver = 1
      if 'Nonlinear pv_ solve did not converge' in line:
        sol.PV_ts_diverged[i0] = i0
      if 'Nonlinear t_ solve did not converge' in line:
        sol.T_ts_diverged[i0] = i0
      if 'Nonlinear phi_ solve did not converge' in line:
        sol.phi_ts_diverged[i0] = i0
      if 'SNES Function norm' in line:
        if (PVsolver==1):
          sol.PV_ts_res[i0]  = float(line[23:41])
          sol.PV_it_res[iPV]  = float(line[23:41])
          sol.PV_iter[iPV] = float(line[0:4])
          iPV+=1
          niPV+=1
          sol.PV_n_it[i0]  = niPV
        if (Tsolver==1):
          sol.T_ts_res[i0]  = float(line[23:41])
          sol.T_it_res[iT]  = float(line[23:41])
          sol.T_iter[iT] = float(line[0:4])
          iT+=1
          niT+=1
          sol.T_n_it[i0]   = niT
        if (phisolver==1):
          sol.phi_ts_res[i0]  = float(line[23:41])
          sol.phi_it_res[iphi]  = float(line[23:41])
          sol.phi_iter[iphi] = float(line[0:4])
          iphi+=1
          niphi+=1
          sol.phi_n_it[i0] = niphi

    f.close()

    return sol
  except OSError:
    print('Cannot open:', fname)
    return 0

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
    scal.T_KELVIN = 273.15
    scal.g = 9.8

    scal.x = data['scalx'][0]
    scal.eta = data['scaleta'][0]
    scal.rho = data['scalrho'][0]
    scal.v = data['scalv'][0]
    scal.t = data['scalt'][0]
    scal.DT = data['scalDT'][0]
    scal.P = data['scaltau'][0]
    scal.kappa = data['scalkappa'][0]
    scal.kT = data['scalkT'][0]
    scal.Kphi = data['scalkphi'][0]
    scal.Gamma = data['scalGamma'][0]
    scal.T0 = scal.T_KELVIN
    scal.eps = scal.v/scal.x

    nd = EmptyStruct()
    nd.L = data['L'][0]
    nd.H = data['H'][0]
    nd.xmin = data['xmin'][0]
    nd.zmin = data['zmin'][0]
    nd.Hs = data['Hs'][0]
    nd.Vext = data['Vext'][0]
    nd.Vin = data['Vin'][0]
    nd.Tbot = data['Tbot'][0]
    nd.Ttop = data['Ttop'][0]
    nd.eta_min = data['eta_min'][0]
    nd.eta_max = data['eta_max'][0]
    nd.eta_K = data['eta_K'][0]
    nd.istep = data['istep'][0]
    nd.t = data['t'][0]
    nd.dt = data['dt'][0]
    nd.tmax = data['tmax'][0]
    nd.dtmax = data['dtmax'][0]

    nd.delta = data['delta'][0]
    nd.R = data['R'][0]
    nd.Ra = data['Ra'][0]
    nd.Gamma = data['Gamma'][0]

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
    lbl.nd.vs= r'$V_s$ [-]'
    lbl.nd.vsx= r'$V_s^x$ [-]'
    lbl.nd.vsz= r'$V_s^z$ [-]'
    lbl.nd.vf= r'$V_f$ [-]'
    lbl.nd.vfx= r'$V_f^x$ [-]'
    lbl.nd.vfz= r'$V_f^z$ [-]'
    lbl.nd.v = r'$V$ [-]'
    lbl.nd.vx= r'$V^x$ [-]'
    lbl.nd.vz= r'$V^z$ [-]'
    lbl.nd.epsxx=r'$\dot{\epsilon}_{xx}$ [-]'
    lbl.nd.epszz=r'$\dot{\epsilon}_{zz}$ [-]'
    lbl.nd.epsxz=r'$\dot{\epsilon}_{xz}$ [-]'
    lbl.nd.epsII=r'$\dot{\epsilon}_{II}$ [-]'
    lbl.nd.tauxx=r'$\tau_{xx}$ [-]'
    lbl.nd.tauzz=r'$\tau_{zz}$ [-]'
    lbl.nd.tauxz=r'$\tau_{xz}$ [-]'
    lbl.nd.tauII=r'$\tau_{II}$ [-]'
    lbl.nd.C = r'$C$ [-]'
    lbl.nd.sigmat = r'$\sigma_T$ [-]'
    lbl.nd.theta = r'$\theta$ [-]'
    lbl.nd.T = r'$\tilde{\theta}$ [-]'
    lbl.nd.Plith = r'$P_{lith}$ [-]'
    lbl.nd.resP = r'res $P$ [-]'
    lbl.nd.resvsx= r'res $V_s^x$ [-]'
    lbl.nd.resvsz= r'res $V_s^z$ [-]'
    lbl.nd.eta = r'$\eta$ [-]'
    lbl.nd.zeta = r'$\zeta$ [-]'
    lbl.nd.Kphi = r'$K_\phi$ [-]'
    lbl.nd.rho = r'$\rho$ [-]'
    lbl.nd.rhof = r'$\rho_f$ [-]'
    lbl.nd.rhos = r'$\rho_s$ [-]'
    lbl.nd.Z = r'$Z$ [-]'
    lbl.nd.G = r'$G$ [-]'
    lbl.nd.Gamma = r'$\Gamma$ [-]'
    lbl.nd.divmass = r'$\nabla\cdot(v)$ [-]'

    # Units and labels - dimensional
    lbl.dim.P = r'$P$ [MPa]'
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
    lbl.dim.epsxx=r'$\dot{\epsilon}_{xx}$ [1/s]'
    lbl.dim.epszz=r'$\dot{\epsilon}_{zz}$ [1/s]'
    lbl.dim.epsxz=r'$\dot{\epsilon}_{xz}$ [1/s]'
    lbl.dim.epsII=r'$\dot{\epsilon}_{II}$ [1/s]'
    lbl.dim.tauxx=r'$\tau_{xx}$ [MPa]'
    lbl.dim.tauzz=r'$\tau_{zz}$ [MPa]'
    lbl.dim.tauxz=r'$\tau_{xz}$ [MPa]'
    lbl.dim.tauII=r'$\tau_{II}$ [MPa]'
    lbl.dim.T = r'$T$ $[^oC]$'
    lbl.dim.Plith = r'$P_{lith}$ [MPa]'
    lbl.dim.resP = r'res $P$ [MPa]'
    lbl.dim.resvsx= r'res $V_s^x$ [cm/yr]'
    lbl.dim.resvsz= r'res $V_s^z$ [cm/yr]'
    lbl.dim.eta = r'$\eta$ [Pa.s]'
    lbl.dim.zeta = r'$\zeta$ [Pa.s]'
    lbl.dim.Kphi = r'$K_\phi$ [m2]'
    lbl.dim.rho = r'$\rho$ [kg/m3]'
    lbl.dim.rhof = r'$\rho_f$ [kg/m3]'
    lbl.dim.rhos = r'$\rho_s$ [kg/m3]'
    lbl.dim.Z = r'$Z$ [MPa]'
    lbl.dim.G = r'$G$ [MPa]'
    lbl.dim.C = r'$C$ [MPa]'
    lbl.dim.sigmat = r'$\sigma_T$ [MPa]'
    lbl.dim.theta = r'$\theta$ [-]'
    lbl.dim.Gamma = r'$\Gamma$ [g/m$^3$/yr]'
    lbl.dim.divmass = r'$\nabla\cdot v$ [/s]'

    return lbl
  except OSError:
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
def parse_marker_file(fname,fdir):
  try: 
    mark = EmptyStruct()
    dim  = np.zeros(3)
    seek = np.zeros(3)
    # print(fdir)
    # print(fname)

    # load info from xmf file first
    f = open(fdir+'/'+fname, 'r')
    line_prev = ''
    for line in f:
      if '<Topology Dimensions' in line:
        iss = line.index('"')
        ise = line.index('"',iss+1,-1)
        n = int(line[iss+1:ise])
      if '.pbin' in line:
        fdata = line[:-1]
      
      if '<DataItem Format="Binary" Endian="Big" DataType="Int" Dimensions' in line:
        iss = line.index('"')
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        dim[0] = int(line[iss+1:ise])
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        seek[0] = int(line[iss+1:ise])
      if '<Geometry Type="XY">' in line_prev:
        iss = line.index('"')
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        dim[1] = int(line[iss+1:ise-2])
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        seek[1] = int(line[iss+1:ise])
      
      if '<Attribute Center' in line_prev:
        iss = line.index('"')
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        dim[2] = int(line[iss+1:ise])
        
        iss = line.index('"',ise+1,-1)
        ise = line.index('"',iss+1,-1)
        seek[2] = int(line[iss+1:ise])
      
      line_prev = line
    f.close()

    mark.n = n
    # load binary data
    dtype0 = '>i'
    dtype1 = '>f8' # float, precision 8

    mark.x = np.zeros(n)
    mark.z = np.zeros(n)
    mark.id = np.zeros(n)

    # load binary data
    # print(fdir+'/'+fdata)
    with open(fdir+'/'+fdata, "rb") as f:
      for i in range(0,3*n):
        topo = np.fromfile(f,np.dtype(dtype0),count=1)
      for i in range(0,n):
        mark.x[i] = np.fromfile(f,np.dtype(dtype1),count=1)
        mark.z[i] = np.fromfile(f,np.dtype(dtype1),count=1)
      for i in range(0,n):
        mark.id[i] = np.fromfile(f,np.dtype(dtype1),count=1)

    # print(mark.n)
    return mark
  except OSError:
    print('Cannot open: '+fdir+'/'+fname)
    return 0.0

# ---------------------------------
def get_label(A,lbl_i,dim):
  if (dim):
    exec("global lbl_out; lbl_out = A.lbl.dim.%s" % (lbl_i))
  else:
    exec("global lbl_out; lbl_out = A.lbl.nd.%s" % (lbl_i))
  global lbl_out
  return lbl_out

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
def parse_Element_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    Tr = data['X_cell']

    # Reshape data in 2D
    T = Tr.reshape(nz,nx)

    return T
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_MPhase_file(fname,fdir):
  try: 
    MPhase = EmptyStruct()
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    data_v  = data['X_vertex']
    data_c  = data['X_cell']
    data_fx = data['X_face_x']
    data_fz = data['X_face_y']
    dof = 6

    # Reshape data in 2D
    MPhase.CornerPh0 = data_v[0::dof].reshape(nz+1,nx+1)
    MPhase.CornerPh1 = data_v[1::dof].reshape(nz+1,nx+1)
    MPhase.CornerPh2 = data_v[2::dof].reshape(nz+1,nx+1)
    MPhase.CornerPh3 = data_v[3::dof].reshape(nz+1,nx+1)
    MPhase.CornerPh4 = data_v[4::dof].reshape(nz+1,nx+1)
    MPhase.CornerPh5 = data_v[5::dof].reshape(nz+1,nx+1)
    
    # For now no need to plot all fields in all dofs

    return MPhase
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_PV_file(fname,fdir):
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

    # Reshape data in 2D
    P  = pr.reshape(nz,nx)
    Vx = vxr.reshape(nz,nx+1)
    Vz = vzr.reshape(nz+1,nx)

    return P,Vx,Vz
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_Tensor_file(fname,fdir):
  try: 
    Tensor = EmptyStruct()
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    data_v  = data['X_vertex']
    data_c  = data['X_cell']
    dof = 4

    # Reshape data in 2D
    Tensor.xx_corner = data_v[0::dof].reshape(nz+1,nx+1)
    Tensor.zz_corner = data_v[1::dof].reshape(nz+1,nx+1)
    Tensor.xz_corner = data_v[2::dof].reshape(nz+1,nx+1)
    Tensor.II_corner = data_v[3::dof].reshape(nz+1,nx+1)

    Tensor.xx_center = data_c[0::dof].reshape(nz,nx)
    Tensor.zz_center = data_c[1::dof].reshape(nz,nx)
    Tensor.xz_center = data_c[2::dof].reshape(nz,nx)
    Tensor.II_center = data_c[3::dof].reshape(nz,nx)

    return Tensor
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_PVcoeff_file(fname,fdir):
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
    dof = 3
    Cr = data_c[0::dof]
    Ar = data_c[1::dof]
    D1r= data_c[2::dof]

    Bxr = data_fx[0::dof]
    D2xr= data_fx[1::dof]
    D3xr= data_fx[2::dof]

    Bzr = data_fz[0::dof]
    D2zr= data_fz[1::dof]
    D3zr= data_fz[2::dof]

    # Reshape data in 2D
    coeff = EmptyStruct()
    coeff.A_corner = data_v.reshape(nz+1,nx+1)

    coeff.C = Cr.reshape(nz,nx)
    coeff.A = Ar.reshape(nz,nx)
    coeff.D1= D1r.reshape(nz,nx)

    coeff.Bx  = Bxr.reshape(nz,nx+1)
    coeff.D2x = D2xr.reshape(nz,nx+1)
    coeff.D3x = D3xr.reshape(nz,nx+1)

    coeff.Bz  = Bzr.reshape(nz+1,nx)
    coeff.D2z = D2zr.reshape(nz+1,nx)
    coeff.D3z = D3zr.reshape(nz+1,nx)

    return coeff
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_PVcoeff_Stokes_file(fname,fdir):
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
    dof = 2
    Cr = data_c[0::dof]
    Ar = data_c[1::dof]

    dof = 1
    Bxr = data_fx[0::dof]
    Bzr = data_fz[0::dof]

    # Reshape data in 2D
    coeff = EmptyStruct()
    coeff.A_corner = data_v.reshape(nz+1,nx+1)

    coeff.C = Cr.reshape(nz,nx)
    coeff.A = Ar.reshape(nz,nx)
    coeff.Bx  = Bxr.reshape(nz,nx+1)
    coeff.Bz  = Bzr.reshape(nz+1,nx)

    return coeff
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def parse_matProp_file(fname,fdir):
  try: 
    # Load output data including directory or path
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    data_c = data['X_cell']
 
    matProp = EmptyStruct()
    dof = 17
    matProp.eta = data_c[0::dof].reshape(nz,nx)
    matProp.etaV = data_c[1::dof].reshape(nz,nx)
    matProp.etaE = data_c[2::dof].reshape(nz,nx)
    matProp.etaP = data_c[3::dof].reshape(nz,nx)
    matProp.zeta = data_c[4::dof].reshape(nz,nx)
    matProp.zetaV = data_c[5::dof].reshape(nz,nx)
    matProp.zetaE = data_c[6::dof].reshape(nz,nx)
    matProp.DPdl = data_c[7::dof].reshape(nz,nx)
    matProp.Z = data_c[8::dof].reshape(nz,nx)
    matProp.G = data_c[9::dof].reshape(nz,nx)
    matProp.C = data_c[10::dof].reshape(nz,nx)
    matProp.sigmat = data_c[11::dof].reshape(nz,nx)
    matProp.theta = data_c[12::dof].reshape(nz,nx)
    matProp.rho = data_c[13::dof].reshape(nz,nx)
    matProp.Kphi = data_c[14::dof].reshape(nz,nx)

    return matProp
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
def parse_Tcoeff_file(fname,fdir):
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
    dof = 2
    Ar = data_c[0::dof]
    Cr = data_c[1::dof]
    
    Bxr = data_fx[0::dof]
    uxr = data_fx[1::dof]

    Bzr = data_fz[0::dof]
    uzr = data_fz[1::dof]

    # Reshape data in 2D
    coeff = EmptyStruct()
    coeff.A = Ar.reshape(nz,nx)
    coeff.C = Cr.reshape(nz,nx)
    
    coeff.Bx = Bxr.reshape(nz,nx+1)
    coeff.ux = uxr.reshape(nz,nx+1)
    coeff.Bz = Bzr.reshape(nz+1,nx)
    coeff.uz = uzr.reshape(nz+1,nx)

    return coeff
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def calc_center_velocities_div(vx,vz,xv,zv,nx,nz):

  vxc   = np.zeros([nz,nx])
  vzc   = np.zeros([nz,nx])
  divV  = np.zeros([nz,nx])

  for i in range(0,nx):
    for j in range(0,nz):
      vxc[j][i]  = 0.5 * (vx[j][i+1] + vx[j][i])
      vzc[j][i]  = 0.5 * (vz[j+1][i] + vz[j][i])
      divV[j][i] = (vx[j][i+1]-vx[j][i])/(xv[i+1]-xv[i]) + (vz[j+1][i] - vz[j][i])/(zv[j+1]-zv[j])

  return vxc, vzc, divV

# ---------------------------------
def calc_dom_rheology_mechanism(A):

  rheol = EmptyStruct()
  rheol.epsV_xx = 0.5*A.tau.xx_center/A.matProp.etaV
  rheol.epsV_zz = 0.5*A.tau.zz_center/A.matProp.etaV
  rheol.epsV_xz = 0.5*A.tau.xz_center/A.matProp.etaV
  rheol.epsV_II = (0.5*(rheol.epsV_xx**2 + rheol.epsV_zz**2)+rheol.epsV_xz**2)**0.5

  rheol.epsE_xx = 0.5*(A.tau.xx_center-A.tauold.xx_center)/A.matProp.etaE
  rheol.epsE_zz = 0.5*(A.tau.zz_center-A.tauold.zz_center)/A.matProp.etaE
  rheol.epsE_xz = 0.5*(A.tau.xz_center-A.tauold.xz_center)/A.matProp.etaE
  rheol.epsE_II = (0.5*(rheol.epsE_xx**2 + rheol.epsE_zz**2)+rheol.epsE_xz**2)**0.5

  rheol.epsP_xx = 0.5*A.tau.xx_center/A.matProp.etaP
  rheol.epsP_zz = 0.5*A.tau.zz_center/A.matProp.etaP
  rheol.epsP_xz = 0.5*A.tau.xz_center/A.matProp.etaP
  rheol.epsP_II = (0.5*(rheol.epsP_xx**2 + rheol.epsP_zz**2)+rheol.epsP_xz**2)**0.5

  rheol.volV = -A.DP/A.matProp.zetaV
  rheol.volE = -(A.DP-A.DPold)/A.matProp.zetaE
  # rheol.volP = -A.DP/A.matProp.zetaP

  # difference 
  rheol.epsP2_II = A.eps.II_center-rheol.epsV_II-rheol.epsE_II
  rheol.volP2    = A.divVs-rheol.volV-rheol.volE

  # print(rheol.epsP_II)
  # print(rheol.epsP_II-rheol.epsP2_II)
  # print(rheol.volP)
  # print(rheol.volP-rheol.volP2)

  return rheol

# ---------------------------------
def calc_VEVP_strain_rates(A):

  rheol = EmptyStruct()
  rheol.epsV_xx = 0.5*A.phis*A.tau.xx_center/A.matProp.etaV
  rheol.epsV_zz = 0.5*A.phis*A.tau.zz_center/A.matProp.etaV
  rheol.epsV_xz = 0.5*A.phis*A.tau.xz_center/A.matProp.etaV
  rheol.epsV_II = (0.5*(rheol.epsV_xx**2 + rheol.epsV_zz**2)+rheol.epsV_xz**2)**0.5

  rheol.epsE_xx = 0.5*A.phis*(A.tau.xx_center-A.tauold.xx_center)/A.matProp.etaE
  rheol.epsE_zz = 0.5*A.phis*(A.tau.zz_center-A.tauold.zz_center)/A.matProp.etaE
  rheol.epsE_xz = 0.5*A.phis*(A.tau.xz_center-A.tauold.xz_center)/A.matProp.etaE
  rheol.epsE_II = (0.5*(rheol.epsE_xx**2 + rheol.epsE_zz**2)+rheol.epsE_xz**2)**0.5

  a1 = A.matProp.C*np.cos(A.matProp.theta*math.pi/180)-A.matProp.sigmat*np.sin(A.matProp.theta*math.pi/180)
  tauIIa1 = np.sqrt(A.tau.II_center**2+a1**2)

  rheol.epsVP_xx = 0.5*A.dotlam*A.tau.xx_center/tauIIa1
  rheol.epsVP_zz = 0.5*A.dotlam*A.tau.zz_center/tauIIa1
  rheol.epsVP_xz = 0.5*A.dotlam*A.tau.xz_center/tauIIa1
  rheol.epsVP_II = (0.5*(rheol.epsVP_xx**2 + rheol.epsVP_zz**2)+rheol.epsVP_xz**2)**0.5

  phi_min = 1e-4
  phi = 1.0-A.phis
  phi[phi<0.0] = 0.0
  cdl = np.exp(-phi_min/phi)
  rheol.volV = -A.phis*A.DP/A.matProp.zetaV
  rheol.volE = -A.phis*(A.DP-A.DPold)/A.matProp.zetaE
  rheol.volVP = A.dotlam*cdl*np.sin(A.matProp.theta*math.pi/180)

  return rheol

# ---------------------------------
def plot_standard(fig,ax,X,extent,title,lblx,lblz,cmin,cmax):
  im = ax.imshow(X,extent=extent,cmap='viridis',origin='lower')
  if (cmax-cmin!=0):
    im.set_clim(cmin,cmax)
  fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(title)
  return im

# ---------------------------------
def plot_solver_residuals(A,fname):

  fig = plt.figure(1,figsize=(30,15))

  ax = plt.subplot(3,4,1)
  pl = ax.plot(np.log10(A.sol.PV_it_res), linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Iteration')
  ax.set_ylabel('log10(PV residual)')

  ax = plt.subplot(3,4,5)
  ax.plot(np.log10(A.sol.T_it_res), linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Iteration')
  ax.set_ylabel('log10(T residual)')

  ax = plt.subplot(3,4,9)
  ax.plot(np.log10(A.sol.phi_it_res), linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Iteration')
  ax.set_ylabel('log10(phi residual)')

  ax = plt.subplot(3,4,2)
  pl = ax.plot(np.log10(A.sol.PV_ts_res), linewidth=0.1)
  for i in range(0,A.sol.tstep):
    if (A.sol.PV_ts_diverged[i]>0):
      ax.plot(i,np.log10(A.sol.PV_ts_res[i]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(PV residual)')

  ax = plt.subplot(3,4,6)
  ax.plot(np.log10(A.sol.T_ts_res), linewidth=0.5)
  for i in range(0,A.sol.tstep):
    if (A.sol.T_ts_diverged[i]>0):
      ax.plot(i,np.log10(A.sol.T_ts_res[i]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(T residual)')

  ax = plt.subplot(3,4,10)
  ax.plot(np.log10(A.sol.phi_ts_res), linewidth=0.5)
  for i in range(0,A.sol.tstep):
    if (A.sol.phi_ts_diverged[i]>0):
      ax.plot(i,np.log10(A.sol.phi_ts_res[i]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(phi residual)')

  ax = plt.subplot(3,4,3)
  pl = ax.plot(A.sol.PV_n_it, '*-', linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('PV iter')

  ax = plt.subplot(3,4,7)
  pl = ax.plot(A.sol.T_n_it, '*-', linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('T iter')

  ax = plt.subplot(3,4,11)
  pl = ax.plot(A.sol.phi_n_it, '*-', linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('phi iter')

  ax = plt.subplot(3,4,4)
  pl = ax.plot(A.sol.runtime, '*-', linewidth=0.1)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('Runtime (s)')

  ax = plt.subplot(3,4,8)
  pl = ax.plot(np.log10(A.sol.dt), linewidth=1.0)
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(dt)')

  plt.savefig(fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_T(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(7,5))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = scale_TC(A,'T','T',dim,1)
  lbl = get_label(A,'T',dim)

  ax = plt.subplot(1,1,1)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  # temperature contour
  if (dim):
    levels = [0, 250, 500, 750, 1000, 1200, 1300,]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap='plasma')
    ax.clabel(ts, fmt=fmt, fontsize=14)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_marker_id(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(6,5))
  ax = plt.subplot(1,1,1)

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap='viridis')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  cbar.ax.set_title('id')
  ax.set_xlim(min(A.grid.xv[istart:iend]*scalx), max(A.grid.xv[istart:iend]*scalx))
  ax.set_ylim(min(A.grid.zv[istart:iend]*scalx), max(A.grid.zv[istart:iend]*scalx))
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep), fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_MPhase(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,10))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]

  ax = plt.subplot(2,3,1)
  plot_standard(fig,ax,A.MPhase.CornerPh0[jstart:jend,istart:iend],extentV,'Corner Ph=0 tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,2)
  plot_standard(fig,ax,A.MPhase.CornerPh1[jstart:jend,istart:iend],extentV,'Corner Ph=1 tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,3)
  plot_standard(fig,ax,A.MPhase.CornerPh2[jstart:jend,istart:iend],extentV,'Corner Ph=2 tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,4)
  plot_standard(fig,ax,A.MPhase.CornerPh3[jstart:jend,istart:iend],extentV,'Corner Ph=3 tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,5)
  plot_standard(fig,ax,A.MPhase.CornerPh4[jstart:jend,istart:iend],extentV,'Corner Ph=4 tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,6)
  plot_standard(fig,ax,A.MPhase.CornerPh5[jstart:jend,istart:iend],extentV,'Corner Ph=5 tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_P(A,istart,iend,jstart,jend,fdir,fname,istep,dim,iplot):
  make_dir(fdir)
  if (iplot==1):
    figsize=(7,5)
  if (iplot==2):
    figsize=(14,5)
  if (iplot==3):
    figsize=(21,5)
  fig = plt.figure(1,figsize=figsize)

  scalx = get_scaling(A,'x',dim,1)
  scalP = get_scaling(A,'P',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  lblP = get_label(A,'P',dim)
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  if (iplot==1):
    ax = plt.subplot(1,1,1)
    plot_standard(fig,ax,A.Plith[jstart:jend,istart:iend]*scalP,extentC,lblP+':Plith tstep = '+str(istep),lblx,lblz,0,0)
  
  if (iplot==2):
    ax = plt.subplot(1,2,1)
    plot_standard(fig,ax,A.Plith[jstart:jend,istart:iend]*scalP,extentC,lblP+':Plith tstep = '+str(istep),lblx,lblz,0,0)

    ax = plt.subplot(1,2,2)
    plot_standard(fig,ax,A.DP[jstart:jend,istart:iend]*scalP,extentC,lblP+':DP tstep = '+str(istep),lblx,lblz,0,0)

  if (iplot==3):
    ax = plt.subplot(1,3,1)
    plot_standard(fig,ax,A.Plith[jstart:jend,istart:iend]*scalP,extentC,lblP+':Plith tstep = '+str(istep),lblx,lblz,0,0)

    ax = plt.subplot(1,3,2)
    plot_standard(fig,ax,A.DP[jstart:jend,istart:iend]*scalP,extentC,lblP+':DP tstep = '+str(istep),lblx,lblz,0,0)

    ax = plt.subplot(1,3,3)
    plot_standard(fig,ax,A.DPold[jstart:jend,istart:iend]*scalP,extentC,lblP+':DPold tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_plastic(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  figsize=(21,5)
  fig = plt.figure(1,figsize=figsize)

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  ax = plt.subplot(1,3,1)
  plot_standard(fig,ax,A.divVs[jstart:jend,istart:iend],extentC,r'$\nabla\cdot v$ [-] tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(1,3,2)
  plot_standard(fig,ax,A.dotlam[jstart:jend,istart:iend],extentC,r'$\dot{\lambda}$ [-] tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(1,3,3)
  plot_standard(fig,ax,A.lam[jstart:jend,istart:iend],extentC,r'$\lambda$ [-] tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_plastic_v2(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  figsize=(21,5)
  fig = plt.figure(1,figsize=figsize)

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  ax = plt.subplot(1,3,1)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14
  X4 = A.divVs*scal_v_ms/scal_x_m/scal

  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentC,cmap='seismic',origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-4,4)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s] tstep = '+str(istep))

  ax = plt.subplot(1,3,2)
  X = A.dotlam*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='binary',origin='lower')
  im.set_clim(0,100)
  # plot_standard(fig,ax,X[jstart:jend,istart:iend],extentC,r'$\dot{\lambda}$ [1/s] tstep = '+str(istep),lblx,lblz,0,0)
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\dot{\lambda}$ [1/s] tstep = '+str(istep))

  ax = plt.subplot(1,3,3)
  im = ax.imshow(A.lam[jstart:jend  ,istart:iend  ],extent=extentC,cmap='binary',origin='lower')
  # plot_standard(fig,ax,A.lam[jstart:jend,istart:iend],extentC,r'$\lambda$ [-] tstep = '+str(istep),lblx,lblz,0,0)
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  im.set_clim(0,5)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\lambda$ [-] tstep = '+str(istep))

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PV(A,istart,iend,jstart,jend,fdir,fname,istep,dim,iplot):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,5))

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
    lbl1 = get_label(A,'P',dim)
    lbl2 = get_label(A,'vsx',dim)
    lbl3 = get_label(A,'vsz',dim)

  if (iplot==1):
    X1 = A.P_res*scalP
    X2 = A.Vsx_res*scalv
    X3 = A.Vsz_res*scalv
    lbl1 = get_label(A,'resP',dim)
    lbl2 = get_label(A,'resvsx',dim)
    lbl3 = get_label(A,'resvsz',dim)

  X1 = X1[jstart:jend  ,istart:iend  ]
  X2 = X2[jstart:jend  ,istart:iend+1]
  X3 = X3[jstart:jend+1,istart:iend  ]

  extentP =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,3,1)
  plot_standard(fig,ax,X1,extentP,lbl1+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(1,3,2)
  plot_standard(fig,ax,X2,extentVx,lbl2,lblx,lblz,0,0)

  ax = plt.subplot(1,3,3)
  plot_standard(fig,ax,X3,extentVz,lbl3,lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Tensor(A,istart,iend,jstart,jend,fdir,fname,istep,dim,iplot):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(28,10))

  # return scaling params and labels
  scalv = get_scaling(A,'v',dim,1)
  scalx = get_scaling(A,'x',dim,1)

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  if (iplot==0): # eps
    scal = get_scaling(A,'eps',dim,0)
    lblxx  = get_label(A,'epsxx',dim)
    lblzz  = get_label(A,'epszz',dim)
    lblxz  = get_label(A,'epsxz',dim)
    lblII  = get_label(A,'epsII',dim)
    X1 = A.eps.xx_corner*scal
    X2 = A.eps.zz_corner*scal
    X3 = A.eps.xz_corner*scal
    X4 = A.eps.II_corner*scal

    X1_c = A.eps.xx_center*scal
    X2_c = A.eps.zz_center*scal
    X3_c = A.eps.xz_center*scal
    X4_c = A.eps.II_center*scal

  if (iplot==1): # tau
    scal = get_scaling(A,'P',dim,1)
    lblxx  = get_label(A,'tauxx',dim)
    lblzz  = get_label(A,'tauzz',dim)
    lblxz  = get_label(A,'tauxz',dim)
    lblII  = get_label(A,'tauII',dim)
    X1 = A.tau.xx_corner*scal
    X2 = A.tau.zz_corner*scal
    X3 = A.tau.xz_corner*scal
    X4 = A.tau.II_corner*scal

    X1_c = A.tau.xx_center*scal
    X2_c = A.tau.zz_center*scal
    X3_c = A.tau.xz_center*scal
    X4_c = A.tau.II_center*scal
  
  if (iplot==2): # tau_old
    scal = get_scaling(A,'P',dim,1)
    lblxx  = get_label(A,'tauxx',dim)
    lblzz  = get_label(A,'tauzz',dim)
    lblxz  = get_label(A,'tauxz',dim)
    lblII  = get_label(A,'tauII',dim)
    X1 = A.tauold.xx_corner*scal
    X2 = A.tauold.zz_corner*scal
    X3 = A.tauold.xz_corner*scal
    X4 = A.tauold.II_corner*scal

    X1_c = A.tauold.xx_center*scal
    X2_c = A.tauold.zz_center*scal
    X3_c = A.tauold.xz_center*scal
    X4_c = A.tauold.II_center*scal


  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,4,1)
  plot_standard(fig,ax,X1,extentV,'CORNER: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,4,2)
  plot_standard(fig,ax,X2,extentV,lblzz,lblx,lblz,0,0)

  ax = plt.subplot(2,4,3)
  plot_standard(fig,ax,X3,extentV,lblxz,lblx,lblz,0,0)

  ax = plt.subplot(2,4,4)
  plot_standard(fig,ax,X4,extentV,lblII,lblx,lblz,0,0)

  ax = plt.subplot(2,4,5)
  plot_standard(fig,ax,X1_c,extentE,'CENTER: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,4,6)
  plot_standard(fig,ax,X2_c,extentE,lblzz,lblx,lblz,0,0)

  ax = plt.subplot(2,4,7)
  plot_standard(fig,ax,X3_c,extentE,lblxz,lblx,lblz,0,0)

  ax = plt.subplot(2,4,8)
  plot_standard(fig,ax,X4_c,extentE,lblII,lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(28,15))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]
  extentN =[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(3,4,1)
  plot_standard(fig,ax,A.PVcoeff.A_corner[jstart:jend+1,istart:iend+1],extentN,'A corner tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(3,4,2)
  plot_standard(fig,ax,A.PVcoeff.A[jstart:jend  ,istart:iend  ],extentE,'A center',lblx,lblz,0,0)

  ax = plt.subplot(3,4,3)
  plot_standard(fig,ax,A.PVcoeff.D1[jstart:jend  ,istart:iend  ],extentE,'D1 center',lblx,lblz,0,0)

  ax = plt.subplot(3,4,4)
  plot_standard(fig,ax,A.PVcoeff.C[jstart:jend  ,istart:iend  ],extentE,'C center',lblx,lblz,0,0)

  ax = plt.subplot(3,4,5)
  plot_standard(fig,ax,A.PVcoeff.Bx[jstart:jend  ,istart:iend+1],extentFx,'Bx face',lblx,lblz,0,0)

  ax = plt.subplot(3,4,6)
  plot_standard(fig,ax,A.PVcoeff.D2x[jstart:jend  ,istart:iend+1],extentFx,'D2x face',lblx,lblz,0,0)

  ax = plt.subplot(3,4,7)
  plot_standard(fig,ax,A.PVcoeff.D3x[jstart:jend  ,istart:iend+1],extentFx,'D3x face',lblx,lblz,0,0)

  ax = plt.subplot(3,4,9)
  plot_standard(fig,ax,A.PVcoeff.Bz[jstart:jend+1,istart:iend  ],extentFz,'Bz face',lblx,lblz,0,0)

  ax = plt.subplot(3,4,10)
  plot_standard(fig,ax,A.PVcoeff.D2z[jstart:jend+1,istart:iend  ],extentFz,'D2z face',lblx,lblz,0,0)

  ax = plt.subplot(3,4,11)
  plot_standard(fig,ax,A.PVcoeff.D3z[jstart:jend+1,istart:iend  ],extentFz,'D3z face',lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff_Stokes(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,10))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]
  extentN =[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,3,1)
  plot_standard(fig,ax,A.PVcoeff.A_corner[jstart:jend+1,istart:iend+1],extentN,'A corner tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,3,2)
  plot_standard(fig,ax,A.PVcoeff.A[jstart:jend  ,istart:iend  ],extentE,'A center',lblx,lblz,0,0)

  ax = plt.subplot(2,3,3)
  plot_standard(fig,ax,A.PVcoeff.C[jstart:jend  ,istart:iend  ],extentE,'C center',lblx,lblz,0,0)

  ax = plt.subplot(2,3,4)
  plot_standard(fig,ax,A.PVcoeff.Bx[jstart:jend  ,istart:iend+1],extentFx,'Bx face',lblx,lblz,0,0)

  ax = plt.subplot(2,3,5)
  plot_standard(fig,ax,A.PVcoeff.Bz[jstart:jend+1,istart:iend  ],extentFz,'Bz face',lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_matProp(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(28,20))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  ax = plt.subplot(4,4,1)
  scal = get_scaling(A,'eta',dim,0)
  scalP = get_scaling(A,'P',dim,1)
  lbl  = get_label(A,'eta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.eta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(4,4,2)
  plot_standard(fig,ax,np.log10(A.matProp.etaV[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 V '+lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,3)
  if np.linalg.norm(A.matProp.etaE)==0:
    plot_standard(fig,ax,A.matProp.etaE[jstart:jend  ,istart:iend  ]*scal,extentE,'E '+lbl,lblx,lblz,0,0)
  else:
    plot_standard(fig,ax,np.log10(A.matProp.etaE[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 E '+lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,4)
  plot_standard(fig,ax,np.log10(A.matProp.etaP[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 P '+lbl,lblx,lblz,0,0)

  X = A.matProp.zeta*scal
  # print(X[X<=1e15])
  
  lbl  = get_label(A,'zeta',dim)
  ax = plt.subplot(4,4,5)
  plot_standard(fig,ax,np.log10(A.matProp.zeta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,6)
  plot_standard(fig,ax,np.log10(A.matProp.zetaV[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 V '+lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,7)
  if np.linalg.norm(A.matProp.zetaE)==0:
    plot_standard(fig,ax,A.matProp.zetaE[jstart:jend  ,istart:iend  ]*scal,extentE,'E '+lbl,lblx,lblz,0,0)
  else:
    plot_standard(fig,ax,np.log10(A.matProp.zetaE[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 E '+lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,8)
  plot_standard(fig,ax,A.matProp.DPdl[jstart:jend  ,istart:iend  ]*scalP,extentE,'DPdl [MPa]',lblx,lblz,0,0)

  ax = plt.subplot(4,4,9)
  lbl  = get_label(A,'Z',dim)
  scal = get_scaling(A,'P',dim,1)
  plot_standard(fig,ax,A.matProp.Z[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,10)
  lbl  = get_label(A,'G',dim)
  scal = get_scaling(A,'P',dim,1)
  plot_standard(fig,ax,A.matProp.G[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,11)
  scal = get_scaling(A,'rho',dim,0)
  lbl  = get_label(A,'rho',dim)
  im = plot_standard(fig,ax,A.matProp.rho[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,12)
  scal = get_scaling(A,'Kphi',dim,0)
  lbl  = get_label(A,'Kphi',dim)
  im = plot_standard(fig,ax,A.matProp.Kphi[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,13)
  lbl  = get_label(A,'C',dim)
  scal = get_scaling(A,'P',dim,1)
  plot_standard(fig,ax,A.matProp.C[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,14)
  lbl  = get_label(A,'sigmat',dim)
  im = plot_standard(fig,ax,A.matProp.sigmat[jstart:jend  ,istart:iend  ]*scal,extentE,lbl,lblx,lblz,0,0)

  ax = plt.subplot(4,4,15)
  lbl  = get_label(A,'theta',dim)
  im = plot_standard(fig,ax,A.matProp.theta[jstart:jend  ,istart:iend  ],extentE,lbl,lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Vel(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(14,10))

  scalv = get_scaling(A,'v',dim,1)
  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentVx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentVz=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,2,1)
  lbl = get_label(A,'vfx',dim)
  plot_standard(fig,ax,A.Vfx[jstart:jend  ,istart:iend+1]*scalv,extentVx,lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(2,2,2)
  lbl = get_label(A,'vfz',dim)
  plot_standard(fig,ax,A.Vfz[jstart:jend+1,istart:iend  ]*scalv,extentVz,lbl,lblx,lblz,0,0)

  ax = plt.subplot(2,2,3)
  lbl = get_label(A,'vx',dim)
  plot_standard(fig,ax,A.Vx[jstart:jend  ,istart:iend+1]*scalv,extentVx,lbl,lblx,lblz,0,0)

  ax = plt.subplot(2,2,4)
  lbl = get_label(A,'vz',dim)
  plot_standard(fig,ax,A.Vz[jstart:jend+1,istart:iend  ]*scalv,extentVz,lbl,lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Tcoeff(A,istart,iend,jstart,jend,fdir,fname,istep,dim,iplot):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(14,15))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE =[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFx=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentFz=[min(A.grid.xc[istart:iend+1])*scalx, max(A.grid.xc[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]
  
  if (iplot==0):
    X1 = A.Tcoeff.A
    X2 = A.Tcoeff.C
    X3 = A.Tcoeff.Bx
    X4 = A.Tcoeff.Bz
    X5 = A.Tcoeff.ux
    X6 = A.Tcoeff.uz
  if (iplot==1):
    X1 = A.phicoeff.A
    X2 = A.phicoeff.C
    X3 = A.phicoeff.Bx
    X4 = A.phicoeff.Bz
    X5 = A.phicoeff.ux
    X6 = A.phicoeff.uz

  ax = plt.subplot(3,2,1)
  plot_standard(fig,ax,X1[jstart:jend  ,istart:iend  ],extentE,'A center',lblx,lblz,0,0)

  ax = plt.subplot(3,2,2)
  plot_standard(fig,ax,X2[jstart:jend  ,istart:iend  ],extentE,'C center',lblx,lblz,0,0)

  ax = plt.subplot(3,2,3)
  plot_standard(fig,ax,X3[jstart:jend  ,istart:iend  ],extentFx,'Bx face',lblx,lblz,0,0)

  ax = plt.subplot(3,2,4)
  plot_standard(fig,ax,X4[jstart:jend+1,istart:iend  ],extentFz,'Bz face',lblx,lblz,0,0)

  ax = plt.subplot(3,2,5)
  plot_standard(fig,ax,X5[jstart:jend  ,istart:iend  ],extentFx,'ux face',lblx,lblz,0,0)

  ax = plt.subplot(3,2,6)
  plot_standard(fig,ax,X6[jstart:jend+1,istart:iend  ],extentFz,'uz face',lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_individual_eps(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(28,25))

  # return scaling params and labels
  scal = get_scaling(A,'eps',dim,0)
  scalx = get_scaling(A,'x',dim,1)
  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  lblII  = get_label(A,'epsII',dim)
  lblxx  = get_label(A,'epsxx',dim)
  lblzz  = get_label(A,'epszz',dim)
  lblxz  = get_label(A,'epsxz',dim)
  
  X1 = A.eps.xx_center*scal
  X2 = A.eps.zz_center*scal
  X3 = A.eps.xz_center*scal
  X4 = A.eps.II_center*scal

  ax = plt.subplot(5,4,1)
  plot_standard(fig,ax,X1,extentE,'Total: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)
  ax = plt.subplot(5,4,2)
  plot_standard(fig,ax,X2,extentE,lblzz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,3)
  plot_standard(fig,ax,X3,extentE,lblxz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,4)
  plot_standard(fig,ax,X4,extentE,lblII,lblx,lblz,0,0)

  X1 = A.rheol.epsV_xx*scal
  X2 = A.rheol.epsV_zz*scal
  X3 = A.rheol.epsV_xz*scal
  X4 = A.rheol.epsV_II*scal

  ax = plt.subplot(5,4,5)
  plot_standard(fig,ax,X1,extentE,'V: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)
  ax = plt.subplot(5,4,6)
  plot_standard(fig,ax,X2,extentE,lblzz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,7)
  plot_standard(fig,ax,X3,extentE,lblxz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,8)
  plot_standard(fig,ax,X4,extentE,lblII,lblx,lblz,0,0)

  X1 = A.rheol.epsE_xx*scal
  X2 = A.rheol.epsE_zz*scal
  X3 = A.rheol.epsE_xz*scal
  X4 = A.rheol.epsE_II*scal

  ax = plt.subplot(5,4,9)
  plot_standard(fig,ax,X1,extentE,'E: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)
  ax = plt.subplot(5,4,10)
  plot_standard(fig,ax,X2,extentE,lblzz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,11)
  plot_standard(fig,ax,X3,extentE,lblxz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,12)
  plot_standard(fig,ax,X4,extentE,lblII,lblx,lblz,0,0)
  
  X1 = A.rheol.epsP_xx*scal
  X2 = A.rheol.epsP_zz*scal
  X3 = A.rheol.epsP_xz*scal
  X4 = A.rheol.epsP_II*scal

  ax = plt.subplot(5,4,13)
  plot_standard(fig,ax,X1,extentE,'P: '+lblxx+' tstep = '+str(istep),lblx,lblz,0,0)
  ax = plt.subplot(5,4,14)
  plot_standard(fig,ax,X2,extentE,lblzz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,15)
  plot_standard(fig,ax,X3,extentE,lblxz,lblx,lblz,0,0)
  ax = plt.subplot(5,4,16)
  plot_standard(fig,ax,X4,extentE,lblII,lblx,lblz,0,0)

  X1 = A.divVs*scal
  X2 = A.rheol.volV*scal
  X3 = A.rheol.volE*scal
  X4 = A.rheol.volP*scal

  ax = plt.subplot(5,4,17)
  plot_standard(fig,ax,X1,extentE,'Total: div(v) tstep = '+str(istep),lblx,lblz,0,0)
  ax = plt.subplot(5,4,18)
  plot_standard(fig,ax,X2,extentE,'volV',lblx,lblz,0,0)
  ax = plt.subplot(5,4,19)
  plot_standard(fig,ax,X3,extentE,'volE',lblx,lblz,0,0)
  ax = plt.subplot(5,4,20)
  plot_standard(fig,ax,X4,extentE,'volP',lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(7,5))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  # print(X[X<0])
  # X[X>0] = 0
  # X = -X
  # X[X<1e-10] = 1e-10

  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-6,0)
  # im.set_clim(-10,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\phi$'+' tstep = '+str(istep))

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_eta_eps_tau(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(28,5))
  

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,4,1)
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap='viridis')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  cbar.ax.set_title('id')
  ax.set_xlim(min(A.grid.xv[istart:iend]*scalx), max(A.grid.xv[istart:iend]*scalx))
  ax.set_ylim(min(A.grid.zv[istart:iend]*scalx), max(A.grid.zv[istart:iend]*scalx))
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  ax = plt.subplot(1,4,2)
  scal = get_scaling(A,'eta',dim,0)
  lbl  = get_label(A,'eta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.eta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  maxV = 1
  nind = 5
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)

  ax = plt.subplot(1,4,3)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  X4 = A.eps.II_corner*scal
  plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)

  ax = plt.subplot(1,4,4)
  lblII  = get_label(A,'tauII',dim)
  scal = get_scaling(A,'P',dim,1)
  X4 = A.tau.II_corner*scal
  plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_eta_eps_tau2(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(14,10))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,2,1)
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap='viridis')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  cbar.ax.set_title('id')
  ax.set_xlim(min(A.grid.xv[istart:iend]*scalx), max(A.grid.xv[istart:iend]*scalx))
  ax.set_ylim(min(A.grid.zv[istart:iend]*scalx), max(A.grid.zv[istart:iend]*scalx))
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  ax = plt.subplot(2,2,2)
  scal = get_scaling(A,'eta',dim,0)
  lbl  = get_label(A,'eta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.eta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  maxV = 1
  nind = 5
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)

  ax = plt.subplot(2,2,3)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  X4 = A.eps.II_corner*scal
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,origin='lower')
  im.set_clim(-16,-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('CORNER: log10 '+lblII+' tstep = '+str(istep))

  ax = plt.subplot(2,2,4)
  lblII  = get_label(A,'tauII',dim)
  scal = get_scaling(A,'P',dim,1)
  X4 = A.tau.II_corner*scal
  plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_eta_eps_tau_T_phi(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,8))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(2,3,1)
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap='viridis')
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  cbar.ax.set_title('id')
  ax.set_xlim(min(A.grid.xv[istart:iend]*scalx), max(A.grid.xv[istart:iend]*scalx))
  ax.set_ylim(min(A.grid.zv[istart:iend]*scalx), max(A.grid.zv[istart:iend]*scalx))
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  ax = plt.subplot(2,3,2)
  X = scale_TC(A,'T','T',dim,1)
  lbl = get_label(A,'T',dim)
  plot_standard(fig,ax,X[jstart:jend,istart:iend],extentE,lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  # temperature contour
  if (dim):
    levels = [0, 250, 500, 750, 1000, 1200, 1250, 1300,]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap='plasma')
    ax.clabel(ts, fmt=fmt, fontsize=14)

  ax = plt.subplot(2,3,3)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im.set_clim(-6,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\phi$'+' tstep = '+str(istep))

  ax = plt.subplot(2,3,4)
  scal = get_scaling(A,'eta',dim,0)
  lbl  = get_label(A,'eta',dim)
  plot_standard(fig,ax,np.log10(A.matProp.eta[jstart:jend  ,istart:iend  ]*scal),extentE,'log10 '+lbl+' tstep = '+str(istep),lblx,lblz,0,0)

  maxV = 1
  nind = 5
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)

  ax = plt.subplot(2,3,5)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  X4 = A.eps.II_corner*scal
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,origin='lower')
  im.set_clim(-16,-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('CORNER: log10 '+lblII+' tstep = '+str(istep))

  ax = plt.subplot(2,3,6)
  lblII  = get_label(A,'tauII',dim)
  scal = get_scaling(A,'P',dim,1)
  X4 = A.tau.II_corner*scal
  plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_eps_phi(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,4))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,3,1)
  # cmap1 = 'cividis'
  # cmap1 = 'viridis'
  cmap1 = 'binary_r'
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap=cmap1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  cbar.ax.set_title('id')
  ax.set_xlim(A.grid.xv[0]*scalx, A.grid.xv[-1]*scalx)
  ax.set_ylim(A.grid.zv[0]*scalx, A.grid.zv[-1]*scalx)
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  maxV = 1
  nind = 10 # 10 for high res/ 5 for low res
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)


  ax = plt.subplot(1,3,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  # cmap1 = cm.broc_r
  X4 = A.eps.II_corner*scal
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  # im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,cmap='seismic',origin='lower')
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='seismic',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(-16,-13)
  im.set_clim(1e-16,1e-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title('log10 '+lblII)
  ax.set_title(lblII)

  ax = plt.subplot(1,3,3)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # X[X>1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_eps_divv(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,4))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,3,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # X[X>1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('time = '+str(round(t/1.0e3,0))+' [kyr]  '+r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  ax = plt.subplot(1,3,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  # cmap1 = cm.broc_r
  X4 = A.eps.II_corner*scal
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  # im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,cmap='seismic',origin='lower')
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='seismic',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(-16,-13)
  im.set_clim(1e-16,1e-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title('log10 '+lblII)
  ax.set_title(lblII)

  # Topography using markers
  x_mark = np.zeros(A.nx*2)
  topo_mark = np.zeros(A.nx*2)

  for i in range(0,A.nx):
    x_mark[2*i  ] = A.grid.xv[i]
    x_mark[2*i+1] = A.grid.xc[i]

  dx2 = (x_mark[1]-x_mark[0])*0.5

  for i in range(0,A.nx*2):
    ztopo = 0.0
    for j in range(0,A.mark.n):
      if (A.mark.x[j]>=x_mark[i]-dx2) & (A.mark.x[j]<x_mark[i]+dx2) & (A.mark.id[j]==0):
        ztopo = min(ztopo,A.mark.z[j]*scalx)
    topo_mark[i] = ztopo

  ax = plt.subplot(1,3,3)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  # cmap1 = cm.broc_r
  scal = 1e-14
  X4 = A.divVs*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='seismic',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  im.set_clim(-4,4)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s]')

  maxV = 1
  nind = 10 # 10 for high res/ 5 for low res
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_eps(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(13,4))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,2,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # X[X>1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  # cbar.ax.set_title(r'log$_{10}\phi$')
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('time = '+str(round(t/1.0e3,0))+' [kyr]  '+r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  ax = plt.subplot(1,2,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  # cmap1 = cm.batlowW
  X4 = np.log10(A.eps.II_corner*scal)
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  # im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,cmap='seismic',origin='lower')
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-16,-13)
  # im.set_clim(1e-16,1e-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('log10 '+lblII)
  # ax.set_title(lblII)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_bulk_darcy_alpha(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  # fig = plt.figure(1,figsize=(13,4))
  fig = plt.figure(1,figsize=(13,6))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  A.Cx   = np.zeros([A.nz,A.nx])
  A.Cz   = np.zeros([A.nz,A.nx])
  A.Sx   = np.zeros([A.nz,A.nx])
  A.Sz   = np.zeros([A.nz,A.nx])

  Sx = A.Vx - A.Vsx
  Sz = A.Vz - A.Vsz
  
  phi = 1.0 - A.phis
  phi[phi<0.0] = 0.0

  Cp = A.divVs - (1.0-phi)*A.DPold/A.matProp.Z/A.nd.dt
  A.zeta_eff = A.matProp.zeta - A.matProp.DPdl/Cp
  A.alpha_eta = A.matProp.eta/(A.zeta_eff+4/3*A.matProp.eta)

  # center velocities
  for i in range(0,A.nx):
    for j in range(0,A.nz):
      A.Cx[j][i]  = 0.5 * (A.Vx[j][i+1] + A.Vx[j][i])
      A.Cz[j][i]  = 0.5 * (A.Vz[j+1][i] + A.Vz[j][i])
      A.Sx[j][i]  = 0.5 * (Sx[j][i+1] + Sx[j][i])
      A.Sz[j][i]  = 0.5 * (Sz[j+1][i] + Sz[j][i])

  # Topography using markers
  x_mark = np.zeros(A.nx*2)
  topo_mark = np.zeros(A.nx*2)

  for i in range(0,A.nx):
    x_mark[2*i  ] = A.grid.xv[i]
    x_mark[2*i+1] = A.grid.xc[i]

  dx2 = (x_mark[1]-x_mark[0])*0.5

  for i in range(0,A.nx*2):
    ztopo = 0.0
    for j in range(0,A.mark.n):
      if (A.mark.x[j]>=x_mark[i]-dx2) & (A.mark.x[j]<x_mark[i]+dx2) & (A.mark.id[j]==0):
        ztopo = min(ztopo,A.mark.z[j]*scalx)
    topo_mark[i] = ztopo

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,2,1)
  levels = [1e-4, 1]
  philevel = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, phi[jstart:jend  ,istart:iend  ], levels=levels,colors=('silver'),linewidths=(0.5,), extend='both')

  maxV = 1
  nind = 4 # 10 for high res/ 5 for low res
  Q = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Cx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Cz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale=0.5, width=0.005, headwidth=4, headaxislength=4, minlength=0, label =r'$\textbf{c}$')
  # Q = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Cx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Cz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
  #     color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.03, headwidth=5, headaxislength=5, minlength=0)

  Q1 = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Sx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Sz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='red', scale=0.5, width=0.005, headwidth=4, headaxislength=4, minlength=0, label =r'$\textbf{s}$')

  maxVf = 5
  qk = ax.quiverkey(Q, X=0.05, Y=-0.1, U=maxVf, label=str(maxVf)+' cm/yr', labelpos='E')
  # qk1 = ax.quiverkey(Q1, X=0.0, Y=-0.1, U=maxVf, label=r'$\textbf{s}$ '+str(maxVf)+' cm/yr', labelpos='E')
  
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # ax.legend()
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('time = '+str(round(t/1.0e3,0))+' [kyr]  '+r'black:$\textbf{c}$ red:$\textbf{s}$ [cm/yr]')

  ax = plt.subplot(1,2,2)
  im = ax.imshow(A.alpha_eta[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  philevel = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, phi[jstart:jend  ,istart:iend  ], levels=levels,colors=('silver'),linewidths=(0.5,), extend='both')
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  im.set_clim(0,1.0)
  cbar = fig.colorbar(im,ax=ax, shrink=1.00)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\alpha_\eta = \eta/(\zeta+4/3\eta)$')
  # ax.set_title(lblII)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_eps_div_eta_zeta_lam(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(25,6))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  nplots = 6
  ax = plt.subplot(1,nplots,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')

  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]  '+r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10-2)

  ax = plt.subplot(1,nplots,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  X4 = np.log10(A.eps.II_corner*scal)
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-16,-13)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title('log10 '+lblII)

  ax = plt.subplot(1,nplots,3)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14
  X4 = A.divVs*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-2,2)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  # ax.set_ylabel(lblz)
  ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s]')

  ax = plt.subplot(1,nplots,4)
  scal_eta = get_scaling(A,'eta',dim,0)
  X = np.log10(A.matProp.eta*scal_eta)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(np.log10(A.nd.eta_min*scal_eta),np.log10(A.nd.eta_max*scal_eta))
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title(r'$\log_{10}\eta$ [Pa.s]')

  ax = plt.subplot(1,nplots,5)
  X = np.log10(A.matProp.zeta*scal_eta)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(np.log10(A.nd.eta_min*scal_eta),np.log10(A.nd.eta_max*scal_eta))
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title(r'$\log_{10}\zeta$ [Pa.s]')

  ax = plt.subplot(1,nplots,6)
  X = A.lam
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(0,3)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title(r'$\lambda$ [-]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_lam(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(8,6))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  # Topography using markers
  x_mark = np.zeros(A.nx*2)
  topo_mark = np.zeros(A.nx*2)

  for i in range(0,A.nx):
    x_mark[2*i  ] = A.grid.xv[i]
    x_mark[2*i+1] = A.grid.xc[i]

  dx2 = (x_mark[1]-x_mark[0])*0.5

  for i in range(0,A.nx*2):
    ztopo = 0.0
    for j in range(0,A.mark.n):
      if (A.mark.x[j]>=x_mark[i]-dx2) & (A.mark.x[j]<x_mark[i]+dx2) & (A.mark.id[j]==0):
        ztopo = min(ztopo,A.mark.z[j]*scalx)
    topo_mark[i] = ztopo

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  nplots = 2
  ax = plt.subplot(1,nplots,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # cmap1 = plt.cm.get_cmap('inferno', 20)
  cmap1 = plt.cm.get_cmap('cubehelix_r', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)

  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]  '+r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  ax = plt.subplot(1,nplots,2)
  X = A.lam
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cm.davos_r,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  im.set_clim(0,0.02)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title(r'$\lambda$ [-]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_eps_div_lam_norm(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(16,6))
  # fig = plt.figure(1,figsize=(35,6))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  # Topography using markers
  x_mark = np.zeros(A.nx*2)
  topo_mark = np.zeros(A.nx*2)

  for i in range(0,A.nx):
    x_mark[2*i  ] = A.grid.xv[i]
    x_mark[2*i+1] = A.grid.xc[i]

  dx2 = (x_mark[1]-x_mark[0])*0.5

  for i in range(0,A.nx*2):
    ztopo = 0.0
    for j in range(0,A.mark.n):
      if (A.mark.x[j]>=x_mark[i]-dx2) & (A.mark.x[j]<x_mark[i]+dx2) & (A.mark.id[j]==0):
        ztopo = min(ztopo,A.mark.z[j]*scalx)
    topo_mark[i] = ztopo

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  nplots = 4
  ax = plt.subplot(1,nplots,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # cmap1 = plt.cm.get_cmap('inferno', 20)
  cmap1 = plt.cm.get_cmap('cubehelix_r', 40)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  cbar.ax.set_title(r'log$_{10}\phi$')
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)

  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]  ')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1200, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  ax = plt.subplot(1,nplots,2)
  im = plot_8ind_epsII_normalised(fig,ax,A,istart,iend,jstart,jend,1,0)
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  ax.set(yticklabels=[])
  ax.set_title(' ')
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  # cbar.ax.set_title(r'log$_{10}\dot{\epsilon}_{II}$ [1/s]')
  cbar.ax.set_title(r'Normalised $\dot{\epsilon}_{II}$')

  ax = plt.subplot(1,nplots,3)
  im = plot_8ind_divvs_normalised(fig,ax,A,istart,iend,jstart,jend,1,0)
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  ax.set(yticklabels=[])
  ax.set_title(' ')
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  # cbar.ax.set_title(r'$\mathcal{C}$ [$\times 10^{-14}$ 1/s]')
  cbar.ax.set_title(r'Normalised $\mathcal{C}$')

  ax = plt.subplot(1,nplots,4)
  X = A.lam
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap=cm.davos_r,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  pl = ax.plot(x_mark*scalx,topo_mark,'k-',linewidth=0.5)
  im.set_clim(0,0.02)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set(yticklabels=[])
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  cbar.ax.set_title(r'$\lambda$ [-]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_eps_div_lam(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(17,6))
  # fig = plt.figure(1,figsize=(35,6))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  nplots = 4
  ax = plt.subplot(1,nplots,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')

  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]  '+r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1200, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10-2)

  ax = plt.subplot(1,nplots,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  X4 = np.log10(A.eps.II_corner*scal)
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-16,-13)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title('log10 '+lblII)

  ax = plt.subplot(1,nplots,3)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14
  X4 = A.divVs*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(-2,2)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  # ax.set_ylabel(lblz)
  ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s]')

  ax = plt.subplot(1,nplots,4)
  X = A.dotlam*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentE,cmap='RdYlBu_r',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_title(r'$\dot{\lambda}$ [1/s]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_phi_cider(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(6,8))
  fontsize0 = 14

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  nplots = 1
  ax = plt.subplot(1,nplots,1)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  cbar = fig.colorbar(im,ax=ax,orientation='horizontal', shrink=0.7)
  cbar.set_label(r'log$_{10}\phi$', fontsize=fontsize0+2)
  cbar.ax.tick_params(labelsize=fontsize0)

  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx,fontsize=fontsize0)
  ax.set_ylabel(lblz,fontsize=fontsize0)
  plt.yticks(fontsize=fontsize0)
  plt.xticks(fontsize=fontsize0)
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]',fontsize=fontsize0+2)

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1200, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=fontsize0)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_divs_phi(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(21,4))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,3,1)
  # cmap1 = 'cividis'
  # cmap1 = 'viridis'
  cmap1 = 'binary_r'
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap=cmap1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  cbar.ax.set_title('id')
  ax.set_xlim(A.grid.xv[0]*scalx, A.grid.xv[-1]*scalx)
  ax.set_ylim(A.grid.zv[0]*scalx, A.grid.zv[-1]*scalx)
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  maxV = 1
  nind = 10 # 10 for high res/ 5 for low res
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)


  ax = plt.subplot(1,3,2)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  # cmap1 = cm.broc_r
  scal = 1e-14
  X4 = A.divVs*scal_v_ms/scal_x_m/scal
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='seismic',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-10,10)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s]')

  ax = plt.subplot(1,3,3)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # X[X>1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_mark_eps_phi_column3(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(10,15))

  scalx = get_scaling(A,'x',dim,1)
  scalv = get_scaling(A,'v',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  scalt = get_scaling(A,'t',dim,1)
  t = A.nd.t*scalt
  
  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
  extentV=[min(A.grid.xv[istart:iend+1])*scalx, max(A.grid.xv[istart:iend+1])*scalx, min(A.grid.zv[jstart:jend+1])*scalx, max(A.grid.zv[jstart:jend+1])*scalx]

  ax = plt.subplot(1,3,1)
  # cmap1 = 'cividis'
  # cmap1 = 'viridis'
  cmap1 = 'binary_r'
  im = ax.scatter(A.mark.x*scalx,A.mark.z*scalx,c=A.mark.id,s=0.5,linewidths=None,cmap=cmap1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  cbar.ax.set_title('id')
  ax.set_xlim(A.grid.xv[0]*scalx, A.grid.xv[-1]*scalx)
  ax.set_ylim(A.grid.zv[0]*scalx, A.grid.zv[-1]*scalx)
  ax.set_aspect('equal')
  ax.set_title('PIC'+' tstep = '+str(istep)+' time = '+str(round(t/1.0e3,0))+' [kyr]', fontweight='bold')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)

  maxV = 1
  nind = 10 # 10 for high res/ 5 for low res
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.003, headwidth=5, headaxislength=5, minlength=0)
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)


  ax = plt.subplot(1,3,2)
  lblII  = get_label(A,'epsII',dim)
  scal = get_scaling(A,'eps',dim,0)
  # cmap1 = cm.broc_r
  X4 = A.eps.II_corner*scal
  # plot_standard(fig,ax,X4,extentV,'CORNER: '+lblII+' tstep = '+str(istep),lblx,lblz,0,0)
  # im = ax.imshow(np.log10(X4[jstart:jend  ,istart:iend  ]),extent=extentV,cmap='seismic',origin='lower')
  im = ax.imshow(X4[jstart:jend  ,istart:iend  ],extent=extentV,cmap='seismic',origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(-16,-13)
  im.set_clim(1e-16,1e-13)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  # ax.set_title('log10 '+lblII)
  ax.set_title(lblII)

  ax = plt.subplot(1,3,3)
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  # X[X>1e-10] = 1e-10
  cmap1 = plt.cm.get_cmap('inferno', 20)
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  ax.axis('image')
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\phi$')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  cmap1 = plt.cm.get_cmap('binary',10)

  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmap1)
    ax.clabel(ts, fmt=fmt, fontsize=10)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_def_mechanisms(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(14,5))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)

  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]

  X1 = A.rheol.epsV_II/A.eps.II_center
  X2 = A.rheol.epsE_II/A.eps.II_center
  X3 = A.rheol.epsP_II/A.eps.II_center
  X = np.zeros([A.nz,A.nx])

  C1 = A.rheol.volV/A.divVs
  C2 = A.rheol.volE/A.divVs
  C3 = A.rheol.volP/A.divVs
  C = np.zeros([A.nz,A.nx])

  for i in range(0,A.nx):
    for j in range(0,A.nz):
      if (X1[j][i]>X2[j][i]) & (X1[j][i]>X3[j][i]):
        X[j][i] = 0
      if (X2[j][i]>X1[j][i]) & (X2[j][i]>X3[j][i]):
        X[j][i] = 1
      if (X3[j][i]>X2[j][i]) & (X3[j][i]>X1[j][i]):
        X[j][i] = 2
      if (C1[j][i]>C2[j][i]) & (C1[j][i]>C3[j][i]):
        C[j][i] = 0
      if (C2[j][i]>C1[j][i]) & (C2[j][i]>C3[j][i]):
        C[j][i] = 1
      if (C3[j][i]>C2[j][i]) & (C3[j][i]>C1[j][i]):
        C[j][i] = 2

  ax = plt.subplot(1,2,1)
  plot_standard(fig,ax,X,extentE,'SHEAR tstep = '+str(istep),lblx,lblz,0,2)

  ax = plt.subplot(1,2,2)
  plot_standard(fig,ax,C,extentE,'VOL tstep = '+str(istep),lblx,lblz,0,2)

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8ind_phi(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
    
  extentE=[min(A.grid.xc[istart:iend  ])*scalx, max(A.grid.xc[istart:iend  ])*scalx, min(A.grid.zc[jstart:jend  ])*scalx, max(A.grid.zc[jstart:jend  ])*scalx]
    
  cmap1 = plt.cm.get_cmap('inferno', 20)
  cmapT = plt.cm.get_cmap('binary',10)
  
  X = 1.0 - A.phis
  X[X<1e-10] = 1e-10
  t = A.nd.t*scalt
  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentE,cmap=cmap1,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-6,-1)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  ax.set_title('t = '+str(round(t/1.0e3))+' [kyr]')

  # temperature contour
  X = scale_TC(A,'T','T',dim,1)
  if (dim):
    levels = [0, 250, 500, 750, 1000, 1300, 1500]
    fmt = r'%0.0f $^o$C'
    ts = ax.contour(A.grid.xc[istart:iend  ]*scalx, A.grid.zc[jstart:jend  ]*scalx, X[jstart:jend  ,istart:iend  ], levels=levels,linewidths=(1.0,), extend='both',cmap=cmapT)
    if (A.nd.istep==0):
      ax.clabel(ts, fmt=fmt, fontsize=10)
  
  return im

# ---------------------------------
def plot_8ind_epsII(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'eps',dim,0)

  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]
  
  X = np.log10(A.eps.II_corner*scal)
  t = A.nd.t*scalt
  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  # im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentV,cmap='RdYlBu_r',origin='lower')
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentV,cmap=cm.davos_r,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(1e-16,1e-13)
  im.set_clim(-18,-14)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   lblII  = get_label(A,'epsII',dim)
  #   ax.set_title('log10 '+lblII+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_epsII_normalised(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'eps',dim,0)

  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]
  t = A.nd.t*scalt
  scal0 = np.amax(A.eps.II_corner)
  X = A.eps.II_corner/scal0

  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentV,cmap=cm.davos_r,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(0,1)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  
  return im

# ---------------------------------
def plot_8ind_divvs(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.divVs*scal_v_ms/scal_x_m/scal
  # scal=max(np.amax(A.divVs),-np.amin(A.divVs))
  # X = A.divVs/scal
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  # im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='RdYlBu_r',origin='lower')
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap=cm.broc,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-0.01,0.01)
  # im.set_clim(-0.5,0.5)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   ax.set_title(r'$\nabla\cdot v_s$ [$\times 10^{-14}$ 1/s] ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  # if (yaxis):
  #   ax.set_title(r'$\nabla\cdot v_s$ [-] ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))

  return im

# ---------------------------------
def plot_8ind_divvs_normalised(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  scal=max(np.amax(A.divVs),-np.amin(A.divVs))
  X = A.divVs/scal

  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap=cm.broc,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-0.5,0.5)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')

  return im

# ---------------------------------
def plot_8ind_tauII(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'P',dim,1)

  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]
  
  X = A.tau.II_corner*scal
  t = A.nd.t*scalt
  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(np.log10(X[jstart:jend  ,istart:iend  ]),extent=extentV,cmap=cm.davos_r,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(0,5)
  im.set_clim(-3.5,1.5)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   lblII  = get_label(A,'tauII',dim)
  #   ax.set_title(lblII+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_DP(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scalP = get_scaling(A,'P',dim,1)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.DP*scalP
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  # im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='seismic',origin='lower')
  # # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(-10,10)

  divnorm=colors.TwoSlopeNorm(vmin=-5, vcenter=0., vmax=30)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap=cm.broc,origin='lower',norm=divnorm)
  im.set_clim(-5,30)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   ax.set_title(r'$\Delta P$ [MPa]'+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_dotlam(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  scal = 1e-14

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.dotlam*scal_v_ms/scal_x_m/scal
  # scal = max(np.amax(A.dotlam),-np.amin(A.dotlam))
  # X = A.dotlam/scal
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap=cm.davos_r,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(0,1)
  im.set_clim(0,0.15)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   # ax.set_title(r'$\dot{\lambda}$ [$\times 10^{-14}$ 1/s]'+' ts = '+str(A.nd.istep))
  #   ax.set_title(r'$\dot{\lambda}$ [-]'+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_lam(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)

  scal = 1
  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.lam*scal
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap=cm.davos_r,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(0,0.0125)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   ax.set_title(r'$\lambda$ [-]'+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_etaeff(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'eta',dim,0)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = np.log10(A.matProp.eta*scal)
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(np.log10(A.nd.eta_min*scal)+3,np.log10(A.nd.eta_max*scal)-4.5)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    ax.set_title(r'$\log_{10}\eta_{eff}$ [Pa.s]'+' ts = '+str(A.nd.istep))
  else:
    ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_zetaeff(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'eta',dim,0)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = np.log10(A.matProp.zeta*scal)
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  # im.set_clim(np.log10(A.nd.eta_min*scal)+3,np.log10(A.nd.eta_max*scal)-4.5)
  im.set_clim(19,20.5)
  
  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    ax.set_title(r'$\log_{10}\zeta_{eff}$ [Pa.s]'+' ts = '+str(A.nd.istep))
  else:
    ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_PV(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scalP = get_scaling(A,'P',dim,1)
  scalv = get_scaling(A,'v',dim,1)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.P*scalP
  t = A.nd.t*scalt
  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  divnorm=colors.TwoSlopeNorm(vmin=-200, vcenter=0., vmax=100)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='seismic',origin='lower',norm=divnorm)
  im.set_clim(-200,100)

  maxV = 1
  nind = 10 # 10 for high res/ 5 for low res
  Q  = ax.quiver(A.grid.xc[istart:iend:nind]*scalx, A.grid.zc[jstart:jend:nind]*scalx, A.Vscx[jstart:jend:nind,istart:iend:nind]*scalv/maxV, A.Vscz[jstart:jend:nind,istart:iend:nind]*scalv/maxV, 
      color='black', scale_units='xy', scale=0.25, units='width', pivot='tail', width=0.008, headwidth=5, headaxislength=5, minlength=0)
  
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)

  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    ax.set_title(r'$P$ [MPa]'+' ts = '+str(A.nd.istep))
  else:
    ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_Vfx(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scalv = get_scaling(A,'v',dim,1)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.Vfx*scalv
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  # divnorm=colors.TwoSlopeNorm(vmin=-200, vcenter=0., vmax=100)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='seismic',origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-10,10)

  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    ax.set_title(r'$V_\ell^x$ [cm/yr]'+' ts = '+str(A.nd.istep))
  else:
    ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_Vfz(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scalv = get_scaling(A,'v',dim,1)

  extentC=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  X = A.Vfz*scalv
  t = A.nd.t*scalt
  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  divnorm=colors.TwoSlopeNorm(vmin=-5, vcenter=0., vmax=10)
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentC,cmap='seismic',origin='lower',norm=divnorm)
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  im.set_clim(-5,10)

  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    ax.set_title(r'$V_\ell^z$ [cm/yr]'+' ts = '+str(A.nd.istep))
  else:
    ax.set_title('ts = '+str(A.nd.istep))
  
  return im

# ---------------------------------
def plot_8ind_epsVEVP_II(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis,iplot):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal = get_scaling(A,'eps',dim,0)

  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]
  
  if (iplot==0):
    # X = A.rheol.epsV_II*scal/scal0
    scal0 = np.amax(A.rheol.epsV_II)
    X = A.rheol.epsV_II/scal0
  elif (iplot==1):
    # X = A.rheol.epsV_II*scal/scal0
    scal0 = np.amax(A.rheol.epsE_II)
    X = A.rheol.epsE_II/scal0
  else:
    # X = A.rheol.epsVP_II*scal/scal0
    scal0 = np.amax(A.rheol.epsVP_II)
    X = A.rheol.epsVP_II/scal0
  t = A.nd.t*scalt
  markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentV,cmap=cm.davos_r,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)

  im.set_clim(0,1)

  # if (iplot==0):
  #   im.set_clim(1e-16,5e-14)
  # elif (iplot==1):
  #   im.set_clim(1e-16,1e-14)
  # else:
  #   im.set_clim(1e-16,1e-13)

  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    if (iplot==0):
      # ax.set_title(r'$\dot{\epsilon}_{II}^V$ [$\times 10^{-16}$ 1/s]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
      ax.set_title(r'$\dot{\epsilon}_{II}^V$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
    elif (iplot==1):
      # ax.set_title(r'$\dot{\epsilon}_{II}^E$ [$\times 10^{-16}$ 1/s]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
      ax.set_title(r'$\dot{\epsilon}_{II}^E$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
    else:
      # ax.set_title(r'$\dot{\epsilon}_{II}^{VP}$ [$\times 10^{-16}$ 1/s]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
      ax.set_title(r'$\dot{\epsilon}_{II}^{VP}$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
  else:
    ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')
  
  return im

# ---------------------------------
def plot_8ind_C_VEVP(fig,ax,A1,istart,iend,jstart,jend,xaxis,yaxis,iplot):

  A = A1
  dim = A.dimensional
  scalx = get_scaling(A1,'x',dim,1)
  lblx = get_label(A1,'x',dim)
  lblz = get_label(A1,'z',dim)
  scalt = get_scaling(A1,'t',dim,1)
  scal_v_ms = get_scaling(A,'v',dim,0)
  scal_x_m = get_scaling(A,'x',dim,0)
  # scal = 1e-6 #1e-14

  extentV=[min(A.grid.xv[istart:iend])*scalx, max(A.grid.xv[istart:iend])*scalx, min(A.grid.zv[jstart:jend])*scalx, max(A.grid.zv[jstart:jend])*scalx]
  
  # if (iplot==0):
  #   X = A.rheol.volV*scal_v_ms/scal_x_m/scal
  # elif (iplot==1):
  #   X = A.rheol.volE*scal_v_ms/scal_x_m/scal
  # else:
  #   X = A.rheol.volVP*scal_v_ms/scal_x_m/scal
  # if (iplot==0):
  #   X = np.log10(A.rheol.volV*scal_v_ms/scal_x_m/scal)
  # elif (iplot==1):
  #   X = np.log10(A.rheol.volE*scal_v_ms/scal_x_m/scal)
  # else:
  #   X = np.log10(A.rheol.volVP*scal_v_ms/scal_x_m/scal)
  if (iplot==0):
    scal = max(np.amax(A.rheol.volV),-np.amin(A.rheol.volV))
    X = A.rheol.volV/scal
  elif (iplot==1):
    scal = max(np.amax(A.rheol.volE),-np.amin(A.rheol.volE))
    X = A.rheol.volE/scal
  else:
    scal = max(np.amax(A.rheol.volVP),-np.amin(A.rheol.volVP))
    X = A.rheol.volVP/scal

  t = A.nd.t*scalt

  # markx = A.mark.x[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  # markz = A.mark.z[(A.mark.id==0) & (A.mark.x>=A.grid.xv[istart]) & (A.mark.x<=A.grid.xv[iend]) & (A.mark.z>=A.grid.zv[jstart]) & (A.mark.z<=A.grid.zv[jend])]
  
  im = ax.imshow(X[jstart:jend  ,istart:iend  ],extent=extentV,cmap=cm.broc,origin='lower')
  # im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)

  # if (iplot==0):
  #   im.set_clim(-0.1,0.1)
  # elif (iplot==1):
  #   im.set_clim(-0.1,0.1)
  # else:
  #   im.set_clim(-0.1,0.1)

  im.set_clim(-0.5,0.5)

  ax.set_xlim([min(A.grid.xv[istart:iend  ])*scalx,max(A.grid.xv[istart:iend  ])*scalx])
  ax.set_ylim([min(A.grid.zv[jstart:jend  ])*scalx,max(A.grid.zv[jstart:jend  ])*scalx])

  if (xaxis):
    ax.set_xlabel(lblx)
    
  if (yaxis):
    ax.set_ylabel(lblz)
    
  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  # if (yaxis):
  #   if (iplot==0):
  #     ax.set_title(r'$\mathcal{C}^V$ [$\times 10^{-14}$ 1/s]'+' ts = '+str(A.nd.istep))
  #   elif (iplot==1):
  #     ax.set_title(r'$\mathcal{C}^E$ [$\times 10^{-14}$ 1/s]'+' ts = '+str(A.nd.istep))
  #   else:
  #     ax.set_title(r'$\mathcal{C}^{VP}$ [$\times 10^{-14}$ 1/s]'+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))
  
  # if (yaxis):
  #   if (iplot==0):
  #     ax.set_title(r'log10 $\mathcal{C}^V$ [1/s]'+' ts = '+str(A.nd.istep))
  #   elif (iplot==1):
  #     ax.set_title(r'log10 $\mathcal{C}^E$ [1/s]'+' ts = '+str(A.nd.istep))
  #   else:
  #     ax.set_title(r'log10 $\mathcal{C}^{VP}$ [1/s]'+' ts = '+str(A.nd.istep))
  # else:
  #   ax.set_title('ts = '+str(A.nd.istep))

  # ax.set_title('ts = '+str(A.nd.istep)+'  t = '+str(round(t/1.0e3,0))+' [kyr]')
  if (yaxis):
    if (iplot==0):
      ax.set_title(r'$\mathcal{C}^V$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
    elif (iplot==1):
      ax.set_title(r'$\mathcal{C}^E$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
    else:
      ax.set_title(r'$\mathcal{C}^{VP}$ [-]'+' t = '+str(round(t/1.0e3,0))+' [kyr]')
  else:
    ax.set_title('t = '+str(round(t/1.0e3,0))+' [kyr]')

  return im

# ---------------------------------
def plot_8multiple_phi(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # Porosity
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'log$_{10}\phi$')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3panels_phi_eps_div(A,istart,iend,jstart,jend,fdir,fname,tstep):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(9,4))
  nx = 3
  nz = 1
  iplot = 0

  # Porosity
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A,istart,iend,jstart,jend,1,1)
  cbar = fig.colorbar(im,ax=ax, shrink=0.65)
  cbar.ax.set_title(r'log$_{10}\phi$')
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII_normalised(fig,ax,A,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  ax.set_title(' ')
  cbar = fig.colorbar(im,ax=ax, shrink=0.65)
  # cbar.ax.set_title(r'log$_{10}\dot{\epsilon}_{II}$ [1/s]')
  cbar.ax.set_title(r'Normalised $\dot{\epsilon}_{II}$')
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs_normalised(fig,ax,A,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  ax.set_title(' ')
  cbar = fig.colorbar(im,ax=ax, shrink=0.65)
  # cbar.ax.set_title(r'$\mathcal{C}$ [$\times 10^{-14}$ 1/s]')
  cbar.ax.set_title(r'Normalised $\mathcal{C}$')
  
  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_phi(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # Porosity
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_phi(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'log$_{10}\phi$')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_epsII(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # epsII
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  # lblII  = get_label(A1,'epsII',A1.dimensional)
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)
  # cbar.ax.set_title(lblII)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_epsII(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # epsII
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsII(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'log$_{10}\dot{\epsilon}_{II}$ [1/s]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_divvs(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # divvs
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  # lblII  = get_label(A1,'epsII',A1.dimensional)
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)
  # cbar.ax.set_title(lblII)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_divvs(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # divvs
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_divvs(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'$\mathcal{C}$ [$\times 10^{-14}$ 1/s]')

  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_tauII(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # tauII
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_tauII(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # tauII
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_tauII(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  # cbar.ax.set_title(r'$\tau_{II}$ [MPa]')
  cbar.ax.set_title(r'log$_{10}\tau_{II}$ [MPa]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_DP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # DP
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_DP(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # DP
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_DP(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'$\Delta P$ [MPa]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_dotlam(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # dotlam
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_dotlam(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # dotlam
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_dotlam(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'$\dot{\lambda}$ [$\times 10^{-14}$ 1/s]')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_lam(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # lam
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_lam(A1,A2,A3,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  # lam
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A2,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_lam(fig,ax,A3,istart,iend,jstart,jend,1,0)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.ax.set_title(r'$\lambda [-]$')

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_etaeff(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # etaeff
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_etaeff(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_zetaeff(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # zetaeff
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_zetaeff(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_PV(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # dynamic P,V
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_PV(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_Vfx(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # Vfx
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfx(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_Vfz(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  # Vfz
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A1,istart,iend,jstart,jend,1,1)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A2,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A3,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A4,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A5,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A6,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A7,istart,iend,jstart,jend,1,0)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_Vfz(fig,ax,A8,istart,iend,jstart,jend,1,0)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_epsVEVP_II(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname,ii):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A1,istart,iend,jstart,jend,1,1,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A2,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A3,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A4,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A5,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A6,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A7,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A8,istart,iend,jstart,jend,1,0,ii)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_epsVEVP_II(A1,A2,A3,istart,iend,jstart,jend,fdir,fname,ii):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A1,istart,iend,jstart,jend,1,1,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A2,istart,iend,jstart,jend,1,0,ii)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_epsVEVP_II(fig,ax,A3,istart,iend,jstart,jend,1,0,ii)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_8multiple_C_VEVP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,fdir,fname,ii):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(21,4))
  nx = 8
  nz = 1
  iplot = 0

  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A1,istart,iend,jstart,jend,1,1,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A2,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A3,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A4,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A5,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A6,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A7,istart,iend,jstart,jend,1,0,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A8,istart,iend,jstart,jend,1,0,ii)
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.91, 0.25, 0.007, 0.50])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_3multiple_C_VEVP(A1,A2,A3,istart,iend,jstart,jend,fdir,fname,ii):

  make_dir(fdir)
  
  fig = plt.figure(1,figsize=(6.5,4))
  nx = 3
  nz = 1
  iplot = 0

  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A1,istart,iend,jstart,jend,1,1,ii)
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A2,istart,iend,jstart,jend,1,0,ii)
  ax.set(yticklabels=[])
  
  iplot = iplot+1
  ax = plt.subplot(nz,nx,iplot)
  im = plot_8ind_C_VEVP(fig,ax,A3,istart,iend,jstart,jend,1,0,ii)
  ax.set(yticklabels=[])
  
  fig.subplots_adjust(bottom=0.25, right=0.9, top=0.75)
  cbar_ax = fig.add_axes([0.94, 0.27, 0.02, 0.46])
  cbar = fig.colorbar(im, cax=cbar_ax)

  # plt.tight_layout()
  plt.savefig(fdir+fname+'.png', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_compaction_length(A,istart,iend,jstart,jend,fdir,fname,istep,dim):
  make_dir(fdir)
  fig = plt.figure(1,figsize=(7,5))

  markx = A.mark.x[A.mark.id==0]
  markz = A.mark.z[A.mark.id==0]

  scal_eta = get_scaling(A,'eta',dim,0)
  scal_K = get_scaling(A,'Kphi',dim,0)
  mu = 1.0

  delta = ((A.matProp.zetaV+4.0/3.0*A.matProp.etaV)*scal_eta*scal_K/mu)**0.5
  print('log10(delta): min = '+str(np.min(np.log10(delta)))+' max = '+str(np.max(np.log10(delta))))

  scalx = get_scaling(A,'x',dim,1)
  lblx = get_label(A,'x',dim)
  lblz = get_label(A,'z',dim)
  extentE=[min(A.grid.xc[istart:iend])*scalx, max(A.grid.xc[istart:iend])*scalx, min(A.grid.zc[jstart:jend])*scalx, max(A.grid.zc[jstart:jend])*scalx]

  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.log10(delta[jstart:jend  ,istart:iend  ]),extent=extentE,origin='lower')
  im2 = ax.scatter(markx*scalx,markz*scalx,c='w',s=0.5,linewidths=None)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  ax.axis('image')
  ax.set_xlabel(lblx)
  ax.set_ylabel(lblz)
  ax.set_title(r'log$_{10}\delta$ [m]'+' tstep = '+str(istep))

  # plt.tight_layout() 
  plt.savefig(fdir+fname+'_dim'+str(dim)+'.png', bbox_inches = 'tight')
  plt.close()
