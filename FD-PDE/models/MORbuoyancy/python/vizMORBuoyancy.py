# ---------------------------------
# Load modules
# ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
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
def get_scaling_labels(A,fname,fdir,dim):
  try: 
    # Load parameters file

    # Create data object
    scal = EmptyStruct()
    lbl  = EmptyStruct()
    SEC_YEAR = 31536000

    # non-dimensional - default
    scal.x  = 1
    scal.P  = 1
    scal.v  = 1 
    scal.H  = 1
    scal.phi= 1
    scal.C  = 0
    scal.T  = 0
    scal.eta= 1
    scal.K  = 1
    scal.rho= 1
    scal.Gamma= 1

    # Units and labels - default
    lbl.x = r'x/H [-]'
    lbl.z = r'z/H [-]'
    lbl.P = r'$P$ [-]'
    lbl.vs= r'$V_s$ [-]'
    lbl.vsx= r'$V_s^x$ [-]'
    lbl.vsz= r'$V_s^z$ [-]'
    lbl.vf= r'$V_f$ [-]'
    lbl.vfx= r'$V_f^x$ [-]'
    lbl.vfz= r'$V_f^z$ [-]'
    lbl.v = r'$V$ [-]'
    lbl.vx= r'$V^x$ [-]'
    lbl.vz= r'$V^z$ [-]'
    lbl.C = r'$\Theta$ [-]'
    lbl.H = r'$H$ [-]'
    lbl.phi = r'$\phi$ [-]'
    lbl.T = r'$\tilde{\theta}$ [-]'
    lbl.TP = r'$\theta$ [-]'
    lbl.Cf = r'$\Theta_f$ [-]'
    lbl.Cs = r'$\Theta_s$ [-]'
    lbl.Plith = r'$P_{lith}$ [-]'
    lbl.resP = r'res $P$ [-]'
    lbl.resvsx= r'res $V_s^x$ [-]'
    lbl.resvsz= r'res $V_s^z$ [-]'
    lbl.resC = r'res $\Theta$ [-]'
    lbl.resH = r'res $H$ [-]'
    lbl.eta = r'$\eta$ [-]'
    lbl.zeta = r'$\zeta$ [-]'
    lbl.K = r'$K$ [-]'
    lbl.rho = r'$\rho$ [-]'
    lbl.rhof = r'$\rho_f$ [-]'
    lbl.rhos = r'$\rho_s$ [-]'
    lbl.Gamma = r'$\Gamma$ [-]'
    lbl.divmass = r'$\nabla\cdot(v)$ [-]'

    if (dim == 1):
      scal.x  = A.H*1e2 # h0/1e3 km
      scal.P  = 1e-6 # MPa
      scal.v  = 1.0e2*SEC_YEAR # cm/yr
      scal.T  = 273.15 # deg C
      scal.Gamma= 1000*SEC_YEAR # g/m3/yr

      lbl.P = r'$P$ [MPa]'
      lbl.x = r'x [km]'
      lbl.z = r'z [km]'      
      lbl.vs= r'$V_s$ [cm/yr]'
      lbl.vsx= r'$V_s^x$ [cm/yr]'
      lbl.vsz= r'$V_s^z$ [cm/yr]'
      lbl.vf= r'$V_f$ [cm/yr]'
      lbl.vfx= r'$V_f^x$ [cm/yr]'
      lbl.vfz= r'$V_f^z$ [cm/yr]'
      lbl.v = r'$V$ [cm/yr]'
      lbl.vx= r'$V^x$ [cm/yr]'
      lbl.vz= r'$V^z$ [cm/yr]'
      lbl.H = r'$H$ [J/kg]'
      lbl.C = r'$C$'
      lbl.Cf = r'$C_f$'
      lbl.Cs = r'$C_s$'
      lbl.T = r'$T$ $[^oC]$'
      lbl.TP = r'$T$ potential $[^oC]$'
      lbl.Plith = r'$P_{lith}$ [MPa]'
      lbl.phi = r'$\phi$ '
      lbl.resP = r'res $P$ [MPa]'
      lbl.resvsx= r'res $V_s^x$ [cm/yr]'
      lbl.resvsz= r'res $V_s^z$ [cm/yr]'
      lbl.resC = r'res $C$'
      lbl.resH = r'res $H$ [J/kg]'
      lbl.eta = r'$\eta$ [Pa.s]'
      lbl.zeta = r'$\zeta$ [Pa.s]'
      lbl.K = r'$K$ [m2]'
      lbl.rho = r'$\rho$ [kg/m3]'
      lbl.rhof = r'$\rho_f$ [kg/m3]'
      lbl.rhos = r'$\rho_s$ [kg/m3]'
      lbl.Gamma = r'$\Gamma$ [g/m$^3$/yr]'
      lbl.divmass = r'$\nabla\cdot v$ [/s]'

    # set clim 

    return scal, lbl
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------------
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

# ---------------------------------------
def parse_log_file(fname):
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

    # variables - sill
    sill = EmptyStruct()
    sill.t = np.zeros(tstep+1)
    sill.F = np.zeros(tstep+1)
    sill.C = np.zeros(tstep+1)
    sill.h = np.zeros(tstep+1)

    # Convergence 
    sol = EmptyStruct()
    sol.HCres = np.zeros(tstep+1)
    sol.PVres = np.zeros(tstep+1)
    sol.dt = np.zeros(tstep+1)

    # Parse output and save norm info
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
      
      if (i0<tstep):
        if 'SILL FLUXES:' in line:
          sill.t[i0+1] = float(line[19:37])
          sill.F[i0+1] = float(line[82:101])
          sill.C[i0+1] = float(line[47:66])
          sill.h[i0+1] = float(line[121:140])
        
        if 'Nonlinear hc_ solve' in line:
          sol.HCres[i0+1] = float(line_prev[23:41])
        if 'Nonlinear pv_ solve' in line:
          sol.PVres[i0+1] = float(line_prev[23:41])

        if '# TIME:' in line:
          sol.dt[i0+1] = float(line[44:62])

        line_prev = line

    f.close()

    return tstep, sill, sol
  except OSError:
    print('Cannot open:', fname)
    return 0, 0, 0

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
def parse_HC_file(fname,fdir):
  try: 
    spec = importlib.util.spec_from_file_location(fname,fdir+'/'+fname+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()

    # Split data
    nx = data['Nx'][0]
    nz = data['Ny'][0]
    xc = data['x1d_cell']
    zc = data['y1d_cell']
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    xv = data['x1d_vertex']
    zv = data['y1d_vertex']
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    xv = data['x1d_vertex']
    zv = data['y1d_vertex']
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
    coeff.A_cor = data_v.reshape(nz+1,nx+1)
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    xv = data['x1d_vertex']
    zv = data['y1d_vertex']
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
def plot_PV(iplot,A,fname,istep,framex,framez):

  fig = plt.figure(1,figsize=(framex,framez))

  if (iplot==0):
    X1 = A.P*A.scal.P
    X2 = A.Vsx*A.scal.v
    X3 = A.Vsz*A.scal.v
    lbl1 = A.lbl.P
    lbl2 = A.lbl.vsx
    lbl3 = A.lbl.vsz
    # im.set_clim(-100,0)
    # im.set_clim(0,2.5)
    # im.set_clim(-1,6)

  if (iplot==1):
    X1 = A.resP*A.scal.P
    X2 = A.resVsx*A.scal.v
    X3 = A.resVsz*A.scal.v
    lbl1 = A.lbl.resP
    lbl2 = A.lbl.resvsx
    lbl3 = A.lbl.resvsz

  extentP =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(3,1,1)
  im = ax.imshow(X1,extent=extentP,cmap='viridis',origin='lower',interpolation='nearest')
  xa = A.grid.xc[::4]*A.scal.x
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[0]*A.scal.x])
  stream = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, A.Vscx*A.scal.v, A.Vscz*A.scal.v,color='k',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(lbl1)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(lbl1+' tstep = '+str(istep))

  ax = plt.subplot(3,1,2)
  im = ax.imshow(X2,extent=extentVx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(lbl2)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(lbl2)

  ax = plt.subplot(3,1,3)
  im = ax.imshow(X3,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(lbl3)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(lbl3)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HC(iplot,A,fname,istep,framex,framez):

  fig = plt.figure(1,figsize=(framex,framez))
  extentC=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]

  if (iplot==0):
    X1 = A.H*A.scal.H
    X2 = A.C
    lbl1 = A.lbl.H
    lbl2 = A.lbl.C
    # im.set_clim(-30,0)
    # im.set_clim(0,0.2)

  if (iplot==1):
    X1 = A.phi
    X2 = A.T-A.scal.T
    lbl1 = A.lbl.phi
    lbl2 = A.lbl.T
    # im.set_clim(0,0.005)
    # im.set_clim(-30,0)
  
  if (iplot==2):
    X1 = A.resH*A.scal.H
    X2 = A.resC
    lbl1 = A.lbl.resH
    lbl2 = A.lbl.resC

  ax = plt.subplot(2,1,1)
  im = ax.imshow(X1,extent=extentC,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  # cbar.ax.set_title(lbl1)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(lbl1+' tstep = '+str(istep))

  ax = plt.subplot(2,1,2)
  im = ax.imshow(X2,extent=extentC,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  # cbar.ax.set_title(lbl2)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(lbl2)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Enth(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentC=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]

  ax = plt.subplot(4,2,1)
  im = ax.imshow(A.Enth.H*A.scal.H,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.H)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.H+' tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(A.Enth.C,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.2)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.C)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.C)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(A.Enth.T-A.scal.T,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.T)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.T)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(A.Enth.Cf,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-1,-0.8)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.Cf)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.Cf)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(A.Enth.TP-A.scal.T,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.TP)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.TP)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(A.Enth.Cs,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.2)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.Cs)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.Cs)

  ax = plt.subplot(4,2,7)
  im = ax.imshow(A.Enth.P*A.scal.P,extent=extentC,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.Plith)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.Plith)

  ax = plt.subplot(4,2,8)
  im = ax.imshow(A.Enth.phi,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.01)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.phi)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.phi)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Vel(A,fname,istep):

  fig = plt.figure(1,figsize=(14,7))
  extentVfx=[min(A.grid.xv[1:])*A.scal.x, max(A.grid.xv[1:])*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(2,2,1)
  im = ax.imshow(A.Vfx[:,1:]*A.scal.v,extent=extentVfx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.vfx)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.vfx+' tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(A.Vfz*A.scal.v,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.vfz)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.vfz)

  ax = plt.subplot(2,2,3)
  im = ax.imshow(A.Vx*A.scal.v,extent=extentVx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.vx)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.vx)

  ax = plt.subplot(2,2,4)
  im = ax.imshow(A.Vz*A.scal.v,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.vz)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.vz)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff(A,fname,istep):

  fig = plt.figure(1,figsize=(14,16))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]
  extentN =[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.PV_coeff.A_cor,extent=extentN,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('A corner')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('A corner tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.PV_coeff.A,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('A center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('A center')

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.PV_coeff.C,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('C center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('C center')

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.PV_coeff.D1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D1 center')

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.PV_coeff.Bx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('Bx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('Bx face')

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.PV_coeff.Bz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('Bz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('Bz face')

  ax = plt.subplot(5,2,7)
  im = ax.imshow(A.PV_coeff.D2x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D2x face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D2x face')

  ax = plt.subplot(5,2,8)
  im = ax.imshow(A.PV_coeff.D2z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D2z face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D2z face')

  ax = plt.subplot(5,2,9)
  im = ax.imshow(A.PV_coeff.D3x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D3x face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D3x face')

  ax = plt.subplot(5,2,10)
  im = ax.imshow(A.PV_coeff.D3z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D3z face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D3z face')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HCcoeff(A,fname,istep):

  fig = plt.figure(1,figsize=(14,16))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]
  extentN =[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.HC_coeff.A1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('A1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('A1 center tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.HC_coeff.A2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('A2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('A2 center')

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.HC_coeff.B1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('B1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('B1 center')

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.HC_coeff.B2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('B2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('B2 center')

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.HC_coeff.D1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D1 center')

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.HC_coeff.D2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('D2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('D2 center')

  ax = plt.subplot(5,2,7)
  im = ax.imshow(A.HC_coeff.C1x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('C1x face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('C1x face')

  ax = plt.subplot(5,2,8)
  im = ax.imshow(A.HC_coeff.C1z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('C1z face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('C1z face')

  ax = plt.subplot(5,2,9)
  im = ax.imshow(A.HC_coeff.C2x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('C1x face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('C1x face')

  ax = plt.subplot(5,2,10)
  im = ax.imshow(A.HC_coeff.C2z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('C1z face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('C1z face')

  plt.savefig(fname+'_part1.pdf', bbox_inches = 'tight')
  plt.close()

  # part 2
  fig = plt.figure(1,figsize=(14,16))

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.HC_coeff.vx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vx face tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.HC_coeff.vz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vz face')

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.HC_coeff.vfx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vfx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vfx face')

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.HC_coeff.vfz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vfz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vfz face')

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.HC_coeff.vsx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vsx face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vsx face')

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.HC_coeff.vsz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.9)
  # cbar.ax.set_title('vsz face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('vsz face')

  plt.savefig(fname+'_part2.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_matProp(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]

  ax = plt.subplot(4,2,1)
  im = ax.imshow(np.log10(A.matProp.eta*A.scal.eta),extent=extentE,cmap='viridis',origin='lower')
  # im.set_clim(-4,6)
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title('log10 '+A.lbl.eta)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('log10 '+A.lbl.eta+' tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(np.log10(A.matProp.zeta*A.scal.eta),extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # im.set_clim(-4,6)
  # cbar.ax.set_title('log10 '+A.lbl.zeta)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('log10 '+A.lbl.zeta)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(A.matProp.K*A.scal.K,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # im.set_clim(-20,0)
  # cbar.ax.set_title(A.lbl.K)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.K)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(A.matProp.rho*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.rho)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.rho)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(A.matProp.rhof*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.rhof)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.rhof)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(A.matProp.rhos*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  # cbar.ax.set_title(A.lbl.rhos)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.rhos)

  ax = plt.subplot(4,2,7)
  im = ax.imshow(A.matProp.Gamma*A.scal.Gamma,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.Gamma)

  # div(mass)
  ax = plt.subplot(4,2,8)
  im = ax.imshow(A.divmass,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.80)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.divmass)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_contours(A,fname,istep):

  fig = plt.figure(1,figsize=(14,5))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  # 1. porosity
  ax = plt.subplot(1,2,1)
  im = ax.imshow(A.phi,extent=extentE,cmap='ocean_r',origin='lower')
  im.set_clim(0,0.002)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  xa = A.grid.xc[::4]*A.scal.x
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[0]*A.scal.x])

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, A.Vscx*A.scal.v, A.Vscz*A.scal.v,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)
  nind = 4
  if (istep>0):
    Q  = ax.quiver(A.grid.xc[::nind]*A.scal.x, A.grid.zc[::nind]*A.scal.x, Vfx[::nind,::nind]*A.scal.v, Vfz[::nind,::nind]*A.scal.v, 
      color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, Vfx*A.scal.v, Vfz*A.scal.v,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  if (A.dim_output):
    levels = [250, 500, 750, 1000, 1200, 1300,]
    fmt = r'%0.0f $^o$C'
  else:
    levels = [-30, -25, -20, -10, -5, -2.5,]
    fmt = r'%0.1f'
  ts = ax.contour(A.grid.xc*A.scal.x, A.grid.zc*A.scal.x, A.T-A.scal.T, levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc*A.scal.x, A.grid.zc*A.scal.x, A.phi, levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')

  if (istep == 0):
    t = 0.0
  else:
    t = A.sill.t[istep-1]
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.phi+' tstep = '+str(istep)+' time = '+str(t)+' [yr]')

  # 2. Vertical solid velocity
  ax = plt.subplot(1,2,2)
  im = ax.imshow(A.Vsz*A.scal.v,extent=extentVz,cmap='viridis',origin='lower')
  if (A.dim_output):
    im.set_clim(0,4.0)
  else:
    im.set_clim(0,2.5)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  # solid streamlines
  stream = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, A.Vscx*A.scal.v, A.Vscz*A.scal.v,
        color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')
  
  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc*A.scal.x, A.grid.zc*A.scal.x, A.phi, levels=[1e-8,], colors = ('k',),linewidths=(0.8,), extend='both')
  
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(A.lbl.vsz)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_temperature_slices(A,fname,istep):

  fig = plt.figure(1,figsize=(4,4))

  ax = plt.subplot(1,1,1)
  pl = ax.plot(A.T[:,1]-A.scal.T, A.grid.zc*A.scal.x, label = 'x = '+str(A.grid.xc[1]*A.scal.x))
  pl = ax.plot(A.T[:,50]-A.scal.T, A.grid.zc*A.scal.x, label = 'x = '+str(A.grid.xc[50]*A.scal.x))
  pl = ax.plot(A.T[:,-1]-A.scal.T, A.grid.zc*A.scal.x, label = 'x = '+str(A.grid.xc[-1]*A.scal.x))
  ax.legend()
  plt.grid(True)
  # plt.xlim(1200, 1450)

  ax.set_xlabel(A.lbl.T)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_sill_outflux(A,fname):

  fig = plt.figure(1,figsize=(10,10))

  ax = plt.subplot(3,1,1)
  pl = ax.plot(A.sill.t[:-1], A.sill.h[:-1]/1000)
  plt.grid(True)
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('Crustal thickness [km]')

  ax = plt.subplot(3,1,2)
  pl = ax.plot(A.sill.t[:-1], A.sill.F[:-1])
  plt.grid(True)
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('Flux out - sill [kg/m/yr]')

  ax = plt.subplot(3,1,3)
  pl = ax.plot(A.sill.t[:-1], A.sill.C[:-1])
  plt.grid(True)
  ax.set_xlabel('Time [yr]')
  ax.set_ylabel('C out - sill [wt. frac.]')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_solver_residuals(A,fname):

  fig = plt.figure(1,figsize=(10,10))

  ax = plt.subplot(3,1,1)
  pl = ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.HCres[1:-1]), linewidth=0.1)
  # pl1 = ax.plot(1420, np.log10(A.sol.HCres[1421]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(HC residual)')

  ax = plt.subplot(3,1,2)
  pl = ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.PVres[1:-1]), linewidth=0.1)
  # pl1 = ax.plot(1420, np.log10(A.sol.PVres[1421]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(PV residual)')

  ax = plt.subplot(3,1,3)
  pl = ax.plot(np.arange(1,A.ts,1), np.log10(A.sol.dt[1:-1]), linewidth=0.1)
  # pl1 = ax.plot(1420, np.log10(A.sol.dt[1421]), 'r*')
  plt.grid(True)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('log10(dt)')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_porosity_solid_stream(A,fname,istep,ext):

  fig = plt.figure(1,figsize=(14,5))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  # 1. porosity
  ax = plt.subplot(1,2,1)
  im = ax.imshow(A.phi,extent=extentE,cmap='ocean_r',origin='lower',label=A.lbl.phi)
  im.set_clim(0,0.002)
  cbar = fig.colorbar(im,ax=ax, shrink=0.50)

  xa = A.grid.xc[::4]*A.scal.x
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[0]*A.scal.x])

  # solid streamlines
  # stream = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, A.Vscx*A.scal.v, A.Vscz*A.scal.v,
  #       color='grey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # fluid streamlines
  mask = np.zeros(A.Vfcx.shape, dtype=bool)
  mask[np.where(A.phi<1e-8)] = True
  Vfx = np.ma.array(A.Vfcx, mask=mask)
  Vfz = np.ma.array(A.Vfcz, mask=mask)
  nind = 4
  # if (istep>0):
  #   Q  = ax.quiver(A.grid.xc[::nind]*A.scal.x, A.grid.zc[::nind]*A.scal.x, Vfx[::nind,::nind]*A.scal.v, Vfz[::nind,::nind]*A.scal.v, 
  #     color='orange', units='width', pivot='mid', width=0.002, headaxislength=3, minlength=0)

  #   stream_fluid = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, Vfx*A.scal.v, Vfz*A.scal.v,
  #       color='r', linewidth=0.5, density=2.0, minlength=0.5, arrowstyle='-')

  # solid streamlines
  stream = ax.streamplot(A.grid.xc*A.scal.x,A.grid.zc*A.scal.x, A.Vscx*A.scal.v, A.Vscz*A.scal.v,
        color='lightgrey',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  # temperature contour
  if (A.dim_output):
    levels = [250, 500, 750, 1000, 1200, 1300,]
    fmt = r'%0.0f $^o$C'
  else:
    levels = [-30, -25, -20, -10, -5, -2.5,]
    fmt = r'%0.1f'
  ts = ax.contour(A.grid.xc*A.scal.x, A.grid.zc*A.scal.x, A.T-A.scal.T, levels=levels,linewidths=(0.8,), extend='both')
  ax.clabel(ts, fmt=fmt, fontsize=8)

  # solidus contour
  if (istep>0):
    cs = ax.contour(A.grid.xc*A.scal.x, A.grid.zc*A.scal.x, A.phi, levels=[1e-8,], colors = ('k',),linewidths=(0.7,), extend='both')

  if (istep == 0):
    t = 0.0
  else:
    t = A.sill.t[istep-1]
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title(' time = %0.1f [yr]' % t)

  plt.savefig(fname+'.'+ext, bbox_inches = 'tight')
  plt.close()