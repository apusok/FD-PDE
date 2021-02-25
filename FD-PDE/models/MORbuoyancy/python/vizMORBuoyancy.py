# ---------------------------------
# Load modules
# ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import importlib

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class EmptyStruct:
  pass

# ---------------------------------
# Definitions
# ---------------------------------
def get_scaling_labels(fname,fdir,dim):
  try: 
    # Load parameters file

    # Create data object
    scal = EmptyStruct()
    lbl  = EmptyStruct()

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

    # if (dim == 1):
    #   scal.x  = 1e2 # km
    #   scal.P  = 1e-9 # GPa
    #   scal.v  = 1.0e2*SEC_YEAR # cm/yr
    #   scal.T  = 273.15 # deg C

    # Units and labels - default
    lbl.x = r'x [-]'
    lbl.z = r'z [-]'
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

    # if (dim == 1):
    #   lbl_P = r'$P [GPa]$'
    #   lbl_v = r'$V [cm/yr]$'
    #   lbl_x = '[km]'
    #   lbl_C = r'$\Theta$'
    #   lbl_H  = r'$H [-]$'
    #   lbl_T = r'$T [^oC]$'

    # set clim 

    return scal, lbl
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

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
    dof = 6

    etar  = Matprops_data[0::dof]
    zetar = Matprops_data[1::dof]
    Kr    = Matprops_data[2::dof]
    rhor  = Matprops_data[3::dof]
    rhofr = Matprops_data[4::dof]
    rhosr = Matprops_data[5::dof]

    # Reshape data in 2D
    matProp = EmptyStruct()
    matProp.eta  = etar.reshape(nz,nx)
    matProp.zeta = zetar.reshape(nz,nx)
    matProp.K    = Kr.reshape(nz,nx)
    matProp.rho  = rhor.reshape(nz,nx)
    matProp.rhof = rhofr.reshape(nz,nx)
    matProp.rhos = rhosr.reshape(nz,nx)

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
  xa = A.grid.xc[::4]
  stream_points = []
  for xi in xa:
    stream_points.append([xi,A.grid.zc[0]])
  stream = ax.streamplot(A.grid.xc,A.grid.zc, A.Vscx, A.Vscz,color='k',linewidth=0.5, start_points=stream_points, density=2.0, minlength=0.5, arrowstyle='-')

  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl1)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(3,1,2)
  im = ax.imshow(X2,extent=extentVx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl2)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(3,1,3)
  im = ax.imshow(X3,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl3)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

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
    X2 = A.T
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
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl1)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(2,1,2)
  im = ax.imshow(X2,extent=extentC,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl2)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Enth(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentC=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]

  ax = plt.subplot(4,2,1)
  im = ax.imshow(A.Enth.H*A.scal.H,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.H)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(A.Enth.C,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.2)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.C)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(A.Enth.T-A.scal.T,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.T)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(A.Enth.Cf,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-1,-0.8)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.Cf)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(A.Enth.TP-A.scal.T,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(-30,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.TP)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(A.Enth.Cs,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.2)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.Cs)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,7)
  im = ax.imshow(A.Enth.P,extent=extentC,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.Plith)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,8)
  im = ax.imshow(A.Enth.phi,extent=extentC,cmap='viridis',origin='lower')
  # im.set_clim(0,0.01)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.phi)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Vel(A,fname,istep):

  fig = plt.figure(1,figsize=(14,7))
  extentVx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentVz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(2,2,1)
  im = ax.imshow(A.Vfx*A.scal.v,extent=extentVx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.vfx)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(A.Vfz*A.scal.v,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.vfz)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(2,2,3)
  im = ax.imshow(A.Vx*A.scal.v,extent=extentVx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.vx)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(2,2,4)
  im = ax.imshow(A.Vz*A.scal.v,extent=extentVz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.vz)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]
  extentN =[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.PV_coeff.A_cor,extent=extentN,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A corner')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.PV_coeff.A,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.PV_coeff.C,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.PV_coeff.D1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.PV_coeff.Bx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('Bx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.PV_coeff.Bz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('Bz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,7)
  im = ax.imshow(A.PV_coeff.D2x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2x face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,8)
  im = ax.imshow(A.PV_coeff.D2z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2z face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,9)
  im = ax.imshow(A.PV_coeff.D3x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D3x face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,10)
  im = ax.imshow(A.PV_coeff.D3z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D3z face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HCcoeff(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFx=[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]
  extentFz=[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]
  extentN =[min(A.grid.xv)*A.scal.x, max(A.grid.xv)*A.scal.x, min(A.grid.zv)*A.scal.x, max(A.grid.zv)*A.scal.x]

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.HC_coeff.A1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.HC_coeff.A2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.HC_coeff.B1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('B1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.HC_coeff.B2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('B2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.HC_coeff.D1,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.HC_coeff.D2,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2 center')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,7)
  im = ax.imshow(A.HC_coeff.C1x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1x face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,8)
  im = ax.imshow(A.HC_coeff.C1z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1z face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,9)
  im = ax.imshow(A.HC_coeff.C2x,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1x face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,10)
  im = ax.imshow(A.HC_coeff.C2z,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1z face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'_part1.pdf', bbox_inches = 'tight')
  plt.close()

  # part 2
  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A.HC_coeff.vx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A.HC_coeff.vz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(A.HC_coeff.vfx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vfx face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(A.HC_coeff.vfz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vfz face')
  ax.axis('image')
  # ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(A.HC_coeff.vsx,extent=extentFx,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vsx face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(A.HC_coeff.vsz,extent=extentFz,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vsz face')
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'_part2.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_matProp(A,fname,istep):

  fig = plt.figure(1,figsize=(14,14))
  extentE =[min(A.grid.xc)*A.scal.x, max(A.grid.xc)*A.scal.x, min(A.grid.zc)*A.scal.x, max(A.grid.zc)*A.scal.x]

  ax = plt.subplot(4,2,1)
  im = ax.imshow(np.log10(A.matProp.eta*A.scal.eta),extent=extentE,cmap='viridis',origin='lower')
  # im.set_clim(-4,6)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title('log10 '+A.lbl.eta)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(np.log10(A.matProp.zeta*A.scal.eta),extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  # im.set_clim(-4,6)
  cbar.ax.set_title('log10 '+A.lbl.zeta)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(A.matProp.K*A.scal.K,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  # im.set_clim(-20,0)
  cbar.ax.set_title(A.lbl.K)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(A.matProp.rho*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.rho)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(A.matProp.rhof*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.rhof)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(A.matProp.rhos*A.scal.rho,extent=extentE,cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(A.lbl.rhos)
  ax.axis('image')
  ax.set_xlabel(A.lbl.x)
  ax.set_ylabel(A.lbl.z)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()