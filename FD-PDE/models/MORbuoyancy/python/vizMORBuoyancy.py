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

# ---------------------------------
# Definitions
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
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    xv = data['x1d_vertex']
    zv = data['y1d_vertex']
    vxr= data['X_face_x']
    vzr= data['X_face_y']
    pr = data['X_cell']

    # Reshape data in 2D
    P  = pr.reshape(nz,nx)
    Vx = vxr.reshape(nz,nx+1)
    Vz = vzr.reshape(nz+1,nx)

    return P,Vx,Vz,nx,nz,xc,zc,xv,zv
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

    return H,C,nx,nz,xc,zc
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

    return Vfx,Vfz,Vbx,Vbz,nx,nz,xc,zc,xv,zv
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
    H  = Hr.reshape(nz,nx)
    T  = Tr.reshape(nz,nx)
    TP = TPr.reshape(nz,nx)
    phi= phir.reshape(nz,nx)
    P  = Pr.reshape(nz,nx)
    C  = Cr.reshape(nz,nx)
    Cs = Csr.reshape(nz,nx)
    Cf = Cfr.reshape(nz,nx)

    return H,T,TP,phi,P,C,Cs,Cf,nx,nz,xc,zc
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
    eta  = etar.reshape(nz,nx)
    zeta = zetar.reshape(nz,nx)
    K    = Kr.reshape(nz,nx)
    rho  = rhor.reshape(nz,nx)
    rhof = rhofr.reshape(nz,nx)
    rhos = rhosr.reshape(nz,nx)

    return eta,zeta,K,rho,rhof,rhos
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
    A_cor = data_v.reshape(nz+1,nx+1)
    C = Cr.reshape(nz,nx)
    A = Ar.reshape(nz,nx)
    D1= D1r.reshape(nz,nx)

    Bx  = Bxr.reshape(nz,nx+1)
    D2x = D2xr.reshape(nz,nx+1)
    D3x = D3xr.reshape(nz,nx+1)

    Bz  = Bzr.reshape(nz+1,nx)
    D2z = D2zr.reshape(nz+1,nx)
    D3z = D3zr.reshape(nz+1,nx)

    return A_cor,A,C,D1,Bx,Bz,D2x,D2z,D3x,D3z
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
    A1 = A1r.reshape(nz,nx)
    B1 = B1r.reshape(nz,nx)
    D1 = D1r.reshape(nz,nx)
    A2 = A2r.reshape(nz,nx)
    B2 = B2r.reshape(nz,nx)
    D2 = D2r.reshape(nz,nx)

    C1x = C1xr.reshape(nz,nx+1)
    C2x = C2xr.reshape(nz,nx+1)
    vx  = vxr.reshape(nz,nx+1)
    vfx = vfxr.reshape(nz,nx+1)
    vsx = vsxr.reshape(nz,nx+1)

    C1z = C1zr.reshape(nz+1,nx)
    C2z = C2zr.reshape(nz+1,nx)
    vz  = vzr.reshape(nz+1,nx)
    vfz = vfzr.reshape(nz+1,nx)
    vsz = vszr.reshape(nz+1,nx)

    return A1,B1,D1,A2,B2,D2,C1x,C1z,C2x,C2z,vx,vz,vfx,vfz,vsx,vsz
  except OSError:
    print('Cannot open: '+fdir+'/'+fname+'.py')
    return 0.0

# ---------------------------------
def plot_PV(P,Vx,Vz,nx,nz,xc,zc,xv,zv,scalP,scalv,scalx,lbl_P,lbl_v,lbl_x,fname,istep,framex,framez):

  fig = plt.figure(1,figsize=(framex,framez))
  nind = 4

  vxc  = np.zeros([nz,nx])
  vzc  = np.zeros([nz,nx])

  for i in range(0,nx):
    for j in range(0,nz):
      vxc[j][i]  = 0.5 * (Vx[j][i+1] + Vx[j][i])
      vzc[j][i]  = 0.5 * (Vz[j+1][i] + Vz[j][i])

  ax = plt.subplot(3,1,1)
  im = ax.imshow( P*scalP, extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],
                  origin='lower', cmap='ocean', interpolation='nearest')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl_P)
  Q  = ax.quiver( xc[::nind]*scalx, zc[::nind]*scalx, vxc[::nind,::nind]*scalv, vzc[::nind,::nind]*scalv, color='grey', units='width', pivot='mid')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))
  # ax.set_title(r'a) $P$ tstep = '+str(istep))

  ax = plt.subplot(3,1,2)
  im = ax.imshow(Vx*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(label=lbl_v+' x')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  # ax.set_title(r'b) $V_s^x$')

  ax = plt.subplot(3,1,3)
  im = ax.imshow(Vz*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(label=lbl_v+' z')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  # ax.set_title(r'c) $V_s^z$')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HC(H,C,nx,nz,xc,zc,scalH,scalC,scalx,lbl_H,lbl_C,lbl_x,fname,istep,framex,framez):

  fig = plt.figure(1,figsize=(framex,framez))

  ax = plt.subplot(2,1,1)
  im = ax.imshow(H*scalH,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl_H)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  # ax.set_title(r'a) $H$ tstep = '+str(istep))
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(2,1,2)
  im = ax.imshow(C,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title(lbl_C)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  # ax.set_title(r'b) $\Theta$ ')

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Enth(H,T,TP,phi,P,C,Cs,Cf,nx,nz,xc,zc,scalH,scalT,scalC,scalx,lbl_H,lbl_T,lbl_TP,lbl_Plith,lbl_C,lbl_Cf,lbl_Cs,lbl_phi,lbl_x,fname,istep):

  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(4,2,1)
  im = ax.imshow(H*scalH,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_H)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(C,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_C)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(T-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_T)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(Cf,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_Cf)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(TP-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_TP)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(Cs,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_Cs)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,7)
  im = ax.imshow(P,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_Plith)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,8)
  im = ax.imshow(phi,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_phi)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_Vel(Vfx,Vfz,Vbx,Vbz,nx,nz,xc,zc,xv,zv,scalv,scalx,lbl_vf,lbl_v,lbl_x,fname,istep):

  fig = plt.figure(1,figsize=(14,7))

  ax = plt.subplot(2,2,1)
  im = ax.imshow(Vfx*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_vf+' x')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(Vfz*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_vf+' z')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(2,2,3)
  im = ax.imshow(Vbx*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_v+' x')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(2,2,4)
  im = ax.imshow(Vbz*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_v+' z')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_PVcoeff(A_cor,A,C,D1,Bx,Bz,D2x,D2z,D3x,D3z,nx,nz,xc,zc,xv,zv,scalx,lbl_x,fname,istep):

  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A_cor,extent=[min(xv)*scalx, max(xv)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A corner')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(C,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(D1,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(Bx,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('Bx face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(Bz,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('Bz face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,7)
  im = ax.imshow(D2x,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2x face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,8)
  im = ax.imshow(D2z,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2z face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,9)
  im = ax.imshow(D3x,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D3x face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,10)
  im = ax.imshow(D3z,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D3z face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_HCcoeff(A1,B1,D1,A2,B2,D2,C1x,C1z,C2x,C2z,vx,vz,vfx,vfz,vsx,vsz,nx,nz,xc,zc,xv,zv,scalx,lbl_x,fname,istep):

  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(5,2,1)
  im = ax.imshow(A1,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A1 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(5,2,2)
  im = ax.imshow(A2,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('A2 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(B1,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('B1 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(B2,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('B2 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(D1,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D1 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(D2,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('D2 center')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,7)
  im = ax.imshow(C1x,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1x face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,8)
  im = ax.imshow(C1z,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1z face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,9)
  im = ax.imshow(C2x,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1x face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,10)
  im = ax.imshow(C2z,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('C1z face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'_part1.pdf', bbox_inches = 'tight')
  plt.close()

  # part 2
  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(5,2,1)
  im = ax.imshow(vx,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vx face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,2)
  im = ax.imshow(vz,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vz face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,3)
  im = ax.imshow(vfx,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vfx face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,4)
  im = ax.imshow(vfz,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vfz face')
  ax.axis('image')
  # ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,5)
  im = ax.imshow(vsx,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vsx face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(5,2,6)
  im = ax.imshow(vsz,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.85)
  cbar.ax.set_title('vsz face')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'_part2.pdf', bbox_inches = 'tight')
  plt.close()

# ---------------------------------
def plot_matProp(eta,zeta,K,rho,rhof,rhos,nx,nz,xc,zc,scaleta,scalK,scalrho,scalx,lbl_eta,lbl_zeta,lbl_K,lbl_rho,lbl_x,fname,istep):

  fig = plt.figure(1,figsize=(14,14))

  ax = plt.subplot(4,2,1)
  im = ax.imshow(np.log10(eta*scaleta),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title('log10 '+lbl_eta)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)
  ax.set_title('tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(np.log10(zeta*scaleta),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title('log10 '+lbl_zeta)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,3)
  im = ax.imshow(np.log10(K*scalK),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title('log10 '+lbl_K)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,4)
  im = ax.imshow(rho*scalrho,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_rho)
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,5)
  im = ax.imshow(rhof*scalrho,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_rho+' (f)')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  ax = plt.subplot(4,2,6)
  im = ax.imshow(rhos*scalrho,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60)
  cbar.ax.set_title(lbl_rho+' (s)')
  ax.axis('image')
  ax.set_xlabel('x '+lbl_x)
  ax.set_ylabel('z '+lbl_x)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')
  plt.close()