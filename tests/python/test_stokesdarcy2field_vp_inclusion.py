# ---------------------------------------
# Shortening of a visco-(elasto)-plastic block in the absence of gravity
# Setup from T. Gerya, 2018, Ch. 13, ex. 13.2
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

# ---------------------------------------
# Function definitions
# ---------------------------------------
def plot_norm_iteration(fname):
  try: # try to open directory
    # parse number of iterations
    f = open(fname, 'r')
    i0=0
    for line in f:
      if '# SNES SOLVE #' in line:
        i0=0
      if 'SNES Function norm' in line:
        i0+=1
    f.close()
    nit = i0

    nrm = np.zeros(nit)

    # Parse solver info
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# SNES SOLVE #' in line:
        i0=-1
      if 'SNES Function norm' in line:
        i0+=1
        nrm[i0] = float(line[23:43])
    f.close()

    # Plot iterations number
    plt.figure(1,figsize=(6,6))
    plt.grid(color='lightgray', linestyle=':')

    plt.plot(range(0,nit),nrm,'k*-')
    plt.yscale("log")

    plt.ylabel('SNES norm',fontweight='bold',fontsize=12)
    plt.xlabel('it',fontweight='bold',fontsize=12)
    # plt.legend()

    plt.savefig(fname[:-4]+'_nrm_it.pdf')
    plt.close()

  except OSError:
    print('Cannot open:', fname)

def plot_solution(fname,nx,initial,dim):

  # Load data
  if (initial == 0):
    fout = fname+'_solution_initial'
  else:
    fout = fname+'_solution' 
  
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  if (initial == 0):
    fout = fname+'_coefficient_initial'
  else:
    fout = fname+'_coefficient'
  
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  # Prepare cell center velocities
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5

  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))
  if (dim==0):
    etamax = 3
    etamin =-3
  else:
    etamax = 23
    etamin = 17

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(12,8))
  cmaps='RdBu_r' 

  ax = plt.subplot(2,3,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,4)
  im = ax.imshow(vc.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,5)
  im = ax.imshow(np.log10(etac.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=etamin,vmax=etamax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,6)
  im = ax.imshow(np.log10(etan.reshape(mz+1,mx+1)),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=etamin,vmax=etamax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{corner}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  if (initial == 0):
    fout = fname+'_solution_initial'+'_nx_'+str(nx)
  else:
    fout = fname+'_solution'+'_nx_'+str(nx)

  if (dim == 1):
    fout = fout+'_dim' 
  
  plt.savefig(fname+'/'+fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_strain_rates(fname,nx,dim):

  # Load data
  fout = fname+'_strain' # 1. Numerical solution stokes - strain rates
  if (dim == 1):
    fout = fout+'_dim' 

  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  # split into dofs
  dof = 4
  exx1c = eps1c[0::dof]
  ezz1c = eps1c[1::dof]
  exz1c = eps1c[2::dof]
  eII1c = eps1c[3::dof]

  exx1n = eps1n[0::dof]
  ezz1n = eps1n[1::dof]
  exz1n = eps1n[2::dof]
  eII1n = eps1n[3::dof]

  exxmax = max(max(exx1c),max(exx1n))
  exxmin = min(min(exx1c),min(exx1n))
  ezzmax = max(max(ezz1c),max(ezz1n))
  ezzmin = min(min(ezz1c),min(ezz1n))
  exzmax = max(max(exz1c),max(exz1n))
  exzmin = min(min(exz1c),min(exz1n))
  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  # Plot all fields - epsII, epsxx, epszz, epsxz 
  fig = plt.figure(1,figsize=(16,8))
  cmaps='RdBu_r' 

  ax = plt.subplot(2,4,1)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,2)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,3)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,4)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,5)
  im = ax.imshow(exx1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,6)
  im = ax.imshow(ezz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,7)
  im = ax.imshow(exz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,8)
  im = ax.imshow(eII1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_strain'
  if (dim == 1):
    fout = fout+'_dim' 
  fout = fout +'_nx_'+str(nx)
  plt.savefig(fname+'/'+fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_stress(fname,nx,dim):

  # Load data
  fout = fname+'_stress' # 1. Numerical solution stokes - stress components
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']

  fout = fname+'_yield' # yield stress
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  sy1n = data['X_vertex']
  sy1c = data['X_cell']

  # split into dofs
  dof = 4
  sxx1c = s1c[0::dof]
  szz1c = s1c[1::dof]
  sxz1c = s1c[2::dof]
  sII1c = s1c[3::dof]

  sxx1n = s1n[0::dof]
  szz1n = s1n[1::dof]
  sxz1n = s1n[2::dof]
  sII1n = s1n[3::dof]

  syc = sy1c[0::dof]
  syn = sy1n[0::dof]

  sxxmax = max(max(sxx1c),max(sxx1n))
  sxxmin = min(min(sxx1c),min(sxx1n))
  szzmax = max(max(szz1c),max(szz1n))
  szzmin = min(min(szz1c),min(szz1n))
  sxzmax = max(max(sxz1c),max(sxz1n))
  sxzmin = min(min(sxz1c),min(sxz1n))
  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))
  symax = max(max(syc),max(syn))
  symin = min(min(syc),min(syn))

  # Plot all fields 
  fig = plt.figure(1,figsize=(20,8))
  cmaps='RdBu_r' 

  ax = plt.subplot(2,5,1)
  im = ax.imshow(sxx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,2)
  im = ax.imshow(szz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,3)
  im = ax.imshow(sxz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,4)
  im = ax.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,5)
  im = ax.imshow(syc.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=symin,vmax=symax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{yield}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,6)
  im = ax.imshow(sxx1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xx}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,7)
  im = ax.imshow(szz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,8)
  im = ax.imshow(sxz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,9)
  im = ax.imshow(sII1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{II}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,5,10)
  im = ax.imshow(syn.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=symin,vmax=symax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{yield}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_stress'
  if (dim == 1):
    fout = fout+'_dim' 
  fout = fout +'_nx_'+str(nx)
  plt.savefig(fname+'/'+fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_residuals(fname,nx):

  # Load data
  fout = fname+'_residual' 
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(12,4))
  cmaps='RdBu_r' 

  ax = plt.subplot(1,3,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(1,3,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(1,3,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_residual'+'_nx_'+str(nx)
  plt.savefig(fname+'/'+fout+'.pdf')
  plt.close()



def plot_allinone(fname,nx,dim):

  # First row P, Vx, Vz, |V|, eta
  # Load data
  fout = fname+'_solution' 
  
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'
  
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  # Prepare cell center velocities and pressure
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )
  pp = p.reshape(mz,mx)

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  divv= np.zeros( [mz , mx] )
  dx  = np.zeros( [mx] )
  dz  = np.zeros( [mz] )
  dpdx = np.zeros( [mz , mx+1] )
  dpdz = np.zeros( [mz+1 , mx] )
  
  for i in range(0,mx):
    dx[i] = xv[i+1] - xv[i]
  for j in range(0,mz):
    dz[j] = zv[j+1] - zv[j]
  
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      divv[j][i] = (vxface[j][i+1] - vxface[j][i])/dx[i] + (vzface[j+1][i] - vzface[j][i])/dz[j]

      if i != 0:
        dpdx[j][i+1] = pp[j][i] - pp[j][i-1]
      if j != 0:
        dpdz[j+1][i] = pp[j][i] - pp[j-1][i]

  dpdxc = np.zeros([mz,mx])
  dpdzc = np.zeros([mz,mx])

  for i in range(0,mx):
    for j in range(0,mz):
      dpdxc[j][i] = 0.5* (dpdx[j][i+1] + dpdx[j][i])
      dpdzc[j][i] = 0.5* (dpdz[j+1][i] + dpdz[j][i])



  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))
  if (dim==0):
    etamax = 3
    etamin =-3
  else:
    etamax = 23
    etamin = 17

  nind = 10

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(15,9))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,5,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  Q  = ax.quiver( xc[::nind], zc[::nind], -dpdxc[::nind,::nind], -dpdzc[::nind,::nind], units='width', pivot='mid' )
  ax.set_title(r'$P_f, \phi(\mathbf{V}_f-\mathbf{V}_s)$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,4)
  im = ax.imshow(vc.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,5)
  im = ax.imshow(np.log10(etac.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=etamin,vmax=etamax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)


  #Second row: plot epsII, epsxx, epszz, epsxz 
    # Load data
  fout = fname+'_strain' # 1. Numerical solution stokes - strain rates
  if (dim == 1):
    fout = fout+'_dim' 

  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  # split into dofs
  dof = 4
  exx1c = eps1c[0::dof]
  ezz1c = eps1c[1::dof]
  exz1c = eps1c[2::dof]
  eII1c = eps1c[3::dof]

  exx1n = eps1n[0::dof]
  ezz1n = eps1n[1::dof]
  exz1n = eps1n[2::dof]
  eII1n = eps1n[3::dof]

  exxmax = max(max(exx1c),max(exx1n))
  exxmin = min(min(exx1c),min(exx1n))
  ezzmax = max(max(ezz1c),max(ezz1n))
  ezzmin = min(min(ezz1c),min(ezz1n))
  exzmax = max(max(exz1c),max(exz1n))
  exzmin = min(min(exz1c),min(exz1n))
  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  i1 = int(0.15/0.5 * mz)
  i2 = int(0.2/0.5 * mz)
  i3 = int(0.25/0.5 * mz)
  
  divmin = np.amin(divv[i1,:])
  divmax = np.amax(divv)
  if divmin > -0.1:
    divmin = -3


  ax = plt.subplot(3,5,6)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,7)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,8)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,9)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,10)
  im = ax.imshow(divv,extent=[min(xc), max(xc), min(zc), max(zc)],vmin = divmin, vmax = divmax, cmap=cmaps,origin='lower')
  ax.set_title(r'$\nabla \cdot \mathbf{V}_s$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  #Third row: tauII, tauxx, tauzz, tauxz, yield 
    # Load data
  fout = fname+'_stress' # 1. Numerical solution stokes - stress components
  if (dim == 1):
    fout = fout+'_dim' 
  
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']

  fout = fname+'_yield' # yield stress
  if (dim == 1):
    fout = fout+'_dim' 

  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  sy1n = data['X_vertex']
  sy1c = data['X_cell']

  # split into dofs
  dof = 4
  sxx1c = s1c[0::dof]
  szz1c = s1c[1::dof]
  sxz1c = s1c[2::dof]
  sII1c = s1c[3::dof]

  sxx1n = s1n[0::dof]
  szz1n = s1n[1::dof]
  sxz1n = s1n[2::dof]
  sII1n = s1n[3::dof]

  syc = sy1c[0::dof]
  syn = sy1n[0::dof]

  sxxmax = max(max(sxx1c),max(sxx1n))
  sxxmin = min(min(sxx1c),min(sxx1n))
  szzmax = max(max(szz1c),max(szz1n))
  szzmin = min(min(szz1c),min(szz1n))
  sxzmax = max(max(sxz1c),max(sxz1n))
  sxzmin = min(min(sxz1c),min(sxz1n))
  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))
  symax = max(max(syc),max(syn))
  symin = min(min(syc),min(syn))


  ax = plt.subplot(3,5,11)
  im = ax.imshow(sxx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,12)
  im = ax.imshow(szz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,13)
  im = ax.imshow(sxz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,14)
  im = ax.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,15)
  p1 = ax.plot(xc, divv[i1,:], lw = 3, color='k')
  p2 = ax.plot(xc, divv[i2,:], lw = 3, color='r')
  p3 = ax.plot(xc, divv[i3,:], lw = 3, color='b')
  ax.set_title(r'$\nabla \cdot \mathbf{V}_s$ at y = 0.15(k), 0.2(r), 0.25(b)')
  
  #ax = plt.subplot(3,5,15)
  #im = ax.imshow(syc.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=symin,vmax=100,cmap=cmaps,origin='lower')
  #ax.set_title(r'$\tau_{yield}^{CENTER}$')
  #cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  
  fout = fname+'_allinone'+'_nx_'+str(nx)

  if (dim == 1):
    fout = fout+'_dim' 
  
  plt.savefig(fname+'/'+fout+'.pdf')
  plt.close()


def plot_eta_all(fname,num_r,num_c,num_p,phi):

  fout = fname+'_solution' 
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  # Prepare cell center velocities
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5

  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))
  
  etamax = 3
  etamin =-3
     
  ax = plt.subplot(num_r,num_c,num_p)
  im = ax.imshow(np.log10(etac.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=etamin,vmax=etamax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$ $\phi=$'+str(phi))
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)


def plot_divv_all(fname,num_r,num_c,num_p,phi):

  fout = fname+'_solution' 
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  # Prepare cell center velocities
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  divv= np.zeros( [mz , mx] )
  dx  = np.zeros( [mx] )
  dz  = np.zeros( [mz] )
  
  for i in range(0,mx):
    dx[i] = xv[i+1] - xv[i]
  for j in range(0,mz):
    dz[j] = zv[j+1] - zv[j]

  
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      divv[j][i] = (vxface[j][i+1] - vxface[j][i])/dx[i] + (vzface[j+1][i] - vzface[j][i])/dz[j]

  i1 = int(0.15/0.5 * mz)
  i2 = int(0.2/0.5 * mz)
  i3 = int(0.25/0.5 * mz)
  
  divmin = np.amin(divv[i1,:])
  divmax = np.amax(divv)
  if divmin > -0.1:
    divmin = -3
     
  ax = plt.subplot(num_r,num_c,num_p)
  im = ax.imshow(divv,extent=[min(xc), max(xc), min(zc), max(zc)],vmin = divmin, vmax = divmax, cmap=cmaps,origin='lower')
  ax.set_title(r'$\nabla\cdot\mathbf{V}_s$ $\phi=$'+str(phi))
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)


  
  
# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# Shortening of a visco-plastic block in the absence of gravity ')
print('# --------------------------------------- #')

fname = 'out_stokesdarcy2field_vp_inclusion'
try:
  os.mkdir(fname)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Set main parameters and run test
nx    = 100 # resolution

eb = 0.25e3
ew = [1e-1, 1e-3, 1e-1, 1e-3]
cb = [20,  20,  2,  2 ]

phi0 = [1e-2, 5e-2, 1e-1, 2e-1] #, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]

R = 1
m_tri = 2
bulk_eff = 1
lam = 0#10#25
phis = 0#0.2



for i in range(0, 1):
  for j in range(0,len(phi0)) :
    casename = '_case'+str(i+1)
    eta_w = ew[i]
    C_b = cb[i]
    blabel = '_b'+str(bulk_eff)
    pv = '_p'+str(j)

    # filenames 
    fout = fname+'/'+'out_stokesdarcy2field_vp_inclusion'+casename+blabel+pv+'_'+str(nx)+'.out'

    # solver
    picard = ' -p_pc_factor_mat_solver_type umfpack'+ \
             ' -p_pc_type lu'+ \
             ' -p_snes_linesearch_damping 1.0'+ \
             ' -p_snes_linesearch_type bt'+ \
             ' -p_snes_max_it 60 -p_snes_monitor'

    newton = ' -pc_factor_mat_solver_type umfpack'+ \
             ' -pc_type lu'+ \
             ' -snes_linesearch_type bt'+ \
             ' -snes_linesearch_monitor'+ \
             ' -snes_atol 1e-10'+ \
             ' -snes_rtol 1e-50'+ \
             ' -snes_stol 1e-10'+ \
             ' -snes_max_it 200'+ \
             ' -snes_monitor'+ \
             ' -snes_view'+ \
             ' -snes_monitor_true_residual'+ \
             ' -ksp_monitor_true_residual'+ \
             ' -snes_converged_reason'+ \
             ' -ksp_converged_reason'+ \
             ' -python_snes_failed_report'

    solver= ' -snes_mf_operator'

    sdpar = ' -R ' + str(R)+ \
            ' -m ' + str(m_tri)+ \
            ' -phi_0 ' + str(phi0[j])+ \
            ' -bulk_eff ' + str(bulk_eff)+ \
            ' -lambda ' + str(lam) +\
            ' -eta_b ' + str(eb) +\
            ' -phi_s ' + str(phis)
    #solver = ''

    # Run simulation
    str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_vp_inclusion.app'+ picard + newton + solver + sdpar + \
           ' -output_file '+fname+ \
           ' -C_b '+str(C_b)+ \
           ' -eta_w '+str(eta_w)+ \
           ' -output_dir '+fname+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
    print(str1)
    os.system(str1)

    # Plot solution and error
    #plot_solution(fname,nx,0,0)
    #plot_solution(fname,nx,0,1)
    #plot_solution(fname,nx,1,0)
    #plot_solution(fname,nx,1,1)

    #plot_strain_rates(fname,nx,0)
    #plot_strain_rates(fname,nx,1)
    #plot_stress(fname,nx,0)
    #plot_stress(fname,nx,1)
    #plot_residuals(fname,nx)
  #  plot_norm_iteration(fout)

    plot_allinone(fname,nx,0)

    
# Plot eat for phi0
  fig = plt.figure(1,figsize=(12,8))
  cmaps='RdBu_r'

  num_r = 3
  num_c = 4

  fout = fname+'/'+'out_stokesdarcy2field_vp_inclusion'+casename+blabel+'_alleta_'+str(nx)

  for j in range(0,len(phi0)):
    if j !=15 and j!=16: 
      pv = '_p'+str(j)
      # filenames 
      # fname0 = 'out_stokesdarcy2field_vp_inclusion'+casename+blabel+pv
      plot_eta_all(fname,num_r,num_c,j+1,phi0[j])

  plt.tight_layout() 
    
  plt.savefig(fout+'.pdf')
  plt.close()
      
  # Plot div*v for phi0
  fig = plt.figure(1,figsize=(12,8))
  cmaps='RdBu_r'

  num_r = 3
  num_c = 4

  fout = fname+'/'+'out_stokesdarcy2field_vp_inclusion'+casename+blabel+'_alldivv_'+str(nx)

  for j in range(0,len(phi0)):
    if j !=15 and j!=16: 
      pv = '_p'+str(j)
      # filenames 
      # fname0 = 'out_stokesdarcy2field_vp_inclusion'+casename+blabel+pv
      plot_divv_all(fname,num_r,num_c,j+1,phi0[j])

  plt.tight_layout() 
    
  plt.savefig(fout+'.pdf')
  plt.close()


    
os.system('rm -r '+fname+'/__pycache__')
