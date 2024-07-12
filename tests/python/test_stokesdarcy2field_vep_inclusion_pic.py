# ---------------------------------------
# Shortening of a visco-(elasto)-plastic block in the absence of gravity
# Setup from T. Gerya, 2018, Ch. 13, ex. 13.2
# METHOD WITH MARKERS (DMSWARM)
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
def plot_allinone(fname,ft,nx):

  # First row P, Vx, Vz, |V|, eta
  # Load data
  fout = fname+'_solution'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  etac_data = data['X_cell']
  etan = data['X_vertex']

  rhsx_data = data['X_face_x']
  rhsy_data = data['X_face_y']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  rhsx = rhsx_data[0::dof]
  rhsy = rhsy_data[0::dof]

  # reshaped rhs
  rx = rhsx.reshape(mz, mx+1)
  ry = rhsy.reshape(mz+1, mx)

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

  etamax = 3
  etamin =-3

  nind = 10

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(15,9))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,5,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  Q  = ax.quiver( xc[::nind], zc[::nind], -dpdxc[::nind,::nind], -dpdzc[::nind,::nind], units='width', pivot='mid' )
  ax.set_title(r'$P_f, \phi(\mathbf{V}_f-\mathbf{V}_s)$')
  ax.set_title(r'$P_f$')
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
  im = ax.imshow(vc,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,5)
  im = ax.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  #Second row: plot epsII, epsxx, epszz, epsxz 
    # Load data
  fout = fname+'_strain'+ft # 1. Numerical solution stokes - strain rates
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

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
  
  # get the deviatoric strain rates
  divc = exx1c + ezz1c
  exx1c = exx1c - 1.0/3.0*divc
  ezz1c = ezz1c - 1.0/3.0*divc
  eII1c = np.power(0.5*(np.power(exx1c,2)+np.power(ezz1c,2)+2.0*np.power(exz1c,2)) ,0.5)
  
  divn = exx1n + ezz1n
  exx1n = exx1n - 1.0/3.0*divn
  ezz1n = ezz1n - 1.0/3.0*divn
  eII1n = np.power(0.5*(np.power(exx1n,2)+np.power(ezz1n,2)+2.0*np.power(exz1n,2)) ,0.5)

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
#  if divmin > -0.1:
#    divmin = -3

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

  #Third row: Updated old value of tauII, tauxx, tauzz, tauxz
  # Load data
  fout = fname+'_stressold'+ft # historic stresses
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']

  fout = fname+'_dpold'+ft # yield stress
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  dpold_data = data['X_cell']
  
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

  dpold = dpold_data

  sxxmax = max(max(sxx1c),max(sxx1n))
  sxxmin = min(min(sxx1c),min(sxx1n))
  szzmax = max(max(szz1c),max(szz1n))
  szzmin = min(min(szz1c),min(szz1n))
  sxzmax = max(max(sxz1c),max(sxz1n))
  sxzmin = min(min(sxz1c),min(sxz1n))
  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))

  dpmax = max(dpold)
  dpmin = min(dpold)

  print(sxxmax,sxxmin,dpmax,dpmin)
  stress = sxxmax
 
  ax = plt.subplot(3,5,11)
  im = ax.imshow(sxx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xx}$(Center)')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,12)
  im = ax.imshow(szz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{zz}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,13)
  im = ax.imshow(sxz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xz}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,14)
  im = ax.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{II}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,15)
  im = ax.imshow(dpold.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=dpmin,vmax=dpmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\Delta P$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'/'+fname+'_allinone'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

  return stress

# ---------------------------------------
def plot_eta(fname,ft,nx):

  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]
  zetac= etac_data[2::dof]
  zetac=zetac + 2.0/3.0*etac

  eta = etac.reshape(mz,mx)
  zeta = zetac.reshape(mz,mx)

  line_cut = [0.1, 0.15, 0.2, 0.25]
  lc = ['k', 'g', 'r', 'b']

  p0 = xc[0]
  dh = xc[1]-xc[0]
  
  #plot eta
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax1.set_title(r'$\eta_{vep}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\eta_{vep}$')
  ax2.set_xlabel('x')

  eta_out = np.zeros(4)

  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    eta_p = xc
    eta_v = eta[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(eta_p, eta_v, lc[nl])

    eta_out[nl] = min(eta_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_eta'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot zeta
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(zeta,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax1.set_title(r'$\zeta_{vep}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\zeta_{vep}$')
  ax2.set_xlabel('x')

  zeta_out = np.zeros(4)

  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    zeta_p = xc
    zeta_v = zeta[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(zeta_p, zeta_v, lc[nl])

    zeta_out[nl] = min(zeta_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_zeta'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_strain(fname,ft,nx):

  fout = fname+'_strain'+ft # strain rates
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

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

  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  divv = exx1c.reshape(mz,mx) + ezz1c.reshape(mz,mx)

  fy = eII1c.reshape(mz,mx)
  
  line_cut = [0.1, 0.15, 0.2, 0.25]
  lc = ['k', 'g', 'r', 'b']

  p0 = xc[0]
  dh = xc[1]-xc[0]
  
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax1.set_title(r'$\dot{\varepsilon}_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\dot{\varepsilon}_{II}$')
  ax2.set_xlabel('x')

  fy_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)

    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    fy_p = xc
    fy_v = fy[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(fy_p, fy_v, lc[nl])

    fy_out[nl] = max(fy_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_strain'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #Plot div V
  fy = divv
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(fy,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax1.set_title(r'$\nabla\cdot\mathbf{V}_s$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\nabla\cdot\mathbf{V}_s$')
  ax2.set_xlabel('x')

  fy_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    fy_p = xc
    fy_v = fy[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(fy_p, fy_v, lc[nl])

    fy_out[nl] = max(fy_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_divv'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_stress(fname,ft,nx,eg,zz,phi,th1,th2):

  #prepare data of stress  
  fout = fname+'_stressold'+ft # historic stresses
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']
  
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

  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))
  sII = sII1c.reshape(mz,mx)
  
  #prepare data of stress  
  fout = fname+'_dpold'+ft # 1. Numerical solution stokes - stress components
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  dp1c = data['X_cell']
    
  # split into dofs
  dpold = dp1c
  dpmax = max(dpold)
  dpmin = min(dpold)
  dp = dpold.reshape(mz,mx)
  
  #prepare data of eta
  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 3
  # dof = 2
  etac = etac_data[1::dof]
  eta = etac.reshape(mz,mx)
  
  #zeta
  zetac= etac_data[2::dof]
  zetac=zetac + 2.0/3.0*etac
  zeta = zetac.reshape(mz,mx)
  # zeta = etac.reshape(mz,mx)
  
  #compute the true viscosity
  eta = eta / (1-phi)
  zeta = zeta/(1-phi)

  #prepare data of strain rates
  fout = fname+'_strain'+ft # 1. Numerical solution stokes - strain rates
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()

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

  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  eII = eII1c.reshape(mz,mx)
  exx = exx1c.reshape(mz,mx)
  ezz = ezz1c.reshape(mz,mx)
  
  # get the deviatoric strain rates
  divc = exx1c + ezz1c
  exx1c = exx1c - 1.0/3.0*divc
  ezz1c = ezz1c - 1.0/3.0*divc
  eII1c = np.power(0.5*(np.power(exx1c,2)+np.power(ezz1c,2)+2.0*np.power(exz1c,2)) ,0.5)
  
  eII = eII1c.reshape(mz,mx)
  divv = divc.reshape(mz,mx)
  
  line_cut = [0.1, 0.15, 0.2, 0.25]
  lc = ['k', 'g', 'r', 'b']

  p0 = xc[0]
  dh = xc[1]-xc[0]
  
  #Compute the stress for the same strain rates without elasticity
  evp = 1/(1/eta - 1/eg)
  sII_vp = 2 * eII * evp #/(1/eta - 1/eg)
  sII_diff = sII_vp - sII #(sII_vp - sII)/sII_vp

  sIImin_diff = np.min(sII_diff)
  sIImax_diff = np.max(sII_diff)
  
  st_diff = np.sum( np.power(sII_diff, 2) )/np.sum( np.power(sII_vp,2) )
  st_diff = st_diff**0.5
  
  idx = np.unravel_index(np.argmin(sII_diff, axis=None), sII_diff.shape)
  print('eta at mim(sIIdiff)',idx, eta[idx],sII_diff[idx], evp[idx],eII[idx])
  print('min evp', np.max(evp), np.min(eta))
  
  #compute the compaction stress without elasticity
  zvp = 1/(1/zeta - 1/zz)
  dp_vp = -zvp*divv
  dp_diff = dp_vp - dp
  dpe2 = np.sum( np.power(dp_diff,2) )/np.sum(np.power(dp_vp,2))
  dpe2 = dpe2**0.5
  
  #find the range of x where sII > ratio * C
  threshold = th1 #ratio*C
  xrange = np.zeros((mz,3))
  xrange[:,2] = zc
  xp_eps = np.zeros((mz,3))  #value and location of the peak strain rates (value,x,z)
  xp_eps[:,2] = zc
  for i in range(0,mz):
    ixs = np.argmax(sII[i,:]>threshold)
    if (ixs == 0 and sII[i,0]>threshold) :
        a = sII[i,:]
        b = a[::-1]
        ixe = len(b) - np.argmax(b>threshold) - 1
        #interpolate
        xrange[i,0] = xc[ixs]
        if ixe < mz-1:
            xrange[i,1] = xc[ixe] + (sII[i,ixe]-threshold)/(sII[i,ixe]-sII[i,ixe+1])*(xc[ixe+1]-xc[ixe])
        else:
            xrange[i,1] = xc[ixe]
        #get the peak of strain rates
        xp_eps[i,0],xp_eps[i,1] = get_peak(ixs,ixe,xc,eII[i,:])
    if ixs != 0:
        a = sII[i,:]
        b = a[::-1]
        ixe = len(b) - np.argmax(b>threshold) - 1
        #interpolate
        xrange[i,0] = xc[ixs-1] + (threshold-sII[i,ixs-1])/(sII[i,ixs]-sII[i,ixs-1])*(xc[ixs]-xc[ixs-1])
        if ixe < mz-1:
            xrange[i,1] = xc[ixe] + (sII[i,ixe]-threshold)/(sII[i,ixe]-sII[i,ixe+1])*(xc[ixe+1]-xc[ixe])
        else:
            xrange[i,1] = xc[ixe]    
        #get the peak of strain rates
        xp_eps[i,0],xp_eps[i,1] = get_peak(ixs,ixe,xc,eII[i,:])
  
  xrange2 = xrange[np.all(xrange > 0.001,axis=1)]
  xp_eps2 = xp_eps[np.all(xrange > 0.001,axis=1)]
  
  #find the range of x where dp > ratio * C (Here it temporialy take as the same threshold to sIz)
  threshold = th2 #threshold for compaction failure
  xrangec = np.zeros((mz,3))
  xrangec[:,2] = zc
  xp_dv = np.zeros((mz,3))  #value and location of the peak div(v) (value,x,z)
  xp_dv[:,2] = zc
  for i in range(0,mz):
    ixs = np.argmax(dp[i,:]>threshold)
    if (ixs == 0 and dp[i,0]>threshold):
        a = dp[i,:]
        b = a[::-1]
        ixe = len(b) - np.argmax(b>threshold) - 1
        #interpolate
        xrangec[i,0] = xc[ixs]
        xrangec[i,1] = xc[ixe] + (dp[i,ixe]-threshold)/(dp[i,ixe]-dp[i,ixe+1])*(xc[ixe+1]-xc[ixe])
        #get the peak of div(v)
        xp_dv[i,0],xp_dv[i,1] = get_peak(ixs,ixe,xc,divv[i,:])
    if ixs != 0:
        a = dp[i,:]
        b = a[::-1]
        ixe = len(b) - np.argmax(b>threshold) - 1
        #interpolate
        xrangec[i,0] = xc[ixs-1] + (threshold-dp[i,ixs-1])/(dp[i,ixs]-dp[i,ixs-1])*(xc[ixs]-xc[ixs-1])
        xrangec[i,1] = xc[ixe] + (dp[i,ixe]-threshold)/(dp[i,ixe]-dp[i,ixe+1])*(xc[ixe+1]-xc[ixe])
        #get the peak of div(v)
        xp_dv[i,0],xp_dv[i,1] = get_peak(ixs,ixe,xc,divv[i,:])
  
  xrangec2 = xrangec[np.all(xrangec > 0.001,axis=1)]
  xp_dv2 = xp_dv[np.all(np.abs(xp_dv) > 0.001,axis=1)]
  
  #plot eta
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax1.set_title(r'$\eta_{vep}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\eta_{vep}$')
  ax2.set_xlabel('x')

  eta_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    eta_p = xc
    eta_v = eta[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(eta_p, eta_v, lc[nl])

    eta_out[nl] = min(eta_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_eta'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot strain rates
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax1.set_title(r'$\dot{\varepsilon}_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\dot{\varepsilon}_{II}$')
  ax2.set_xlabel('x')

  eII_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    eII_p = xc
    eII_v = eII[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(eII_p, eII_v, lc[nl])

    eII_out[nl] = max(eII_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_strain'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot stress - 3rd group of plot
  fig = plt.figure(1,figsize=(12,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,3,1)
  im = ax1.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax1.set_title(r'$\tau_{II}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')
  xr1 = ax1.plot(xrange2[:,0],xrange2[:,2],'k:')
  xr1 = ax1.plot(xrange2[:,1],xrange2[:,2],'k:')
  xp1 = ax1.plot(xp_eps2[:,1],xp_eps2[:,2],'w:')
  
  ax3 =  plt.subplot(1,3,2)
  im = ax3.imshow(sII_diff,extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin_diff,vmax=sIImax_diff,cmap=cmaps,origin='lower')
  ax3.set_title(r'$\tau_{II}^{vp} - \tau_{II}$')
  cbar = fig.colorbar(im,ax=ax3, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,3,3)
  ax2.set_ylabel(r'$\tau_{II}$')
  ax2.set_xlabel('x')

  str_out = np.zeros(4)
  str_fail = np.zeros(4)

  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    stress_p = xc
    stress_v = sII[i1-1, :]
    stress_v_vp = sII_vp[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(stress_p, stress_v, lc[nl])
    line2 = ax2.plot(stress_p, stress_v_vp, lc[nl], ls = '--')
    
    #compute the width of the failure region
    str_fail[nl] = xrange[i1-1,1] - xrange[i1-1,0]
    str_out[nl] = max(stress_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_stress'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot compaction stress - 4th group of plot
  fig = plt.figure(1,figsize=(12,4))
  cmaps='RdBu_r' 
  
  ax1 =  plt.subplot(1,3,1)
  xr1 = ax1.plot(xrangec2[:,0],xrangec2[:,2],'k:')
  xr1 = ax1.plot(xrangec2[:,1],xrangec2[:,2],'k:')
  xp1 = ax1.plot(xp_dv2[:,1],xp_dv2[:,2],'w:')
  im = ax1.imshow(dp,extent=[min(xc), max(xc), min(zc), max(zc)],vmin=dpmin,vmax=dpmax,cmap=cmaps,origin='lower')
  ax1.set_title(r'$\Delta P$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')
  
  ax3 =  plt.subplot(1,3,2)
  im = ax3.imshow(dp_diff,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  #ax3.set_title(r'$(\tau_{II}^{vp} - \tau_{II})/\tau_{II}^{vp}$')
  ax3.set_title(r'$\Delta P^{vp} - \Delta P$')
  cbar = fig.colorbar(im,ax=ax3, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,3,3)
  ax2.set_ylabel(r'$\Delta P$')
  ax2.set_xlabel('x')

  dp_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    stress_p = xc
    stress_v = dp[i1-1, :]
    stress_v_vp = dp_vp[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(stress_p, stress_v, lc[nl])
    line2 = ax2.plot(stress_p, stress_v_vp, lc[nl], ls = '--')
    
    dp_out[nl] = max(stress_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_dp'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot zeta
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(zeta,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax1.set_title(r'$\zeta_{vep}$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\zeta_{vep}$')
  ax2.set_xlabel('x')

  zeta_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
    
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    zeta_p = xc
    zeta_v = zeta[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(zeta_p, zeta_v, lc[nl])

    zeta_out[nl] = min(zeta_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_zeta'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot Maxwell time for shear
  mwmin = np.log10(np.min(eg/evp))#-3
  mwmax = np.log10(np.max(eg/evp))#np.max(np.log(eg/evp))
  fig = plt.figure(1,figsize=(4,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,1,1)
  im = ax1.imshow(np.log10(eg/evp),extent=[min(xc), max(xc), min(zc), max(zc)],vmin = mwmin, vmax = mwmax, cmap=cmaps,origin='lower')
  ax1.set_title(r'$log_{10}(\Delta t/(\eta_{vp}/G))$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  plt.tight_layout()
  fout = fname+'/'+fname+'_mwshear'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #plot Maxwell time for compaction
  mwmin = np.log10(np.min(zz/zvp))
  mwmax = np.log10(np.max(zz/zvp))#np.log(np.max(zz/zvp))
  fig = plt.figure(1,figsize=(4,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,1,1)
  im = ax1.imshow(np.log10(zz/zvp),extent=[min(xc), max(xc), min(zc), max(zc)],vmin = mwmin, vmax = mwmax, cmap=cmaps,origin='lower')
  ax1.set_title(r'$log_{10}(\Delta t/(\zeta_{vp}/Z))$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  plt.tight_layout()
  fout = fname+'/'+fname+'_mwcp'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  #Plot div V
  fy = divv
  divmin = -2
  divmax = np.amax(divv)
  fig = plt.figure(1,figsize=(8,4))
  cmaps='RdBu_r' 

  ax1 =  plt.subplot(1,2,1)
  im = ax1.imshow(fy,extent=[min(xc), max(xc), min(zc), max(zc)],vmin=divmin,vmax=divmax,cmap=cmaps,origin='lower')
  ax1.set_title(r'$\nabla\cdot\mathbf{V}_s$')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75, format='%.2f')

  ax2 =  plt.subplot(1,2,2)
  ax2.set_ylabel(r'$\nabla\cdot\mathbf{V}_s$')
  ax2.set_xlabel('x')

  fy_out = np.zeros(4)
  for nl in range(0,4):
    l1 = line_cut[nl]
    i1 = int(mx*l1/0.5)
  
    p1 = xc[mx-1]
    l1x = [p0,p1]
    l1y = [zc[i1-1], zc[i1-1]]

    fy_p = xc
    fy_v = fy[i1-1, :]

    slice1 = ax1.plot(l1x,l1y, lc[nl] )
    line1 = ax2.plot(fy_p, fy_v, lc[nl])

    fy_out[nl] = max(fy_v)

  plt.tight_layout()
  fout = fname+'/'+fname+'_divv'+ft+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()
  
  print((str_out - 2*eII_out/(1/eta_out - 1/eg ))/str_out  )
  return str_out, eta_out, eII_out, sIImax_diff, st_diff, str_fail, dpe2

# ---------------------------------------
# use get the peak value and location between ixs and ixe for y = y(x)
def get_peak(ixs,ixe,x,y):
    
    n = len(x)
    i = np.argmax(np.abs(y[ixs:ixe+1]))
    ip = np.zeros((3),dtype=int)
    ip[0] = ixs + i -1
    ip[1] = ip[0]+1  #actual peak in the data
    ip[2] = ip[1]+1
    
    a = np.zeros((3,3))
    b = np.zeros(3)
    a[0,0] = np.power(x[ip[0]],2)
    a[1,0] = np.power(x[ip[1]],2)
    a[2,0] = np.power(x[ip[2]],2)
    a[0,1] = x[ip[0]]
    a[1,1] = x[ip[1]]
    a[2,1] = x[ip[2]]
    a[0,2] = 1.0
    a[1,2] = 1.0
    a[2,2] = 1.0
    b[0] = y[ip[0]]
    b[1] = y[ip[1]]
    b[2] = y[ip[2]]
    abc = np.linalg.solve(a,b)
    
    aa = abc[0]
    bb = abc[1]
    cc = abc[2]
    
    xp = -0.5*bb/aa
    yp = aa*xp*xp + bb*xp + cc

    if i == 0:
        xp = x[ixs]
        yp = y[ixs]
        print(ip[0],xp,yp)
        
    if xp<x[0]:
        xp = x[0]
        yp = y[0]
        print(ip[0],xp,yp)
        
    return yp,xp

# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# StokesDarcy-VEP, inclusion test with DMSWARM (PIC)')
print('# --------------------------------------- #')

# Set main parameters and run test
nx    = 100 # resolution

etamin = 0.0
phi0   = 1e-4
R      = 0.0 #1.0
lam    = 0.0

G  = 1.0
Z0 = 100.0
tstep = 101 #1000#60
tmax = 20000.0
dt = 1.0

eb = 1e3#1e3
ew = 1e-1#1e-1
cb = 20
lam_p = 1.0  #YC = (C_b or C_w)*lam_p
lam_v = 0.1 #zeta_v = eta_v / lam_v
nh = 1.0

rheology = 1

tout = 10

eg = G*dt
zz = Z0*dt
ratio = 0.99
th1 = cb*ratio   #threshold for high shear stresses
th2 = cb*lam_p*ratio #threshold for high compaction stresses

# filenames 
fname = 'out_stokesdarcy2field_vep_inclusion_pic'
fout = fname+'/'+fname+'_'+str(nx)+'.out'

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

if (ncpu == -1):
  solver_cpu = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
  # nx = 50
else:
  solver_cpu = ' -pc_type lu -pc_factor_mat_solver_type mumps'
  # nx = 21 # Warning: mumps seg fault with higher resolution on laptop

newton = ' -snes_linesearch_type bt'+ \
  ' -snes_linesearch_monitor'+ \
  ' -snes_atol 1e-10'+ \
  ' -snes_rtol 1e-50'+ \
  ' -snes_stol 1e-10'+ \
  ' -snes_max_it 200'+ \
  ' -snes_monitor'+ \
  ' -snes_view'+ \
  ' -snes_monitor_true_residual'+ \
  ' -ksp_monitor_true_residual'+ \
  ' -ksp_gmres_restart 300' + \
  ' -snes_converged_reason'+ \
  ' -ksp_converged_reason'+ \
  ' -python_snes_failed_report'+ \
  ' -ksp_type fgmres -log_view'

solver= ' -snes_mf_operator'

sdpar = ' -R ' + str(R)+ \
  ' -phi_0 ' + str(phi0)+ \
  ' -lambda ' + str(lam)+ \
  ' -G ' + str(G) +\
  ' -Z0 ' + str(Z0) +\
  ' -tstep ' + str(tstep)+\
  ' -tmax ' + str(tmax)+\
  ' -dt ' + str(dt)

# Run simulation
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_vep_inclusion_pic.app' + solver_cpu + newton + solver + sdpar + \
  ' -output_file '+fname+ \
  ' -C_b '+str(cb)+ \
  ' -lam_p '+str(lam_p)+ \
  ' -ew_v0 '+str(ew)+ \
  ' -eb_v0 '+str(eb)+ \
  ' -lam_v '+str(lam_v)+ \
  ' -etamin '+str(etamin)+ \
  ' -nh ' +str(nh)+ \
  ' -rheology ' +str(rheology)+ \
  ' -output_dir '+fname+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
print(str1)
os.system(str1)

# Parse log file
f = open(fout, 'r')
i0=0
for line in f:
  if '# TIMESTEP' in line:
      i0+=1
f.close()
tstep =i0

stress = np.zeros([tstep])
tt = np.zeros((tstep,1))

str_samp = np.zeros((tstep,4))
eta_samp = np.zeros((tstep,4))
eps_samp = np.zeros((tstep,4))
sdiff = np.zeros((tstep,1))
stdiff = np.zeros((tstep,1))
stfail = np.zeros((tstep,4))
dpe2 = np.zeros((tstep,1))

print(tstep)

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep <= 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)
  
  fout = fname
  # stress[istep+1] = plot_allinone(fout,ft,nx)
  tt[istep,0] = dt* istep
  str_samp[istep,:],eta_samp[istep,:],eps_samp[istep,:],sdiff[istep,0],stdiff[istep,0], stfail[istep,:], dpe2[istep,0] \
      = plot_stress(fout,ft,nx,eg,zz,phi0,th1,th2)
  # eta_samp[istep,:] = plot_eta(fout,ft,nx)
  # eps_samp[istep,:] = plot_strain(fout,ft,nx)

  print("t = ", tt[istep,0]) 
  print(str_samp[istep,:])
  print(eta_samp[istep,:])
  print(eps_samp[istep,:])
  print(sdiff[istep,0])
  print(stdiff[istep,0])
  print(dpe2[istep,0])
  print(stfail[istep,:])

#combine all sample data in column-wise
datasav = np.concatenate((tt,str_samp,eta_samp,eps_samp,sdiff,stdiff,stfail,dpe2),axis=1)
np.savetxt(fname+'/'+'vep_evolution.txt', datasav, delimiter = ',')

# #load data from vep_evo_data.txt
# evodata = np.loadtxt(fname+'/'+'vep_evolution.txt', delimiter=',')
# tt = np.zeros((len(evodata),1))
# sdiff = np.zeros((len(evodata),1))
# stdiff = np.zeros((len(evodata),1))
# dpe2 = np.zeros((len(evodata),1))
# tt[:,0] = evodata[:,0]
# str_samp = evodata[:,1:5]
# eta_samp = evodata[:,5:9]
# eps_samp = evodata[:,9:13]
# sdiff[:,0] = evodata[:,13]
# stdiff[:,0] = evodata[:,14]
# stfail = evodata[:,15:19]
# dpe2[:,0] = evodata[:,19]

lc = ['k', 'g', 'r', 'b']
fig = plt.figure(1,figsize=(15,8))

tmax = 50

ax2 = plt.subplot(2, 3, 1)
l1 = ax2.plot(tt[:,0], eta_samp[:,0], lc[0])
l2 = ax2.plot(tt[:,0], eta_samp[:,1], lc[1])
l3 = ax2.plot(tt[:,0], eta_samp[:,2], lc[2])
l4 = ax2.plot(tt[:,0], eta_samp[:,3], lc[3])
ax2.set_xlabel('t',fontsize=12)
ax2.set_ylabel(r'$MIN(\eta_{vep})$',fontsize=12)
ax2.set_xlim(0,tmax)
ax2.set_ylim(np.min(eta_samp)*0.9, G*dt)

ax3 = plt.subplot(2, 3, 2)
l1 = ax3.plot(tt[:,0], eps_samp[:,0], lc[0])
l2 = ax3.plot(tt[:,0], eps_samp[:,1], lc[1])
l3 = ax3.plot(tt[:,0], eps_samp[:,2], lc[2])
l4 = ax3.plot(tt[:,0], eps_samp[:,3], lc[3])
ax3.set_xlabel('t',fontsize=12)
ax3.set_ylabel(r'$MAX(\dot{\varepsilon}_{II})$',fontsize=12)
ax3.set_xlim(0,tmax)
ax3.set_ylim(0,np.max(eps_samp)*1.05)

ax1 = plt.subplot(2, 3, 3)
l1 = ax1.plot(tt[:,0], str_samp[:,0], lc[0])
l2 = ax1.plot(tt[:,0], str_samp[:,1], lc[1])
l3 = ax1.plot(tt[:,0], str_samp[:,2], lc[2])
l4 = ax1.plot(tt[:,0], str_samp[:,3], lc[3])
ax1.set_xlabel('t',fontsize=12)
ax1.set_ylabel(r'$MAX(\tau_{II})$',fontsize=12)
ax1.set_xlim(0,tmax)
ax1.set_ylim(0,1.05*cb)

ax6 = plt.subplot(2, 3, 4)
l1 = ax6.plot(tt[:,0], stfail[:,0], lc[0])
l2 = ax6.plot(tt[:,0], stfail[:,1], lc[1])
l3 = ax6.plot(tt[:,0], stfail[:,2], lc[2])
l4 = ax6.plot(tt[:,0], stfail[:,3], lc[3])
ax6.set_xlabel('t',fontsize=12)
ax6.set_ylabel(r'Width for $\tau_{II} > 0.99C$',fontsize=12)
ax6.set_xlim(0,tmax*2)
ax6.set_ylim(0,np.max(stfail)*1.1)

# ax4 = plt.subplot(2, 3, 5)
# l1 = ax4.plot(tt[:,0], sdiff[:,0], 'k-')
# ax4.set_xlabel('t')
# ax4.set_ylabel(r'$MAX(\frac{\tau_{II}^{vp}-\tau_{II}}{\tau_{II}^{vp}})$')
# ax4.set_yscale('log')
# ax4.set_xlim(0,500)
# ax4.set_ylim(np.min(sdiff)*0.5,np.max(sdiff)+0.1)

ax5 = plt.subplot(2, 3, 5)
l1 = ax5.plot(tt[:,0], stdiff[:,0], 'k-')
ax5.set_xlabel('t',fontsize=12)
ax5.set_ylabel(r'$\frac{||\tau_{II}^{vp}-\tau_{II} ||_2}{||\tau_{II}^{vp}||_2}$',fontsize=12)
ax5.set_yscale('log')
ax5.set_xlim(0,tstep*dt)
ax5.set_ylim(np.min(stdiff)*0.5,np.max(stdiff)+0.1)

ax7 = plt.subplot(2, 3, 6)
l1 = ax7.plot(tt[:,0], dpe2[:,0], 'k-')
ax7.set_xlabel('t',fontsize=12)
ax7.set_ylabel(r'$\frac{||\Delta P^{vp}-\Delta P ||_2}{|| \Delta P^{vp} ||_2}$',fontsize=12)
ax7.set_yscale('log')
ax7.set_xlim(0,tstep*dt)
ax7.set_ylim(np.min(dpe2)*0.5,np.max(dpe2)+0.1)

plt.tight_layout()
plt.savefig(fname+'/'+'vep_evolution.pdf')
plt.close()

os.system('rm -r '+fname+'/__pycache__')