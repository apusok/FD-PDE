# ---------------------------------------
# MMS test for power-law viscosity for Stokes and StokesDarcy
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import importlib
import os

# ---------------------------------------
# Function definitions
# ---------------------------------------
def parse_log_file(fname):
  try: # try to open directory
    # Parse output and save norm info
    line_ind = 6
    f = open(fname, 'r')
    for line in f:
      if 'Velocity test1:' in line:
          nrm_v1_num = float(line[20+line_ind:38+line_ind])
      if 'Pressure test1:' in line:
          nrm_p1_num = float(line[20+line_ind:38+line_ind])
      if 'Strain rates CENTER test1:' in line:
          nrm_exxc1_num = float(line[41:41+18])
          nrm_ezzc1_num = float(line[72:72+18])
          nrm_exzc1_num = float(line[103:103+18])
      if 'Strain rates CORNER test1:' in line:
          nrm_exxn1_num = float(line[41:41+18])
          nrm_ezzn1_num = float(line[72:72+18])
          nrm_exzn1_num = float(line[103:103+18])
      if 'Velocity test2:' in line:
          nrm_v2_num = float(line[20+line_ind:38+line_ind])
      if 'Pressure test2:' in line:
          nrm_p2_num = float(line[20+line_ind:38+line_ind])
      if 'Strain rates CENTER test2:' in line:
          nrm_exxc2_num = float(line[41:41+18])
          nrm_ezzc2_num = float(line[72:72+18])
          nrm_exzc2_num = float(line[103:103+18])
      if 'Strain rates CORNER test2:' in line:
          nrm_exxn2_num = float(line[41:41+18])
          nrm_ezzn2_num = float(line[72:72+18])
          nrm_exzn2_num = float(line[103:103+18])
      if 'Grid info test1:' in line:
          hx_num = float(line[18+line_ind:36+line_ind])
    f.close()

    return nrm_v1_num, nrm_p1_num, nrm_exxc1_num, nrm_ezzc1_num, nrm_exzc1_num, nrm_exxn1_num, nrm_ezzn1_num, nrm_exzn1_num, nrm_v2_num, nrm_p2_num, nrm_exxc2_num, nrm_ezzc2_num, nrm_exzc2_num, nrm_exxn2_num, nrm_ezzn2_num, nrm_exzn2_num, hx_num
  except OSError:
    print('Cannot open:', fdir)
    return tstep

# ---------------------------------------
def plot_solution_mms_error(fname,nx,j):

  # Load data
  fout = fname+'_stokes' # 1. Numerical solution stokes
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx1 = data['X_face_x']
  vz1 = data['X_face_y']
  p1 = data['X_cell']

  fout = fname+'_stokesdarcy' # 2. Numerical solution stokesdarcy
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  vx2 = data['X_face_x']
  vz2 = data['X_face_y']
  p2  = data['X_cell']

  fout = 'out_mms_solution'
  imod = importlib.import_module(fout) # 3. MMS solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  vx_mms = data['X_face_x']
  vz_mms = data['X_face_y']
  p_mms = data['X_cell']

  pmax = max(p_mms)
  pmin = min(p_mms)
  vxmax = max(vx_mms)
  vxmin = min(vx_mms)
  vzmax = max(vz_mms)
  vzmin = min(vz_mms)

  # Plot all fields - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(18,12))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,5,1)
  im = ax.imshow(p_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'MMS $P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,2)
  im = ax.imshow(p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,3)
  im = ax.imshow(p_mms.reshape(mz,mx)-p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,4)
  im = ax.imshow(p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,5)
  im = ax.imshow(p_mms.reshape(mz,mx)-p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,6)
  im = ax.imshow(vx_mms.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'MMS $v_x$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,7)
  im = ax.imshow(vx1.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,8)
  im = ax.imshow(vx_mms.reshape(mz,mx+1)-vx1.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,9)
  im = ax.imshow(vx2.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,10)
  im = ax.imshow(vx_mms.reshape(mz,mx+1)-vx2.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,11)
  im = ax.imshow(vz_mms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'MMS $v_z$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,12)
  im = ax.imshow(vz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,13)
  im = ax.imshow(vz_mms.reshape(mz+1,mx)-vz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,14)
  im = ax.imshow(vz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,15)
  im = ax.imshow(vz_mms.reshape(mz+1,mx)-vz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_solution'+'_npind'+str(j)+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()


# ---------------------------------------
def plot_rhs_mms(fname,nx,j):

  # Load data
  fout = fname+'_rhs_stokes' # 1. Numerical solution stokes
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx1 = data['X_face_x']
  vz1 = data['X_face_y']
  p1 = data['X_cell']

  fout = fname+'_rhs_stokesdarcy' # 2. Numerical solution stokesdarcy
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  vx2 = data['X_face_x']
  vz2 = data['X_face_y']
  p2  = data['X_cell']

  pmax = max(max(p1),max(p2))
  pmin = min(min(p1),min(p2))
  vxmax = max(max(vx1),max(vx2))
  vxmin = min(min(vx1),min(vx2))
  vzmax = max(max(vz1),max(vz2))
  vzmin = min(min(vz1),min(vz2))

  # Plot all fields - rhs for P, ux, uz
  fig = plt.figure(1,figsize=(9,12))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,2,1)
  im = ax.imshow(p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes $P$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,2)
  im = ax.imshow(p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $P$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,3)
  im = ax.imshow(vx1.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_x$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,4)
  im = ax.imshow(vx2.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_x$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,5)
  im = ax.imshow(vz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_z$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,6)
  im = ax.imshow(vz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_z$ RHS')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_rhs'+'_npind'+str(j)+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_strain_rates_error(fname,nx,j):

  # Load data
  fout = fname+'_strain_stokes' # 1. Numerical solution stokes - strain rates
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  fout = fname+'_strain_stokesdarcy' # 2. Numerical solution stokesdarcy
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  eps2n = data['X_vertex']
  eps2c = data['X_cell']

  fout = 'out_mms_strain'
  imod = importlib.import_module(fout) # 3. MMS solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  epsn_mms = data['X_vertex']
  epsc_mms = data['X_cell']

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

  exx2c = eps2c[0::dof]
  ezz2c = eps2c[1::dof]
  exz2c = eps2c[2::dof]
  eII2c = eps2c[3::dof]

  exx2n = eps2n[0::dof]
  ezz2n = eps2n[1::dof]
  exz2n = eps2n[2::dof]
  eII2n = eps2n[3::dof]

  exxc_mms = epsc_mms[0::dof]
  ezzc_mms = epsc_mms[1::dof]
  exzc_mms = epsc_mms[2::dof]
  exxn_mms = epsn_mms[0::dof]
  ezzn_mms = epsn_mms[1::dof]
  exzn_mms = epsn_mms[2::dof]

  exxmax = max(max(exxc_mms),max(exxn_mms))
  exxmin = min(min(exxc_mms),min(exxn_mms))
  ezzmax = max(max(ezzc_mms),max(ezzn_mms))
  ezzmin = min(min(ezzc_mms),min(ezzn_mms))
  exzmax = max(max(exzc_mms),max(exzn_mms))
  exzmin = min(min(exzc_mms),min(exzn_mms))

  # Plot all fields - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(15,24))
  cmaps='RdBu_r' 

  ax = plt.subplot(8,5,1)
  im = ax.imshow(exxc_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,2)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{xx}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,3)
  im = ax.imshow(exxc_mms.reshape(mz,mx)-exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{xx}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,4)
  im = ax.imshow(exx2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{xx}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,5)
  im = ax.imshow(exxc_mms.reshape(mz,mx)-exx2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{xx}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,6)
  im = ax.imshow(ezzc_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{zz}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,7)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,8)
  im = ax.imshow(ezzc_mms.reshape(mz,mx)-ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,9)
  im = ax.imshow(ezz2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,10)
  im = ax.imshow(ezzc_mms.reshape(mz,mx)-ezz2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,11)
  im = ax.imshow(exzc_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{xz}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,12)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,13)
  im = ax.imshow(exzc_mms.reshape(mz,mx)-exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,14)
  im = ax.imshow(exz2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,15)
  im = ax.imshow(exzc_mms.reshape(mz,mx)-exz2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,17)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,19)
  im = ax.imshow(eII2c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,21)
  im = ax.imshow(exxn_mms.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{xx}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,22)
  im = ax.imshow(exx1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{xx}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,23)
  im = ax.imshow(exxn_mms.reshape(mz+1,mx+1)-exx1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{xx}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,24)
  im = ax.imshow(exx2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exxmin,vmax=exxmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{xx}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,25)
  im = ax.imshow(exxn_mms.reshape(mz+1,mx+1)-exx2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{xx}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,26)
  im = ax.imshow(ezzn_mms.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{zz}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,27)
  im = ax.imshow(ezz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,28)
  im = ax.imshow(ezzn_mms.reshape(mz+1,mx+1)-ezz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,29)
  im = ax.imshow(ezz2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,30)
  im = ax.imshow(ezzn_mms.reshape(mz+1,mx+1)-ezz2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,31)
  im = ax.imshow(exzn_mms.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'MMS $\epsilon_{xz}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,32)
  im = ax.imshow(exz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,33)
  im = ax.imshow(exzn_mms.reshape(mz+1,mx+1)-exz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes Error $\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,34)
  im = ax.imshow(exz2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exzmin,vmax=exzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,35)
  im = ax.imshow(exzn_mms.reshape(mz+1,mx+1)-exz2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy Error $\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,37)
  im = ax.imshow(eII1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes $\epsilon_{II}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(8,5,39)
  im = ax.imshow(eII2n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $\epsilon_{II}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_strain'+'_npind'+str(j)+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error(fname,nexp,hx,nrm_p1,nrm_v1,nrm_p2,nrm_v2):
  hx_log    = np.log10(hx)
  nrmp1_log = np.log10(nrm_p1)
  nrmv1_log = np.log10(nrm_v1)
  nrmp2_log = np.log10(nrm_p2)
  nrmv2_log = np.log10(nrm_v2)

  slp1 = np.zeros(len(nexp))
  slv1 = np.zeros(len(nexp))
  slp2 = np.zeros(len(nexp))
  slv2 = np.zeros(len(nexp))

  # Perform linear regression
  for j in range(len(nexp)):
    slp1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmp1_log[j,:])
    slv1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmv1_log[j,:])
    slp2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmp2_log[j,:])
    slv2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmv2_log[j,:])

  colors = plt.cm.viridis(np.linspace(0,1,len(nexp)))
  plt.figure(1,figsize=(12,6))

  plt.subplot(121)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_p1[j,:],'o--',color=colors[j],label='np='+str(nexp[j])+' P sl='+str(round(slp1[j],5)))
    plt.plot(hx[j,:],nrm_v1[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' v sl='+str(round(slv1[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.title('a) Stokes',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(122)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_p2[j,:],'o--',color=colors[j],label='np='+str(nexp[j])+' P sl='+str(round(slp2[j],5)))
    plt.plot(hx[j,:],nrm_v2[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' v sl='+str(round(slv2[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.title('b) Stokes-Darcy',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_hx_L2.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error_strain_rate(fname,nexp,hx,nrm_exxc1,nrm_ezzc1,nrm_exzc1,nrm_exxn1,nrm_ezzn1,nrm_exzn1,nrm_exxc2,nrm_ezzc2,nrm_exzc2,nrm_exxn2,nrm_ezzn2,nrm_exzn2):
  hx_log    = np.log10(hx)
  nrmexxc1_log = np.log10(nrm_exxc1)
  nrmezzc1_log = np.log10(nrm_ezzc1)
  nrmexzc1_log = np.log10(nrm_exzc1)
  nrmexxc2_log = np.log10(nrm_exxc2)
  nrmezzc2_log = np.log10(nrm_ezzc2)
  nrmexzc2_log = np.log10(nrm_exzc2)

  nrmexxn1_log = np.log10(nrm_exxn1)
  nrmezzn1_log = np.log10(nrm_ezzn1)
  nrmexzn1_log = np.log10(nrm_exzn1)
  nrmexxn2_log = np.log10(nrm_exxn2)
  nrmezzn2_log = np.log10(nrm_ezzn2)
  nrmexzn2_log = np.log10(nrm_exzn2)

  slexxc1 = np.zeros(len(nexp))
  slezzc1 = np.zeros(len(nexp))
  slexzc1 = np.zeros(len(nexp))
  slexxc2 = np.zeros(len(nexp))
  slezzc2 = np.zeros(len(nexp))
  slexzc2 = np.zeros(len(nexp))

  slexxn1 = np.zeros(len(nexp))
  slezzn1 = np.zeros(len(nexp))
  slexzn1 = np.zeros(len(nexp))
  slexxn2 = np.zeros(len(nexp))
  slezzn2 = np.zeros(len(nexp))
  slexzn2 = np.zeros(len(nexp))

  # Perform linear regression
  for j in range(len(nexp)):
    slexxc1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexxc1_log[j,:])
    slezzc1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmezzc1_log[j,:])
    slexzc1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexzc1_log[j,:])
    slexxc2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexxc2_log[j,:])
    slezzc2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmezzc2_log[j,:])
    slexzc2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexzc2_log[j,:])

    slexxn1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexxn1_log[j,:])
    slezzn1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmezzn1_log[j,:])
    slexzn1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexzn1_log[j,:])
    slexxn2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexxn2_log[j,:])
    slezzn2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmezzn2_log[j,:])
    slexzn2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmexzn2_log[j,:])

  colors = plt.cm.viridis(np.linspace(0,1,len(nexp)))
  plt.figure(1,figsize=(12,12))

  plt.subplot(221)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_exxc1[j,:],'o-',color=colors[j],label='np='+str(nexp[j])+' exx sl='+str(round(slexxc1[j],5)))
    plt.plot(hx[j,:],nrm_ezzc1[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' ezz sl='+str(round(slezzc1[j],5)))
    plt.plot(hx[j,:],nrm_exzc1[j,:],'s:',color=colors[j],label='np='+str(nexp[j])+' exz* sl='+str(round(slexzc1[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\epsilon_{xx}), E(\epsilon_{zz}), E(\epsilon_{xz})$',fontweight='bold',fontsize=12)
  plt.title('a) Stokes - CENTER',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(222)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_exxc2[j,:],'o-',color=colors[j],label='np='+str(nexp[j])+' exx sl='+str(round(slexxc2[j],5)))
    plt.plot(hx[j,:],nrm_ezzc2[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' ezz sl='+str(round(slezzc2[j],5)))
    plt.plot(hx[j,:],nrm_exzc2[j,:],'s:',color=colors[j],label='np='+str(nexp[j])+' exz* sl='+str(round(slexzc2[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\epsilon_{xx}), E(\epsilon_{zz}), E(\epsilon_{xz})$',fontweight='bold',fontsize=12)
  plt.title('b) Stokes-Darcy - CENTER',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(223)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_exxn1[j,:],'o-',color=colors[j],label='np='+str(nexp[j])+' exx* sl='+str(round(slexxn1[j],5)))
    plt.plot(hx[j,:],nrm_ezzn1[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' ezz* sl='+str(round(slezzn1[j],5)))
    plt.plot(hx[j,:],nrm_exzn1[j,:],'s:',color=colors[j],label='np='+str(nexp[j])+' exz sl='+str(round(slexzn1[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\epsilon_{xx}), E(\epsilon_{zz}), E(\epsilon_{xz})$',fontweight='bold',fontsize=12)
  plt.title('c) Stokes - CORNER',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(224)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_exxn2[j,:],'o-',color=colors[j],label='np='+str(nexp[j])+' exx* sl='+str(round(slexxn2[j],5)))
    plt.plot(hx[j,:],nrm_ezzn2[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' ezz* sl='+str(round(slezzn2[j],5)))
    plt.plot(hx[j,:],nrm_exzn2[j,:],'s:',color=colors[j],label='np='+str(nexp[j])+' exz sl='+str(round(slexzn2[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\epsilon_{xx}), E(\epsilon_{zz}), E(\epsilon_{xz})$',fontweight='bold',fontsize=12)
  plt.title('d) Stokes-Darcy - CORNER',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_strain_rate_hx_L2.pdf')
  plt.close()

# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# MMS tests for power-law effective viscosity ')
print('# --------------------------------------- #')

# Set main parameters and run test
fname = 'out_effvisc'
n  = [20, 40, 100, 120, 150, 200, 250, 300] #[21, 41, 101, 121, 151, 201, 251, 301] # resolution
nexp = [1.0, 2.0, 3.0] #[1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # [1.0, 5.0, 10.0, 15.0, 30.0] #power-law exponent

# Prepare errors and convergence
nrm_p1  = np.zeros((len(nexp),len(n))) # 1- stokes
nrm_p2  = np.zeros((len(nexp),len(n))) # 2- stokes-darcy
nrm_v1  = np.zeros((len(nexp),len(n)))
nrm_v2  = np.zeros((len(nexp),len(n)))
hx      = np.zeros((len(nexp),len(n)))

nrm_exxc1 = np.zeros((len(nexp),len(n)))
nrm_ezzc1 = np.zeros((len(nexp),len(n)))
nrm_exzc1 = np.zeros((len(nexp),len(n)))
nrm_exxc2 = np.zeros((len(nexp),len(n)))
nrm_ezzc2 = np.zeros((len(nexp),len(n)))
nrm_exzc2 = np.zeros((len(nexp),len(n)))

nrm_exxn1 = np.zeros((len(nexp),len(n)))
nrm_ezzn1 = np.zeros((len(nexp),len(n)))
nrm_exzn1 = np.zeros((len(nexp),len(n)))
nrm_exxn2 = np.zeros((len(nexp),len(n)))
nrm_ezzn2 = np.zeros((len(nexp),len(n)))
nrm_exzn2 = np.zeros((len(nexp),len(n)))

# Run simulations
for j in range(len(nexp)):  
  for i in range(len(n)):
    # Create output filename
    inp = nexp[j]
    nx  = n[i]
    fout = fname+'_np'+str(j)+'_'+str(nx)+'.out'

    # Run with different resolutions - 1 timestep
    str1 = '../test_effvisc_mms.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor_true_residual -ksp_monitor_true_residual -snes_converged_reason -ksp_converged_reason'+ \
        ' -output_file '+fname+ \
        ' -nexp '+str(inp)+ \
        ' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
    print(str1)
    os.system(str1)

    # Parse variables
    nrm_v1_num, nrm_p1_num, nrm_exxc1_num, nrm_ezzc1_num, nrm_exzc1_num, nrm_exxn1_num, nrm_ezzn1_num, nrm_exzn1_num, nrm_v2_num, nrm_p2_num, nrm_exxc2_num, nrm_ezzc2_num, nrm_exzc2_num, nrm_exxn2_num, nrm_ezzn2_num, nrm_exzn2_num, hx_num = parse_log_file(fout)
    nrm_p1[j,i] = nrm_p1_num
    nrm_v1[j,i] = nrm_v1_num
    nrm_p2[j,i] = nrm_p2_num
    nrm_v2[j,i] = nrm_v2_num
    hx[j,i]     = hx_num

    nrm_exxc1[j,i] = nrm_exxc1_num
    nrm_ezzc1[j,i] = nrm_ezzc1_num
    nrm_exzc1[j,i] = nrm_exzc1_num
    nrm_exxc2[j,i] = nrm_exxc2_num
    nrm_ezzc2[j,i] = nrm_ezzc2_num
    nrm_exzc2[j,i] = nrm_exzc2_num

    nrm_exxn1[j,i] = nrm_exxn1_num
    nrm_ezzn1[j,i] = nrm_ezzn1_num
    nrm_exzn1[j,i] = nrm_exzn1_num
    nrm_exxn2[j,i] = nrm_exxn2_num
    nrm_ezzn2[j,i] = nrm_ezzn2_num
    nrm_exzn2[j,i] = nrm_exzn2_num

    # Plot solution and error
    plot_solution_mms_error(fname,nx,j)
    plot_strain_rates_error(fname,nx,j)
    plot_rhs_mms(fname,nx,j)

# Convergence plot
plot_convergence_error(fname,nexp,hx,nrm_p1,nrm_v1,nrm_p2,nrm_v2)
plot_convergence_error_strain_rate(fname,nexp,hx,nrm_exxc1,nrm_ezzc1,nrm_exzc1,nrm_exxn1,nrm_ezzn1,nrm_exzn1,nrm_exxc2,nrm_ezzc2,nrm_exzc2,nrm_exxn2,nrm_ezzn2,nrm_exzn2)

os.system('rm -r __pycache__')