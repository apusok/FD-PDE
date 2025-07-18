# ---------------------------------------

# ---------------------------------------

def plot_allinone(fname,ft,nx):

  # First row P, Vx, Vz, |V|, eta
  # Load data
  fout = fname+'_solution'+ft
  #fout = fname+'_solution_initial'
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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

  dpold = dpold_data[0::dof]

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



# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

# Input file
fname = 'out_stokesdarcy2field_beam'

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

print('# --------------------------------------- #')
print('# Visco-elastic bending beam')
print('# --------------------------------------- #')

n = 100
z_in = 0.5
eps = 0.8/n #1.0/n #0.7/n
gamma = 1.0
vfopt = 3

lp = 0.5 + 0.1/2.0

dt = 1.0/8.0/n #0.5*0.5/n**2/eps

tstep = 100001#641#int(0.1/dt) + 1
tstep = 51

dtck = 0.01
# maxckpt = 10 + 99
maxckpt = 12

if dt >= dtck:
    dt = dtck/2.0

tout = 1 #int((tstep-1)/4) #10
tout2 = 1

etamin = 0.0
phi0 = 1e-4
R = 0.0 #1.0
lam = 0.0

Gu = 100
Gd = 10
Z = 1e40

eta_u = 1e-3
eta_d = 1e1
C_u = 1e40
C_d = 1e40
Fu = 0.2/phi0
Fd = 2.0/phi0 #1e-10/phi0

lam_p = 1.0  #YC = (C_b or C_w)*lam_p
lam_v = 0.001 #zeta_v = eta_v / lam_v
nh = 1.0

fname = fname + '_n'+str(n)
try:
  os.mkdir(fname)
except OSError:
  pass

# solver parameters
phase = ' -eps '+str(eps)+' -gamma '+str(gamma) + ' -vfopt '+str(vfopt)
model = ' -L 1.2 -H 1.0  -dt '+str(dt)+' -z_in '+str(z_in) +\
    ' -eta_u '+str(eta_u) + ' -eta_d '+str(eta_d) + ' -C_d '+str(C_d) + ' -C_u '+str(C_u) +\
    ' -Fu ' + str(Fu) + ' -Fd ' + str(Fd) +\
    ' -Gu ' + str(Gu) + ' -Gd ' + str(Gd) +\
    ' -dtck ' + str(dtck) + ' -maxckpt ' +str(maxckpt)

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
  ' -ksp_gmres_restart 300' + \
  ' -snes_converged_reason'+ \
  ' -ksp_converged_reason'+ \
  ' -python_snes_failed_report'+ \
  ' -ksp_type fgmres' + \
  ' -pc_factor_mat_ordering_type external'  #this one is used for petsc 3.14

solver= ' -snes_mf_operator'
solver = ''

sdpar = ' -R ' + str(R)+ \
  ' -phi_0 ' + str(phi0)+ \
  ' -lambda ' + str(lam)+ \
  ' -Z ' + str(Z) +\
  ' -lam_p '+str(lam_p)+ \
  ' -lam_v '+str(lam_v)+ \
  ' -etamin '+str(etamin)+ \
  ' -nh ' +str(nh)

# Run test
# Forward euler
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_beam' + \
       ' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep) + \
       newton + model + phase + solver + sdpar + \
       ' -output_dir '+fname+' -output_file '+fname+' -tout '+str(tout)
print(str1)
os.system(str1)

# Prepare data for time series
# Load data - initial
f1out = fname+'_phase_initial'
spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_ini = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_ini)
# imod = importlib.import_module(fname+'_phase_initial')
# data_ini = imod._PETScBinaryLoad()
# imod._PETScBinaryLoadReportNames(data_ini)

# Get general data (elements, grid)
m = data_ini['Nx'][0]
n = data_ini['Ny'][0]
xc = data_ini['x1d_cell']
zc = data_ini['y1d_cell']

f_ini = data_ini['X_cell']
fres_ini = f_ini.reshape(n,m)

# total mass
mass = 0
for j in range(0, n):
    for i in range(0, m):
        mass += Fd*phi0 *(1- fres_ini[j, i]) + Fu*phi0 *fres_ini[j,i] 

# analytical solution - prepare the initial state
mm = 200
xini = np.linspace(0.5/n, 1 - 0.5/n, mm)
zini = z_in + 0.1*np.sin(2*np.pi*xini - 0.5*np.pi)



nind = 5

# data to store the root-mean-sqare velocity
vrms = np.zeros([maxckpt+1,2])

for ickpt in range(0,maxckpt+1):
  # Load python module describing data
  if (ickpt < 10): ft = '_ts00'+str(ickpt)
  if (ickpt >= 10) & (ickpt < 99): ft = '_ts0'+str(ickpt)
  if (ickpt >= 100): ft = '_ts'+str(ickpt)

  # Load data - phase
  f1out = fname+'_phase'+ft
  spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data0 = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fname+'_phase'+ft)
  # data0 = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data0)

  # load data - PV  
  fout = fname+'_solution'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data1 = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data1 = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data1['Nx'][0]
  mz = data1['Ny'][0]
  xc = data1['x1d_cell']
  zc = data1['y1d_cell']
  xv = data1['x1d_vertex']
  zv = data1['y1d_vertex']
  vx = data1['X_face_x']
  vz = data1['X_face_y']
  p  = data1['X_cell']
  
  # Prepare cell center velocities and pressure
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )
  pp = p.reshape(mz,mx)

  # load data - coefficent
  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data2 = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data2 = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  etac_data = data2['X_cell']
  dof = 3
  etac = etac_data[1::dof]
  
  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  
  vrms[ickpt,0] = dtck * ickpt
  
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      
      vrms[ickpt,1] += vc[j][i]**2 / (mx*mz)
      
  vrms[ickpt,1] = vrms[ickpt,1]**0.5


  print('Maximum velocity magnitude = ', np.max(vc))

  # Get individual data sets - phase
  f0 = data0['X_cell']
  fres0 = f0.reshape(n,m)

  # check mass continuity
  mass1 = 0
  for j in range(0, n):
    for i in range(0, m):
      mass1 += Fd*phi0 *(1- fres0[j, i]) + Fu*phi0 *fres0[j,i] 
  
  print('Mass change =', mass1 - mass, (mass1 -mass)/mass)

#  if (ickpt == maxckpt-1):
  if ( ickpt % tout2 == 0):
      
    # Plot solution 
    fig1, axs1 = plt.subplots(1, 1,figsize=(4,4))
      
    tt = dtck * ickpt
        
    cmaps='RdBu_r'
    # color map
    ax0 = axs1
    im = ax0.imshow(fres0,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
    #im = ax0.imshow(pp,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
    Q  = ax0.quiver( xc[::nind], zc[::nind], vxc[::nind,::nind], vzc[::nind,::nind], units='width', pivot='mid', scale_units='xy', scale=8 )
    ax0.contour( xc , zc , fres0, levels=[0.5] , colors='black',linestyles='solid',linewidths=1.0)
    ct01= ax0.contour( xc , zc , fres_ini, levels=[0.5] , colors='black',linestyles='dotted',linewidths=1)
    #ax0.plot(xini, tt+zini, 'k--')
    ax0.set_title('Visco-elastic beam')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    #cbar = fig1.colorbar(im,ax=ax0, shrink=0.75)
    #line = ax0.axhline(y = lp, color='k', linestyle='--')
    ax0.grid(True,color='gray', linestyle=':', linewidth=0.5)        

    fig1.savefig(fname+'/'+fname+ft+'_sol.pdf')
    plt.close(fig1)

    # Plot solution 
    fig2, axs2 = plt.subplots(1, 1,figsize=(4,4))
    # color map
    ax0 = axs2
    im = ax0.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    cbar = fig2.colorbar(im,ax=ax0, shrink=0.75)
    ax0.grid(True,color='gray', linestyle=':', linewidth=0.5)        

    fig2.savefig(fname+'/'+fname+ft+'_coeff.pdf')
    plt.close(fig2)
    

os.system('rm -r '+fname+'/__pycache__')

for ickpt in range(0,maxckpt):
  # Load python module describing data
  if (ickpt < 10): ft = '_ts00'+str(ickpt)
  if (ickpt >= 10) & (ickpt < 99): ft = '_ts0'+str(ickpt)
  if (ickpt >= 100): ft = '_ts'+str(ickpt)
#  plot_allinone(fname,ft,n)

