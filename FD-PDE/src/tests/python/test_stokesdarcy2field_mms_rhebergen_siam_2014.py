# ------------------------------------------------ #
# MMS sympy for Rhebergen et al. SIAM 2014 (EX. 6.1)
# ------------------------------------------------ #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import importlib
import dmstagoutput as dmout
import sys, getopt

# Input file
f1 = 'out_mms_rhebergen_2014_solution'
fname = 'out_stokesdarcy2field_mms_rhebergen_siam_2014'
fname_data = fname+'/data'
try:
  os.mkdir(fname)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Use umfpack for sequential and mumps for parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == 1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Parameters
n = [25, 40, 50, 80, 100, 125, 150, 200, 300, 400]
alpha = [-1.0/3.0, 0, 1, 1e1, 1e3, 1e6]

lbl = ['-1/3',
'0',
'1e0',
'1e1',
'1e3',
'1e6']

nsim = len(alpha)
colors = plt.cm.inferno(np.linspace(0,1,nsim+3))

# Run simulations
for al in alpha:
  for nx in n:
    # Create output filename
    i = alpha.index(al)
    fout1 = fname_data+'/'+f1+'_'+str(nx)+'_'+str(i)+'.out'
  
    # Run with different resolutions
    str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_rhebergen-siam-2014.app '+solver+solver_default+ \
      ' -output_dir '+fname_data+' -alpha '+str(al)+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
    print(str1)
    os.system(str1)
  
    # Get mms solution data
    f1out = 'out_mms_solution'
    # imod = importlib.import_module(f1out)
    spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)

    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)
  
    mx = data['Nx'][0]
    mz = data['Ny'][0]
    uxmms = data['X_face_x']
    uzmms= data['X_face_y']
    pmms = data['X_cell']
  
    # Get solution data
    f1out = 'out_solution'
    # imod = importlib.import_module(f1out)
    spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)

    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)
  
    x = data['x1d_vertex']
    z = data['y1d_vertex']
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    ux = data['X_face_x']
    uz= data['X_face_y']
    p = data['X_cell']
  
    # Plot data
    fig = plt.figure(1,figsize=(12,12))
  
    ax = plt.subplot(331)
    im = ax.imshow(pmms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)])
    ax.set_title('MMS P')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(332)
    im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)])
    ax.set_title('Numerical P')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(333)
    im = ax.imshow(pmms.reshape(mz,mx)-p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)])
    ax.set_title('Error P')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(334)
    im = ax.imshow(uxmms.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)])
    ax.set_title('MMS ux')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(335)
    im = ax.imshow(ux.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)])
    ax.set_title('Numerical ux')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(336)
    im = ax.imshow(uxmms.reshape(mz,mx+1)-ux.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)])
    ax.set_title('Error ux')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(337)
    im = ax.imshow(uzmms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)])
    ax.set_title('MMS uz')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(338)
    im = ax.imshow(uz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)])
    ax.set_title('Numerical uz')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  
    ax = plt.subplot(339)
    im = ax.imshow(uzmms.reshape(mz+1,mx)-uz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)])
    ax.set_title('Error uz')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
    plt.savefig(fname+'/'+f1+'_nx_'+str(nx)+'_alpha_'+str(al)+'.pdf')
    plt.close()

  # output solution - general 
  #   dmout.general_output_imshow('out_mms_solution',None,None)
  #   dmout.general_output_imshow('out_solution',None,None)
  #   f1out = 'out_mms_solution_cell_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
  #   f1out = 'out_mms_solution_face0_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
  #   f1out = 'out_mms_solution_face1_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
  #   f1out = 'out_solution_cell_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
  #   f1out = 'out_solution_face0_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
  #   f1out = 'out_solution_face1_dof0'
  #   os.system('mv '+f1out+'.pdf '+f1out+'_'+str(nx)+'.pdf')
    
# Plot convergence data
plt.figure(1,figsize=(6,6))
plt.grid(color='lightgray', linestyle=':')
    
# Norm variables
nrm1v = np.zeros(len(n))
nrm1vx = np.zeros(len(n))
nrm1vz = np.zeros(len(n))
nrm1p = np.zeros(len(n))
hx = np.zeros(len(n))

# Parse output and save norm info
for j in range(0, len(alpha)):
  for i in range(0,len(n)):
      nx = n[i]
  
      fout1 = fname_data+'/'+f1+'_'+str(nx)+'_'+str(j)+'.out'
  
      # Open file 1 and read
      f = open(fout1, 'r')
      for line in f:
        if 'Velocity:' in line:
            nrm1v[i] = float(line[20:38])
            nrm1vx[i] = float(line[48:66])
            nrm1vz[i] = float(line[76:94])
        if 'Pressure:' in line:
            nrm1p[i] = float(line[20:38])
        if 'Grid info:' in line:
            hx[i] = float(line[18:36])
      f.close()
  
  plt.plot(np.log10(hx),np.log10(nrm1p),'--',color=colors[j+2],label=r'$\alpha$'+'='+lbl[j])
  if (j==len(alpha)-1):
      plt.plot(np.log10(hx),np.log10(nrm1v),'k+--',label='v')
      plt.plot(np.log10(hx),np.log10(nrm1p),'ko--',label='P')
  plt.plot(np.log10(hx),np.log10(nrm1v),'+--',color=colors[j+2])
  plt.plot(np.log10(hx),np.log10(nrm1p),'o--',color=colors[j+2])
  
  # Print convergence orders:
  hx_log    = np.log10(hx)
  nrm1v_log = np.log10(nrm1v)
  nrm1p_log = np.log10(nrm1p)
  
  # Perform linear regression
  sl1v, intercept, r_value, p_value, std_err = linregress(hx_log, nrm1v_log)
  sl1p, intercept, r_value, p_value, std_err = linregress(hx_log, nrm1p_log)
  
  print('# --------------------------------------- #')
  print('# MMS StokesDarcy2Field (Rhebergen et al. SIAM 2014) convergence order:')
  print('     v_slope = '+str(sl1v)+' p_slope = '+str(sl1p))
    
x1 = [-2.6, -1.6]
y1 = [-6, -5]
y2 = [-6, -4]    

plt.plot(x1,y1,'k--',label='slope=1')
plt.plot(x1,y2,'k-',label='slope=2')

plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
plt.legend()

plt.savefig(fname+'/'+f1+'.pdf')
plt.close()

os.system('rm -r '+fname_data+'/__pycache__')

