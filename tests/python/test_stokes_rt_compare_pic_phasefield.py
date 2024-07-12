# ---------------------------------------
# Rayleigh-Taylor instability test
# Compare particle-in-cell vs phase-field method
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import dmstagoutput as dmout
from matplotlib import rc
import os
import sys, getopt

print('# --------------------------------------- #')
print('# Rayleigh-Taylor Instability (STOKES) - PIC and PhaseField')
print('# --------------------------------------- #')

class EmptyStruct:
  pass

def correct_path_load_data_xmf(fname):
  try:
    fname_new = fname[:-4]+'_new.xmf'

    # Copy new file
    f = open(fname, 'r')
    f_new = open(fname_new, 'w')
    line_prev = ''

    for line in f:
      if '<DataItem' in line_prev:
        try:
          idx = line.index('/')
          line = line[idx+1:]
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

    # print(mark.n)

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

# ---------------------------------------
# Main script
# ---------------------------------------

# Input file
fname = 'out_stokes_rt_compare_pic_phasefield'
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

nx = 50
nt = 501
tout = 50
dt = 0.0005

eta0 = 1e-4
eta1 = 1e-4
rho0 = 1.0
rho1 = 0.9

ppcell = 4

solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason -log_view'
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
  # nx = 50
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'
  # nx = 21 # Warning: mumps seg fault with higher resolution on laptop

buoy = ' -eta0 '+str(eta0)+' -eta1 '+str(eta1)+' -rho0 '+str(rho0)+' -rho1 '+str(rho1)

# Run PIC
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_rt_compare_pic_phasefield.app'+solver+solver_default+' -snes_type ksponly -snes_fd_color -output_dir '+fname+ \
    buoy+' -ppcell '+str(ppcell)+' -nt '+str(nt)+' -dt '+str(dt)+' -nx '+str(nx)+' -nz '+str(nx)+' -tout '+str(tout)+' -method 0 > log_'+fname+'0.out'
print(str1)
os.system(str1)

eps = 0.8/nx 
gamma = 1.0
vfopt = 3

# Run PhaseField
phasefield = ' -eps '+str(eps)+' -gamma '+str(gamma)+' -vfopt '+str(vfopt)
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_rt_compare_pic_phasefield.app'+solver+solver_default+' -snes_type ksponly -snes_fd_color -output_dir '+fname+ \
    buoy+phasefield+' -nt '+str(nt)+' -dt '+str(dt)+' -nx '+str(nx)+' -nz '+str(nx)+' -tout '+str(tout)+' -method 1 > log_'+fname+'1.out'
print(str1)
os.system(str1)

# correct paths for xmf
tstep_xmf = os.listdir(fname)
tstep_xmf_check = list.copy(tstep_xmf)
for s in tstep_xmf_check:
  if '.pbin' in s:
    tstep_xmf.remove(s)
tstep_xmf_check = list.copy(tstep_xmf)
for s in tstep_xmf_check:
  if '.py' in s:
    tstep_xmf.remove(s)
tstep_xmf_check = list.copy(tstep_xmf)
for s in tstep_xmf_check:
  if 'pycache' in s:
    tstep_xmf.remove(s)
tstep_xmf_check = list.copy(tstep_xmf)
for s in tstep_xmf_check:
  if '.pdf' in s:
    tstep_xmf.remove(s)

for istep_xmf in tstep_xmf:
  correct_path_load_data_xmf(fname+'/'+istep_xmf)

# ---------------------------------------
# Plotting - compare solution, phase field and markers at nt-1
# ---------------------------------------
for it in np.arange(0,nt,tout):
  if (it > 0):
    # PIC
    f1out = 'out_pic_xPV_ts'+str(it)
    spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    mx = data['Nx'][0]
    mz = data['Ny'][0]
    Vx1 = data['X_face_x']
    Vz1= data['X_face_y']
    P1 = data['X_cell']

    # Phase field
    f1out = 'out_phase_xPV_ts'+str(it)
    spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    x = data['x1d_vertex']
    z = data['y1d_vertex']
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    Vx2 = data['X_face_x']
    Vz2= data['X_face_y']
    P2 = data['X_cell']

    # Compute center velocities - MMS1
    Vx1_sq = Vx1.reshape(mz  ,mx+1)
    Vz1_sq = Vz1.reshape(mz+1,mx  )
    Vx2_sq = Vx2.reshape(mz  ,mx+1)
    Vz2_sq = Vz2.reshape(mz+1,mx  )

    # Compute the cell center values from the face data by averaging neighbouring faces
    Vxc1 = np.zeros([mz,mx])
    Vzc1 = np.zeros([mz,mx])
    Vxc2 = np.zeros([mz,mx])
    Vzc2 = np.zeros([mz,mx])
    for i in range(0,mx):
      for j in range(0,mz):
        Vxc1[j][i] = 0.5 * (Vx1_sq[j][i+1] + Vx1_sq[j][i])
        Vzc1[j][i] = 0.5 * (Vz1_sq[j+1][i] + Vz1_sq[j][i])
        Vxc2[j][i] = 0.5 * (Vx2_sq[j][i+1] + Vx2_sq[j][i])
        Vzc2[j][i] = 0.5 * (Vz2_sq[j+1][i] + Vz2_sq[j][i])

    # Plot data
    fig = plt.figure(1,figsize=(10,5))
    nind = 4

    ax = plt.subplot(1,2,1)
    im = ax.imshow(P1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap='RdBu')
    Q = ax.quiver( xc[::nind], zc[::nind], Vxc1[::nind,::nind], Vzc1[::nind,::nind], units='width', pivot='mid' )
    ax.set_title('PIC', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(1,2,2)
    im = ax.imshow(-P2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap='RdBu')
    Q = ax.quiver( xc[::nind], zc[::nind], Vxc2[::nind,::nind], Vzc2[::nind,::nind], units='width', pivot='mid' )
    ax.set_title('Phase Field', fontweight='bold')
    ax.set_xlabel('x')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    plt.tight_layout() 
    plt.savefig(fname+'/solution_nx_'+str(nx)+'_nt'+str(it+1)+'.pdf')
    plt.close()

  # Plot phase field
  f1out = 'out_num_phase-'+str(it)
  # f1out = 'out_num_phase-0'
  spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  mx = data['Nx'][0]
  mz = data['Ny'][0]
  x = data['x1d_vertex']
  z = data['y1d_vertex']
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  F = data['X_cell']

  fig = plt.figure(1,figsize=(5,5))

  ax = plt.subplot(1,1,1)
  im = ax.imshow(np.flipud(F.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],cmap='RdBu')
  ax.set_title('F', fontweight='bold')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  plt.savefig(fname+'/phase_var_nx_'+str(nx)+'_nt'+str(it)+'.pdf')
  plt.close()

  # Plot PIC - markers
  mark = parse_marker_file('out_num_pic-'+str(it)+'.xmf',fname)
  fig = plt.figure(1,figsize=(5,5))

  ax = plt.subplot(1,1,1)
  im = ax.scatter(mark.x,mark.z,c=mark.id,s=0.5,linewidths=None,cmap='RdBu')
  ax.set_xlim(0.0,1.0)
  ax.set_ylim(0.0,1.0)
  ax.set_aspect('equal')
  ax.set_title('PIC id', fontweight='bold')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  plt.savefig(fname+'/pic_var_nx_'+str(nx)+'_nt'+str(it)+'.pdf')
  plt.close()

os.system('rm -r '+fname+'/__pycache__')