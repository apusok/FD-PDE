# ---------------------------------------
# Four-lid driven simple rotation flow benchmark - constant grid spacing (convergence test)
# 1. Convergence test within a small box 0.05x0.05 at left bottom corners,
# 2. Confirm abs(P)< 1e-6 in all diagonals. 
# 3. Confirm the rotation symmetry feature
# 3. Flow field plot
# ---------------------------------------

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import importlib
import os
import sys, getopt

print('# --------------------------------------- #')
print('# Four-lid-driven rotation flow benchmark ')
print('# --------------------------------------- #')

# Input file
fname = 'out_lid_driven'
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
nlist = [80, 100, 200, 300, 400]

# initialize
hx = np.zeros(len(nlist))
errp = np.zeros(len(nlist))
erru = np.zeros(len(nlist))
errv = np.zeros(len(nlist))

# Run simulations
ii = 0 #initialise the count for for loops
for nx in nlist:

    # Create output filename
    fout1 = fname_data+'/'+fname+'_'+str(nx)+'.out'

    # Run with different resolutions
    str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_lid_driven.app '+solver+solver_default+ \
      ' -output_file '+fname+' -output_dir '+fname_data+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
    print(str1)
    os.system(str1)
    
    # import out_lid as on
    # data = on._PETScBinaryLoad()
    fname_out = 'out_lid_driven'
    spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod) 
    data = imod._PETScBinaryLoad()
    
    # Get data
    m = data['Nx'][0]
    n = data['Ny'][0]
    xv = data['x1d_vertex']
    yv = data['y1d_vertex']
    xc = data['x1d_cell']
    yc = data['y1d_cell']
    vxface = data['X_face_x']
    vyface = data['X_face_y']
    p = data['X_cell']
    
    # Compute the DOF count on each stratum (face,element)
    facedof_x = int(len(vxface)/((m+1)*n))
    facedof_y = int(len(vyface)/(m*(n+1)))
    celldof   = int(len(p)/(m*n))
    
    # Prepare cell center velocities
    vxface = vxface.reshape(n  ,m+1)
    vyface = vyface.reshape(n+1,m  )
    
    # Compute the cell center values from the face data by averaging neighbouring faces
    vxc = np.zeros( [n , m] )
    vyc = np.zeros( [n , m] )
    for i in range(0,m):
      for j in range(0,n):
        vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
        vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

    #get the reshaped data for pressure
    pp = p.reshape(n,m)
    
    #check if it is zero pressure an the two diagnols - n=m for this check, report error on screen if too large
    pd = np.zeros([2*n])
    for i in range(0, n):
        pd[2*i] = pp[i][i]
        pd[2*i+1] = pp[i][n-1-i]
    
    if np.max(abs(pd))>1e-6:
        str2 = 'abs(P) on diagnoals is larger than 1e-6 at nx = ' +str(nx)+ ', that Max(abs(P)) = ' +str(np.max(abs(pd)))
        print(str2)
    
    
    #Confirm the rotational symmetry
    nn = int(n/2)
    mm = int(m/2)
    p_lb_lu = np.zeros([nn, mm])
    u_lb_lu = np.zeros([nn, mm])
    v_lb_lu = np.zeros([nn, mm])
    p_lb_ru = np.zeros([nn, m])
    u_lb_ru = np.zeros([nn, mm])
    v_lb_ru = np.zeros([nn, mm])
    p_lb_rb = np.zeros([nn, mm])
    u_lb_rb = np.zeros([nn, mm])
    v_lb_rb = np.zeros([nn, mm])
    for j in range(0, nn):
        for i in range(0, mm):
            p_lb_lu[j][i] = pp[j][i] - pp[m-1 - i][j]
            p_lb_ru[j][i] = pp[j][i] - pp[n-1 - j][m-1 - i]
            p_lb_rb[j][i] = pp[j][i] - pp[i][n-1 - j]
            u_lb_lu[j][i] = vxc[j][i] + vyc[m-1 - i][j]
            u_lb_ru[j][i] = vxc[j][i] + vxc[n-1 - j][m-1 - i]
            u_lb_rb[j][i] = vxc[j][i] - vyc[i][n-1 - j]
            v_lb_lu[j][i] = vyc[j][i] - vxc[m-1 - i][j]
            v_lb_ru[j][i] = vyc[j][i] + vyc[n-1 - j][m-1 - i]
            v_lb_rb[j][i] = vyc[j][i] + vxc[i][n-1 - j]    
    if np.max(abs(p_lb_lu))>1e-6 or np.max(abs(u_lb_lu))>1e-6 or np.max(abs(v_lb_lu))>1e-6:
        str3 = 'LB-LU symmetry breaks at nx = ' +str(nx)+ ', that Max error of p, u and v = ' +str(np.max(abs(p_lb_lu)))+ ', ' +str(np.max(abs(u_lb_lu)))+ ', ' +str(np.max(abs(v_lb_lu)))
        print(str3)
    if np.max(abs(p_lb_ru))>1e-6 or np.max(abs(u_lb_ru))>1e-6 or np.max(abs(v_lb_ru))>1e-6:
        str3 = 'LB-RU symmetry breaks at nx = ' +str(nx)+ ', that Max error of p, u and v = ' +str(np.max(abs(p_lb_ru)))+ ', ' +str(np.max(abs(u_lb_ru)))+ ', ' +str(np.max(abs(v_lb_ru)))
        print(str3)
    if np.max(abs(p_lb_rb))>1e-6 or np.max(abs(u_lb_rb))>1e-6 or np.max(abs(v_lb_rb))>1e-6:
        str3 = 'LB-RB symmetry breaks at nx = ' +str(nx)+ ', that Max error of p, u and v = ' +str(np.max(abs(p_lb_rb)))+ ', ' +str(np.max(abs(u_lb_rb)))+ ', ' +str(np.max(abs(v_lb_rb)))
        print(str3)
            
    
    
    #analytical solution: p, u and v
    dd = -1/(np.pi/2 + 1)
    cc = -dd
    aa = -np.pi/2/(np.pi/2 + 1)
    ppa = np.zeros([n,m])
    uana = np.zeros([n,m])
    vana = np.zeros([n,m])
    for i in range(0,m):
        for j in range(0,n):
            th = np.arctan(yc[j]/xc[i])
            xr = xc[i]/(xc[i]**2 + yc[j]**2)
            yr = yc[j]/(xc[i]**2 + yc[j]**2)
            uana[j][i] = aa + cc * th + (cc*yc[j] + dd*xc[i])*xr
            vana[j][i] = - (dd*th - (cc*yc[j] + dd*xc[i])*yr)
            ppa[j][i] = 2.0*(cc*yc[j]+dd*xc[i])/(xc[i]**2+yc[j]**2)
    
    #error check, x<0.05, y< 0.05
    corner_size = 0.05
    il = int(n * corner_size)
    
    dx = 1.0/m
    dz = 1.0/n
    for j in range(0, il):
        for i in range(0, il):
            errp[ii] = errp[ii] + abs((pp[j][i] - ppa[j][i]))*dx*dz
            erru[ii] = erru[ii] + abs((vxc[j][i] - uana[j][i]))*dx*dz
            errv[ii] = errv[ii] + abs((vyc[j][i] - vana[j][i]))*dx*dz
            
    #grid size
    hx[ii] = 1.0/nx
    #update the counter
    ii = ii + 1
    

#prepare the straight lines for 1st and 2nd order convergence rate
x1 = [1/400, 1/200]
y1 = [5e-5, 1e-4]
x2 = [1/400, 1/200]
y2 = [5e-5, 2e-4]
ax = plt.subplot(111)
ax.plot(np.log10(hx),np.log10(errp), 'ko--', label='P error') 
ax.plot(np.log10(hx),np.log10(erru+errv), 'bs--', label='v error')
ax.plot(np.log10(x1), np.log10(y1), 'k-', label='order 1')
ax.plot(np.log10(x2), np.log10(y2), 'r-', label='order 2')
ax.legend(loc='lower right')
ax.set_xlabel('log10(dx)')
ax.set_ylabel('log10(P_err), log10(v_err)')
ax.set_title('Discretisation error in the corner x,z in [0,0.05]')
plt.savefig(fname+'/'+'lid_driven_convergence.pdf')

print('# --------------------------------------- #')
print('# Printed Four-lid driven flow convergence results to: lid_driven_convergence.eps')
print('# --------------------------------------- #')


#compute another case to show the flow field, It is easier to show the symmetric feature with odd number cells
nx = 201
# Create output filename
fout1 = fname_data+'/'+fname+'_'+str(nx)+'.out'

# Run with different resolutions
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_lid_driven.app '+solver+solver_default+ \
  ' -output_file '+fname+' -output_dir '+fname_data+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
print(str1)
os.system(str1)

# import out_lid as on
# data = on._PETScBinaryLoad()
fname_out = 'out_lid_driven'
spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod) 
data = imod._PETScBinaryLoad()

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']
yv = data['y1d_vertex']
xc = data['x1d_cell']
yc = data['y1d_cell']
vxface = data['X_face_x']
vyface = data['X_face_y']
p = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(p)/(m*n))

# Prepare cell center velocities
vxface = vxface.reshape(n  ,m+1)
vyface = vyface.reshape(n+1,m  )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

nind = 10

# Open a figure
fig, ax1 = plt.subplots(1)
im = plt.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)], origin='lower', cmap='RdBu', interpolation='nearest' )
Q = plt.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid' )
plt.axis('image')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Four-Lid driven flow, N=200')
cbar = fig.colorbar( im, ax=ax1 )
cbar.ax.set_ylabel('pressure')
plt.savefig(fname+'/'+'lid_driven_flowfield.pdf')

os.system('rm -r '+fname_data+'/__pycache__')

# =============================================================================
# ncut = 5;
# nn = int((n-1)/ncut);
# mm = int((m-1)/ncut);
# pplb = np.zeros([nn, mm]);
# for j in range(0, nn):
#     for i in range(0, mm):
#         pplb[j][i] = ppa[j][i] - pp[j][i];
#         if ppa[j][i] ==0.0:
#             pplb[j][i] = 0.0;
#         else:
#             pplb[j][i] = pplb[j][i]/ppa[j][i];
# 
# # Open a figure
# fig, ax1 = plt.subplots(1)
# im = plt.imshow( pplb, extent=[min(xv), max(xv)/ncut, min(yv), max(yv)/ncut], origin='lower', cmap='RdBu', interpolation='nearest' )
# plt.axis(aspect='image')
# plt.xlabel('x')
# plt.ylabel('z')
# plt.title('Left bottom, (P_num - P_ana)/P_ana, N = 20')
# cbar = fig.colorbar( im, ax=ax1 )
# cbar.ax.set_ylabel('P relative diff')
# =============================================================================
