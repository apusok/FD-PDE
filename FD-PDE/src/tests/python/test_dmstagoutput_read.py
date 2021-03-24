# ----------------------------------------- #
# Run and visualize ../test_dmstagoutput_read.app
# ----------------------------------------- #

# Import modules
import os
import dmstagoutput as dmout
import matplotlib.pyplot as plt
import importlib
import numpy as np

# Run test
n = 5
nx = n+1
ny = n
dof0 = 3 # vertex
dof1 = 1 # face
dof2 = 5 # cell

create_plot = 0

if (create_plot):
  dof0 = 1 # vertex
  dof1 = 2 # face
  dof2 = 1 # cell

# options = ' -log_view -viewer_binary_skip_info'
options = ''
str1 = 'mpiexec -n 1 ../test_dmstagoutput_read.app '+ \
      ' -nx '+str(nx)+' -ny '+str(ny)+' -dof0 '+str(dof0)+' -dof1 '+str(dof1)+' -dof2 '+str(dof2)+ \
      options
os.system(str1)

fname = 'out_test_dmstagoutput_read'

# Load data
fout = fname
spec = importlib.util.spec_from_file_location(fout,fout+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

fout = fname+'_create'
spec = importlib.util.spec_from_file_location(fout,fout+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_create = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_create)

fout = fname+'_new'
spec = importlib.util.spec_from_file_location(fout,fout+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_new = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_new)

mx = data['Nx'][0]
my = data['Ny'][0]
mz = data['Nz'][0]

dim = data['dim'][0]

dof0 = data['dof0'][0]
dof1 = data['dof1'][0]
dof2 = data['dof2'][0]
dof3 = data['dof3'][0]

print('# Data info: M = %d, N = %d, P = %d, dim = %d, dof0 = %d, dof1 = %d, dof2 = %d, dof3 = %d' %(mx,my,mz,dim,dof0,dof1,dof2,dof3))

x1d = data['x1d']
y1d = data['y1d']

# print(x1d)
# print(y1d)

xc = data['x1d_cell']
yc = data['y1d_cell']
xv = data['x1d_vertex']
yv = data['y1d_vertex']

x1d_c = data_create['x1d']
y1d_c = data_create['y1d']
x1d_n = data_new['x1d']
y1d_n = data_new['y1d']

xc_c = data_create['x1d_cell']
yc_c = data_create['y1d_cell']
xv_c = data_create['x1d_vertex']
yv_c = data_create['y1d_vertex']

xc_n = data_new['x1d_cell']
yc_n = data_new['y1d_cell']
xv_n = data_new['x1d_vertex']
yv_n = data_new['y1d_vertex']

if (dof0):
  Xvertex = data['X_vertex']
  Xvertex_c = data_create['X_vertex']
  Xvertex_n = data_new['X_vertex']

if (dof2):
  Xcell = data['X_cell']
  Xcell_c = data_create['X_cell']
  Xcell_n = data_new['X_cell']

if (dof1):
  Xfacex = data['X_face_x']
  Xfacex_c = data_create['X_face_x']
  Xfacex_n = data_new['X_face_x']

  Xfacey = data['X_face_y']
  Xfacey_c = data_create['X_face_y']
  Xfacey_n = data_new['X_face_y']

# print(Xcell)
if (create_plot):
  Xvertex0 = Xvertex[0::dof0]
  Xvertex0_c = Xvertex_c[0::dof0]
  Xvertex0_n = Xvertex_n[0::dof0]

  Xfacex0 = Xfacex[0::dof1]
  Xfacex0_c = Xfacex_c[0::dof1]
  Xfacex0_n = Xfacex_n[0::dof1]

  Xfacex1 = Xfacex[1::dof1]
  Xfacex1_c = Xfacex_c[1::dof1]
  Xfacex1_n = Xfacex_n[1::dof1]

  Xfacey0 = Xfacey[0::dof1]
  Xfacey0_c = Xfacey_c[0::dof1]
  Xfacey0_n = Xfacey_n[0::dof1]

  Xfacey1 = Xfacey[1::dof1]
  Xfacey1_c = Xfacey_c[1::dof1]
  Xfacey1_n = Xfacey_n[1::dof1]

  Xcell0 = Xcell[0::dof2]
  Xcell0_c = Xcell_c[0::dof2]
  Xcell0_n = Xcell_n[0::dof2]

  # Plot 
  fig, axs = plt.subplots(6,3,figsize=(15,20))

  ax1 = plt.subplot(6,3,1)
  ax2 = plt.subplot(6,3,2)
  ax3 = plt.subplot(6,3,3)
  ax4 = plt.subplot(6,3,4)
  ax5 = plt.subplot(6,3,5)
  ax6 = plt.subplot(6,3,6)
  ax7 = plt.subplot(6,3,7)
  ax8 = plt.subplot(6,3,8)
  ax9 = plt.subplot(6,3,9)
  ax10 = plt.subplot(6,3,10)
  ax11 = plt.subplot(6,3,11)
  ax12 = plt.subplot(6,3,12)
  ax13 = plt.subplot(6,3,13)
  ax14 = plt.subplot(6,3,14)
  ax15 = plt.subplot(6,3,15)
  ax16 = plt.subplot(6,3,16)
  ax17 = plt.subplot(6,3,17)
  ax18 = plt.subplot(6,3,18)

  ax1.set_title('Xvertex0')
  im1 = ax1.imshow(Xvertex0.reshape(my+1,mx+1),origin='lower',extent=[min(xv),max(xv),min(yv),max(yv)],interpolation='none')
  im2 = ax2.imshow(Xvertex0_c.reshape(my+1,mx+1),origin='lower',extent=[min(xv_c),max(xv_c),min(yv_c),max(yv_c)],interpolation='none')
  im3 = ax3.imshow(Xvertex0_n.reshape(my+1,mx+1),origin='lower',extent=[min(xv_n),max(xv_n),min(yv_n),max(yv_n)],interpolation='none')
  cbar1 = fig.colorbar(im1,ax=ax1, shrink=0.80)
  cbar2 = fig.colorbar(im2,ax=ax2, shrink=0.80)
  cbar3 = fig.colorbar(im3,ax=ax3, shrink=0.80)

  ax4.set_title('Xfacex0')
  im4 = ax4.imshow(Xfacex0.reshape(my,mx+1),origin='lower',extent=[min(xv),max(xv),min(yc),max(yc)],interpolation='none')
  im5 = ax5.imshow(Xfacex0_c.reshape(my,mx+1),origin='lower',extent=[min(xv_c),max(xv_c),min(yc_c),max(yc_c)],interpolation='none')
  im6 = ax6.imshow(Xfacex0_n.reshape(my,mx+1),origin='lower',extent=[min(xv_n),max(xv_n),min(yc_n),max(yc_n)],interpolation='none')
  cbar4 = fig.colorbar(im4,ax=ax4, shrink=0.80)
  cbar5 = fig.colorbar(im5,ax=ax5, shrink=0.80)
  cbar6 = fig.colorbar(im6,ax=ax6, shrink=0.80)

  ax7.set_title('Xfacex1')
  im7 = ax7.imshow(Xfacex1.reshape(my,mx+1),origin='lower',extent=[min(xv),max(xv),min(yc),max(yc)],interpolation='none')
  im8 = ax8.imshow(Xfacex1_c.reshape(my,mx+1),origin='lower',extent=[min(xv_c),max(xv_c),min(yc_c),max(yc_c)],interpolation='none')
  im9 = ax9.imshow(Xfacex1_n.reshape(my,mx+1),origin='lower',extent=[min(xv_n),max(xv_n),min(yc_n),max(yc_n)],interpolation='none')
  cbar7 = fig.colorbar(im7,ax=ax7, shrink=0.80)
  cbar8 = fig.colorbar(im8,ax=ax8, shrink=0.80)
  cbar9 = fig.colorbar(im9,ax=ax9, shrink=0.80)

  ax10.set_title('Xfacey0')
  im10 = ax10.imshow(Xfacey0.reshape(my+1,mx),origin='lower',extent=[min(xc),max(xc),min(yv),max(yv)],interpolation='none')
  im11 = ax11.imshow(Xfacey0_c.reshape(my+1,mx),origin='lower',extent=[min(xc_c),max(xc_c),min(yv_c),max(yv_c)],interpolation='none')
  im12 = ax12.imshow(Xfacey0_n.reshape(my+1,mx),origin='lower',extent=[min(xc_n),max(xc_n),min(yv_n),max(yv_n)],interpolation='none')
  cbar10 = fig.colorbar(im10,ax=ax10, shrink=0.80)
  cbar11 = fig.colorbar(im11,ax=ax11, shrink=0.80)
  cbar12 = fig.colorbar(im12,ax=ax12, shrink=0.80)

  ax13.set_title('Xfacey1')
  im13 = ax13.imshow(Xfacey1.reshape(my+1,mx),origin='lower',extent=[min(xc),max(xc),min(yv),max(yv)],interpolation='none')
  im14 = ax14.imshow(Xfacey1_c.reshape(my+1,mx),origin='lower',extent=[min(xc_c),max(xc_c),min(yv_c),max(yv_c)],interpolation='none')
  im15 = ax15.imshow(Xfacey1_n.reshape(my+1,mx),origin='lower',extent=[min(xc_n),max(xc_n),min(yv_n),max(yv_n)],interpolation='none')
  cbar13 = fig.colorbar(im13,ax=ax13, shrink=0.80)
  cbar14 = fig.colorbar(im14,ax=ax14, shrink=0.80)
  cbar15 = fig.colorbar(im15,ax=ax15, shrink=0.80)

  ax16.set_title('Xcell0')
  im16 = ax16.imshow(Xcell0.reshape(my,mx),origin='lower',extent=[min(xc),max(xc),min(yc),max(yc)],interpolation='none')
  im17 = ax17.imshow(Xcell0_c.reshape(my,mx),origin='lower',extent=[min(xc_c),max(xc_c),min(yc_c),max(yc_c)],interpolation='none')
  im18 = ax18.imshow(Xcell0_n.reshape(my,mx),origin='lower',extent=[min(xc_n),max(xc_n),min(yc_n),max(yc_n)],interpolation='none')

  cbar16 = fig.colorbar(im16,ax=ax16, shrink=0.80)
  cbar17 = fig.colorbar(im17,ax=ax17, shrink=0.80)
  cbar18 = fig.colorbar(im18,ax=ax18, shrink=0.80)

  plt.savefig(fname+'.pdf', bbox_inches = 'tight')

# norms:
if (dof0):
  print('# NORMs Vertex:')
  for i in range(0,dof0):
    Xvertexi   = Xvertex[i::dof0]
    Xvertexi_c = Xvertex_c[i::dof0]
    Xvertexi_n = Xvertex_n[i::dof0]
    print('#    dof = %d norm_create = %f norm_new = %f' %(i,np.linalg.norm(Xvertexi-Xvertexi_c),np.linalg.norm(Xvertexi-Xvertexi_n)))

if (dof1):
  print('# NORMs Face:')
  for i in range(0,dof1):
    Xfacexi   = Xfacex[i::dof1]
    Xfacexi_c = Xfacex_c[i::dof1]
    Xfacexi_n = Xfacex_n[i::dof1]

    Xfaceyi   = Xfacey[i::dof1]
    Xfaceyi_c = Xfacey_c[i::dof1]
    Xfaceyi_n = Xfacey_n[i::dof1]
    print('#    X dof = %d norm_create = %f norm_new = %f' %(i,np.linalg.norm(Xfacexi-Xfacexi_c),np.linalg.norm(Xfacexi-Xfacexi_n)))
    print('#    Y dof = %d norm_create = %f norm_new = %f' %(i,np.linalg.norm(Xfaceyi-Xfaceyi_c),np.linalg.norm(Xfaceyi-Xfaceyi_n)))

if (dof2):
  print('# NORMs Cell:')
  for i in range(0,dof2):
    Xcelli   = Xcell[i::dof2]
    Xcelli_c = Xcell_c[i::dof2]
    Xcelli_n = Xcell_n[i::dof2]
    print('#    dof = %d norm_create = %f norm_new = %f' %(i,np.linalg.norm(Xcelli-Xcelli_c),np.linalg.norm(Xcelli-Xcelli_n)))


# coordinates:
print('# NORMs Coordinates:')
print('#    x1d: norm_create = %f norm_new = %f' %(np.linalg.norm(x1d-x1d_c),np.linalg.norm(x1d-x1d_n)))
print('#    y1d: norm_create = %f norm_new = %f' %(np.linalg.norm(y1d-y1d_c),np.linalg.norm(y1d-y1d_n)))

if ((dof0) | (dof1)):
  print('#    xvertex: norm_create = %f norm_new = %f' %(np.linalg.norm(xv-xv_c),np.linalg.norm(xv-xv_n)))
  print('#    yvertex: norm_create = %f norm_new = %f' %(np.linalg.norm(yv-yv_c),np.linalg.norm(yv-yv_n)))

if ((dof1) | (dof2)):
  print('#    xcenter: norm_create = %f norm_new = %f' %(np.linalg.norm(xc-xc_c),np.linalg.norm(xc-xc_n)))
  print('#    ycenter: norm_create = %f norm_new = %f' %(np.linalg.norm(yc-yc_c),np.linalg.norm(yc-yc_n)))

os.system('rm -r __pycache__')