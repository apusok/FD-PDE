"""dmstagoutput
===============

Provides functions to visualize PetscBinary data (DMStag type data):
  1. general_output_imshow(fname,cmap_input,interp_input)
  2. general_output_pcolormesh(fname,cmap_input) - (should be used for debugging!)

  imshow - for equidistant grids (fastest plotting version)
  pcolormesh - for variable grids, fast version of pcolor() but cannot use interpolation straightforward (user needs to manipulate data)

The standard usage of this module should look like:
  >>> import dmstagoutput as dmout

  Example:
  >>> dmout.general_output_imshow(fname,'RdBu','bilinear') 
  >>> dmout.general_output_imshow('test1',None,None)

  >>> dmout.general_output_pcolor(fname,'RdBu') 

  where:
  fname - name of PetscBinary file (output)
  cmap_input - colormap for Matplotlib https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
  inter_input - interpolation for Matplotlib https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/interpolation_methods.html 

"""
# ---------------------------------
# Load modules
# ---------------------------------
import importlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Definitions
# ---------------------------------
def general_output_imshow(fname,cmap_input,interp_input):
  print('# ---------------------------- ')
  print('# Loading PetscBinary data for plotting IMSHOW')
  print('# ---------------------------- ')

  # Load python module describing data
  imod = importlib.import_module(fname)

  # Load data
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  # Sort data into variables
  nx = data['Nx'][0]
  nz  = data['Ny'][0]

  # x1d = data['x1d']
  # z1d = data['y1d']

  # Compute dofs from the data
  ndof0  = (nx+1)*(nz+1)
  ndof1x = (nx+1)*nz
  ndof2  = nx*nz

  dof0 = 0
  dof1 = 0
  dof2 = 0

  if 'x1d_vertex' in data:
    x1d_vertex = data['x1d_vertex']
    z1d_vertex = data['y1d_vertex']

  if 'x1d_cell' in data:
    x1d_cell   = data['x1d_cell']
    z1d_cell   = data['y1d_cell']

  if 'X_vertex' in data:
    X_vertex = data['X_vertex']
    dof0  = int(np.size(X_vertex)/ndof0)
  
  if 'X_face_x' in data:
    X_face_x = data['X_face_x']
    X_face_z = data['X_face_y']
    dof1 = int(np.size(X_face_x)/ndof1x)
  
  if 'X_cell' in data:
    X_cell = data['X_cell']
    dof2 = int(np.size(X_cell)/ndof2)

  print('# Cells: x = %d z = %d' %(nx,nz))
  print('# DOFs: vertex = %d face = %d cell = %d' %(dof0,dof1,dof2))

  if (dof1) and (~dof2):
    x1d_cell   = (x1d_vertex[0:-1]+x1d_vertex[1:])*0.5
    z1d_cell   = (z1d_vertex[0:-1]+z1d_vertex[1:])*0.5

  # Clear variables
  del data

  print('# ---------------------------- ')
  print('# Printing:')

  # Print vertex data (1 component)
  if (dof0):
    print('#     -> vertex data')
    for i in range(dof0):
      # Split and reshape data
      Xvec = X_vertex[i::dof0]
      X = np.flip(np.reshape(Xvec,(nz+1,nx+1)),0)
      
      # Print data
      plt.figure(1, clear=True)
      extent = (x1d_vertex[0], x1d_vertex[-1], z1d_vertex[0], z1d_vertex[-1])
      h = plt.imshow(X, extent=extent, cmap=cmap_input, interpolation=interp_input)
      
      fname1 = fname+'_vertex_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvec, X

  # Print face data (2 components)
  if (dof1):
    print('#     -> face data')
    for i in range(dof1):
      # Split and reshape data
      Xvecx = X_face_x[i::dof1]
      Xvecz = X_face_z[i::dof1]
      Xx = np.flip(np.reshape(Xvecx,(nz,nx+1)),0)
      Xz = np.flip(np.reshape(Xvecz,(nz+1,nx)),0)
      
      # Print data - component 1
      plt.figure(1, clear=True)
      extent = (x1d_vertex[0], x1d_vertex[-1], z1d_cell[0], z1d_cell[-1])
      h = plt.imshow(Xx, extent=extent, cmap=cmap_input, interpolation=interp_input)
      
      fname1 = fname+'_face0_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)
      
      # Print data - component 1
      plt.figure(1, clear=True)
      extent = (x1d_cell[0], x1d_cell[-1], z1d_vertex[0], z1d_vertex[-1])
      h = plt.imshow(Xz, extent=extent, cmap=cmap_input, interpolation=interp_input)
      
      fname1 = fname+'_face1_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvecx, Xvecz, Xx, Xz

  # Print vertex data (1 component)
  if (dof2):
    print('#     -> cell data')
    for i in range(dof2):
      # Split and reshape data
      Xvec = X_cell[i::dof2]
      X = np.flip(np.reshape(Xvec,(nz,nx)),0)
      
      # Print data
      plt.figure(1, clear=True)
      extent = (x1d_cell[0], x1d_cell[-1], z1d_cell[0], z1d_cell[-1])
      h = plt.imshow(X, extent=extent, cmap=cmap_input, interpolation=interp_input)
      
      fname1 = fname+'_cell_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvec, X

  print('# ---------------------------- ')
  plt.close()


def general_output_pcolormesh(fname,cmap_input):
  print('# ---------------------------- ')
  print('# Loading PetscBinary data for plotting PCOLORMESH')
  print('# ---------------------------- ')

  # Load python module describing data
  imod = importlib.import_module(fname)

  # Load data
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  # Sort data into variables
  nx = data['Nx'][0]
  nz  = data['Ny'][0]

  # x1d = data['x1d']
  # z1d = data['y1d']

  # Compute dofs from the data
  ndof0  = (nx+1)*(nz+1)
  ndof1x = (nx+1)*nz
  ndof2  = nx*nz

  dof0 = 0
  dof1 = 0
  dof2 = 0

  if 'x1d_vertex' in data:
    x1d_vertex = data['x1d_vertex']
    z1d_vertex = data['y1d_vertex']

  if 'x1d_cell' in data:
    x1d_cell   = data['x1d_cell']
    z1d_cell   = data['y1d_cell']

  if 'X_vertex' in data:
    X_vertex = data['X_vertex']
    dof0  = int(np.size(X_vertex)/ndof0)
  
  if 'X_face_x' in data:
    X_face_x = data['X_face_x']
    X_face_z = data['X_face_y']
    dof1 = int(np.size(X_face_x)/ndof1x)
  
  if 'X_cell' in data:
    X_cell = data['X_cell']
    dof2 = int(np.size(X_cell)/ndof2)

  print('# Cells: x = %d z = %d' %(nx,nz))
  print('# DOFs: vertex = %d face = %d cell = %d' %(dof0,dof1,dof2))

  if (dof1) and (~dof2):
    x1d_cell   = (x1d_vertex[0:-1]+x1d_vertex[1:])*0.5
    z1d_cell   = (z1d_vertex[0:-1]+z1d_vertex[1:])*0.5

  # Clear variables
  del data

  print('# ---------------------------- ')
  print('# Printing:')

  # Print vertex data (1 component)
  if (dof0):
    print('#     -> vertex data')
    for i in range(dof0):
      # Split and reshape data
      Xvec = X_vertex[i::dof0]
      X = np.flip(np.reshape(Xvec,(nz+1,nx+1)),0)

      # Create grids
      xgrid, zgrid = np.meshgrid(x1d_vertex, z1d_vertex)
      
      # Print data
      plt.figure(1, clear=True)
      h = plt.pcolormesh(xgrid,zgrid,X,cmap=cmap_input)
      
      fname1 = fname+'_vertex_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvec, X, xgrid, zgrid

  # Print face data (2 components)
  if (dof1):
    print('#     -> face data')
    for i in range(dof1):
      # Split and reshape data
      Xvecx = X_face_x[i::dof1]
      Xvecz = X_face_z[i::dof1]
      Xx = np.flip(np.reshape(Xvecx,(nz,nx+1)),0)
      Xz = np.flip(np.reshape(Xvecz,(nz+1,nx)),0)

      # Create grids
      xgrid1, zgrid1 = np.meshgrid(x1d_vertex, z1d_cell)
      xgrid2, zgrid2 = np.meshgrid(x1d_cell, z1d_vertex)

      # Print data - component 1
      plt.figure(1, clear=True)
      h = plt.pcolormesh(xgrid1,zgrid1,Xx,cmap=cmap_input)
      
      fname1 = fname+'_face0_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)
      
      # Print data - component 1
      plt.figure(1, clear=True)
      h = plt.pcolormesh(xgrid2,zgrid2,Xz,cmap=cmap_input)
      
      fname1 = fname+'_face1_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvecx, Xvecz, Xx, Xz, xgrid1, zgrid1, xgrid2, zgrid2

  # Print vertex data (1 component)
  if (dof2):
    print('#     -> cell data')
    for i in range(dof2):
      # Split and reshape data
      Xvec = X_cell[i::dof2]
      X = np.flip(np.reshape(Xvec,(nz,nx)),0)

      # Create grids
      xgrid, zgrid = np.meshgrid(x1d_cell, z1d_cell)
      
      # Print data
      plt.figure(1, clear=True)
      extent = (x1d_cell[0], x1d_cell[-1], z1d_cell[0], z1d_cell[-1])
      h = plt.pcolormesh(xgrid,zgrid,X,cmap=cmap_input)
      
      fname1 = fname+'_cell_dof'+str(i)
      plt.xlabel('x-dir')
      plt.ylabel('z-dir')
      plt.title(fname1)
      plt.colorbar(h)
      
      fname_out = fname1+'.pdf'
      plt.savefig(fname_out)

      # Clear variables
      del Xvec, X, xgrid, zgrid

  print('# ---------------------------- ')
  plt.close()