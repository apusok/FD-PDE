print('# --------------------------------------- #')
print('# Mid-ocean ridge model - Buoyancy ')
print('# --------------------------------------- #')

# Import modules
import os
import sys
import numpy as np

# Add path to vizMORBuoyancy
pathViz = '../python/'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct()

# Parameters
A.modelopts    = 'model_test.opts'
A.nx           = 200
A.nz           = 100
A.K0           = 1.0e-9
A.zeta0        = 1.0e20
A.buoyancy     = 0 # 0=off, 1=phi, 2=phi-C, 3=phi-C-T
A.visc         = 0 # 0-constant, 1-Temp,porosity dependent
A.k_hat        = -1.0 # 0.0 or -1.0
A.phi_init     = 0.0  # 0=full extraction, 1e-4-some porosity left
A.dim_output   = 1 # 0 - nondimensional, 1 - dimensional
A.debug_output = 0 # 0 - no debug output, 1 - debug output
A.xsill        = 6e3

A.tmax  = 2.0e6 # yr
A.dtmax = 1.0e1 # yr
A.tout  = 1
A.tstep = 1

A.Tp = 1648 #1573 #1623
A.potentialtemp = 0

# Run test
str1 = '../MORbuoyancy.app'+ \
    ' -options_file '+A.modelopts+ \
    ' -nx '+str(A.nx)+' -nz '+str(A.nz)+ \
    ' -tstep '+str(A.tstep)+ \
    ' -tout '+str(A.tout)+ \
    ' -tmax '+str(A.tmax)+ \
    ' -dtmax '+str(A.dtmax)+ \
    ' -K0 '+str(A.K0)+ \
    ' -zeta0 '+str(A.zeta0)+ \
    ' -xsill '+str(A.xsill)+ \
    ' -buoyancy '+str(A.buoyancy)+ \
    ' -visc '+str(A.visc)+ \
    ' -phi_init '+str(A.phi_init)+ \
    ' -Tp '+str(A.Tp)+ \
    ' -k_hat '+str(A.k_hat)+ \
    ' -potentialtemp '+str(A.potentialtemp)+ \
    ' -log_view > log_run.out'
print(str1)
os.system(str1)

# Create visualization directories
if (A.debug_output):
  A.fname_debug_out = 'debug_output'
  try:
    os.mkdir(A.fname_debug_out)
  except OSError:
    pass

# Read parameters file and get scaling params
A.scal, A.lbl = vizB.get_scaling_labels('parameters_file.out','Timestep0',A.dim_output)

# Read grid parameters - choose PV file timestep0
A.grid = vizB.parse_grid_info('out_xPV_ts0','Timestep0')
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
# print(A.__dict__)

# Plot sill outflux
A.tstep_read, A.sill = vizB.parse_log_file_sill('log_run.out')
vizB.plot_sill_outflux(A,'out_sill_flux')

# Visualize data
for istep in range(0,A.tstep+1):
  fdir  = 'Timestep'+str(istep)

  # Get data
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.H, A.C = vizB.parse_HC_file('out_xHC_ts'+str(istep),fdir)
  A.phi, A.T = vizB.parse_HC_file('out_xphiT_ts'+str(istep),fdir)
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.HC_coeff = vizB.parse_HCcoeff_file('out_xHCcoeff_ts'+str(istep),fdir)

  if (istep > 0):
    A.PV_coeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
    A.resP, A.resVsx, A.resVsz = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
    A.resH, A.resC = vizB.parse_HC_file('out_resHC_ts'+str(istep),fdir)
    A.matProp = vizB.parse_matProps_file('out_matProp_ts'+str(istep),fdir)

  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
  A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)
  A.divmass = vizB.calc_divergence(A.Vx,A.Vz,A.dx,A.dz,A.nx,A.nz)

  # nice output
  vizB.plot_porosity_contours(A,'out_porosity_contours_ts'+str(istep),istep)
  vizB.plot_temperature_slices(A,'out_temp_slices_ts'+str(istep),istep)

  # debug output
  if (A.debug_output):
    vizB.plot_PV(0,A,A.fname_debug_out+'/out_xPV_ts'+str(istep),istep,7,12)
    vizB.plot_HC(0,A,A.fname_debug_out+'/out_xHC_ts'+str(istep),istep,7,8) # H,C
    vizB.plot_HC(1,A,A.fname_debug_out+'/out_xphiT_ts'+str(istep),istep,7,8) # phi,T
    vizB.plot_Enth(A,A.fname_debug_out+'/out_xEnth_ts'+str(istep),istep)
    vizB.plot_Vel(A,A.fname_debug_out+'/out_xVel_ts'+str(istep),istep)
    vizB.plot_HCcoeff(A,A.fname_debug_out+'/out_xHCcoeff_ts'+str(istep),istep)

    if (istep > 0):
      vizB.plot_PVcoeff(A,A.fname_debug_out+'/out_xPVcoeff_ts'+str(istep),istep)
      vizB.plot_PV(1,A,A.fname_debug_out+'/out_resPV_ts'+str(istep),istep,7,12) # res PV
      vizB.plot_HC(2,A,A.fname_debug_out+'/out_resHC_ts'+str(istep),istep,7,8) # res HC
      vizB.plot_matProp(A,A.fname_debug_out+'/out_matProp_ts'+str(istep),istep)

  fname = 'out_divmass_ts'+str(istep)
  os.system('rm -r Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')
