def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phi_min, PetscScalar tau2, PetscScalar vzeta, '+ \
    'PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS for 2D two-phase flow example - 2-Field and 3-Field formulations
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')

delta = Symbol('delta')
phi0  = Symbol('phi0')
phi_min = Symbol('phi_min')
tau2  = Symbol('tau2')
vzeta = Symbol('vzeta')

p_s   = Symbol('p_s')
psi_s = Symbol('psi_s')
U_s   = Symbol('U_s')
n     = Symbol('n')
m     = Symbol('m')
k_hat = Symbol('k_hat') # unit vector of gravity

# Chosen solutions and coefficients
# porosity
# phi = phi0*exp(-(x*x+z*z)/tau2)
phi = phi0*(1.0+phi0*cos(m*pi*x)*cos(m*pi*z))

# pressure
p  = p_s*cos(m*pi*x)*cos(m*pi*z)
pc = p_s*sin(m*pi*x)*sin(m*pi*z)
pc = 0.0

psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
U   = -U_s*cos(m*pi*x)*cos(m*pi*z)

# calculate mms velocities from potentials
curl_psix = diff(psi,z)
curl_psiz = -diff(psi,x)
gradUx    = diff(U,x)
gradUz    = diff(U,z)

ux = curl_psix + gradUx
uz = curl_psiz + gradUz

# mobility
K = (phi/phi0)**n

# viscosity
eta = 1.0
# zeta = vzeta*1.0/(phi+phi_min)
zeta = vzeta
xi = (zeta-2.0/3.0*eta)
xi = 1.0

# Compute equations
dvxdx = diff(ux,x)
dvxdz = diff(ux,z)
dvzdz = diff(uz,z)
dvzdx = diff(uz,x)

dvx2dx = diff(dvxdx,x)
dvx2dz = diff(dvxdz,z)
dvz2dxdz = diff(dvzdx,z)

dvz2dx = diff(dvzdx,x)
dvz2dz = diff(dvzdz,z)
dvx2dzdx = diff(dvxdz,x)

divu  = diff(ux,x) + diff(uz,z)

# 2-Field
darcyx = -K*(diff(p,x) + 0.0)
darcyz = -K*(diff(p,z) + k_hat )
div_darcy = diff(darcyx,x) + diff(darcyz,z)

f2ux = -diff(p,x) + delta**2*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + delta**2*diff(xi*divu,x)
f2uz = -diff(p,z) + delta**2*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + delta**2*diff(xi*divu,z) - phi*k_hat
f2p  = divu + div_darcy

# 3-Field
darcyx3 = -K*(diff(p,x) + 0.0   + diff(pc,x))
darcyz3 = -K*(diff(p,z) + k_hat + diff(pc,z))
div_darcy3 = diff(darcyx3,x) + diff(darcyz3,z)

f3ux = -diff(p,x) + delta**2*(2.0*dvx2dx + dvx2dz + dvz2dxdz)
f3uz = -diff(p,z) + delta**2*(2.0*dvz2dz + dvz2dx + dvx2dzdx) - phi*k_hat
f3p  = divu + div_darcy3
f3pc = divu - pc/delta**2/xi

# Print solution
print('MMS solution:')
print('\n')
write_c_method(p,'p')
write_c_method(pc,'pc')
write_c_method(ux,'ux')
write_c_method(uz,'uz')

write_c_method(f2ux,'f2ux')
write_c_method(f2uz,'f2uz')
write_c_method(f2p,'f2p')

write_c_method(f3ux,'f3ux')
write_c_method(f3uz,'f3uz')
write_c_method(f3p,'f3p')
write_c_method(f3pc,'f3pc')

write_c_method(K,'K')
write_c_method(phi,'phi')
write_c_method(zeta,'zeta')
