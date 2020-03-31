
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, '+ \
    'PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS test to verify stokes-darcy and porosity evolution
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')
t = Symbol('t')

phi_0 = Symbol('phi_0')
p_s   = Symbol('p_s')
eta   = Symbol('eta')
zeta  = Symbol('zeta')

n     = Symbol('n')
m     = Symbol('m')
e3    = Symbol('e3') # unit vector of gravity

# Chosen solutions and coefficients
xi = zeta - 2/3*eta

# Pressure
# p = 1.0 
p = p_s*cos(m*pi*x)*cos(m*pi*z)
# p = p_s*cos(m*pi*x)*cos(m*pi*z)*(t*1.0e3+1.0) # Time-dependent

# Velocity using potential functions
psi_s = 1.0
U_s   = 1.0

psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
U   = -U_s*cos(m*pi*x)*cos(m*pi*z)
# psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))*(t*1.0e3+1.0) # Time-dependent
# U   = -U_s*cos(m*pi*x)*cos(m*pi*z)*(t*1.0e3+1.0)

curl_psix = diff(psi,z)
curl_psiz = diff(psi,x)
gradUx    = diff(U,x)
gradUz    = diff(U,z)

ux = curl_psix + gradUx
uz = curl_psiz + gradUz

# Constant velocity
# ux = 1.0
# uz = 1.0

# Porosity
Q = t**3*(x**2+z**2)
phi = 1.0-Q

# mobility
Kphi = (phi/phi_0)**n

# Compute right-hand-side
dpdx = diff(p,x)
dpdz = diff(p,z)

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

# MMS Solutions 
darcyx = -Kphi*(diff(p,x) - 0.0)
darcyz = -Kphi*(diff(p,z) - e3 )
div_darcy = diff(darcyx,x) + diff(darcyz,z)

fux = -diff(p,x) + eta*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + diff(xi*divu,x)
fuz = -diff(p,z) + eta*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + diff(xi*divu,z) + phi*e3
fp  = divu + div_darcy

fphi = diff(Q,t) + diff(Q*ux,x) + diff(Q*uz,z)

print('MMS solutions:')
print('\n')
write_c_method(p,'p')
write_c_method(ux,'ux')
write_c_method(uz,'uz')
write_c_method(phi,'phi')

write_c_method(fux,'fux')
write_c_method(fuz,'fuz')
write_c_method(fp,'fp')
write_c_method(fphi,'fphi')

print('\n')