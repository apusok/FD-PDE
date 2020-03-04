
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, '+ \
    'PetscScalar p_s, PetscScalar taux, PetscScalar tauz, PetscScalar x0, PetscScalar z0, PetscScalar m, PetscScalar n, PetscScalar e3)\n'
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

taux = Symbol('taux')
tauz = Symbol('tauz')
x0   = Symbol('x0')
z0   = Symbol('z0')

# Chosen solutions and coefficients
xi = zeta - 2/3*eta

# pressure
p = p_s*cos(m*pi*x)*cos(m*pi*z)

dpdx = diff(p,x)
dpdz = diff(p,z)

# velocity
ux = dpdx + sin(m*pi*x)*sin(m*pi*z)
uz = dpdz + cos(m*pi*x)*cos(m*pi*z)

# porosity
# phi = phi_0*(1.0+phi_s*cos(m*pi*x)*cos(m*pi*z))
phi = phi_0*exp(-((x-x0-ux*t)/taux)**2-((z-z0-uz*t)/tauz)**2)

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

fux = -diff(p,x) + eta*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + xi*diff(divu,x)
fuz = -diff(p,z) + eta*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + xi*diff(divu,z) + phi*e3
fp  = divu + div_darcy

fphi = diff(1-phi,t) + diff((1-phi)*ux,x) + diff((1-phi)*uz,z)

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

# write_c_method(Kphi,'Kphi')

print('\n')