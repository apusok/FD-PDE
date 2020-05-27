
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, '+ \
    'PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, '+ \
    'PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS test to verify a power law effective viscosity for Stokes (single phase) and StokesDarcy (two-phase)
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')

eta0  = Symbol('eta0')
eps0  = Symbol('eps0')
np    = Symbol('np') # power-law exponent

phi_0 = Symbol('phi_0')
phi_s = Symbol('phi_s')
p_s   = Symbol('p_s')
psi_s = Symbol('psi_s')
U_s   = Symbol('U_s')
n     = Symbol('n') # permeability exponent
m     = Symbol('m')
k_hat = Symbol('k_hat') # unit vector of gravity
R     = Symbol('R')

# Pressure
p = p_s*cos(m*pi*x)*cos(m*pi*z)

# Velocity using potential functions
psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
U   = -U_s*cos(m*pi*x)*cos(m*pi*z)

curl_psix = diff(psi,z)
curl_psiz = -diff(psi,x)
gradUx    = diff(U,x)
gradUz    = diff(U,z)

# ux = curl_psix + gradUx
# uz = curl_psiz + gradUz

dpdx = diff(p,x)
dpdz = diff(p,z)

k_us = 0.5
k_ls = 0.5
k = 0.25*(k_us-k_ls)/tanh(5.0)*( 2.0 + tanh(10.0*x-5.0) + tanh(10.0*z-5.0) + (2.0*(k_us-k_ls)-2.0*tanh(5.0)*(k_ls+k_us))/(k_ls-k_us) )
ux = k*dpdx + sin(pi*x)*sin(2.0*pi*z) + 2.0
uz = k*dpdz + 0.5*cos(pi*x)*cos(2.0*pi*z) + 2.0

# porosity
phi = phi_0*(1.0+phi_s*cos(m*pi*x)*cos(m*pi*z))

# mobility
Kphi = (phi/phi_0)**n

# Compute right-hand-side
dpdx = diff(p,x)
dpdz = diff(p,z)

# Strain rates
exx = diff(ux,x)
ezz = diff(uz,z)
exz = 0.5*(diff(ux,z)+diff(uz,x))
epsII = (0.5*(exx**2+ezz**2+2*exz**2))**0.5

# Effective viscosity
eta = eta0*(epsII/eps0)**(1/np-1)

tauxx = 2.0*eta*exx
tauxz = 2.0*eta*exz
tauzz = 2.0*eta*ezz

divu  = exx+ezz

# MMS Solutions - Stokes
fux_stokes = -diff(p,x) + diff(tauxx,x) + diff(tauxz,z)
fuz_stokes = -diff(p,z) + diff(tauzz,z) + diff(tauxz,x)
fp_stokes  = divu

# MMS Solutions - Stokes Darcy
zeta = eta/phi
xi   = zeta - 2/3*eta

darcyx = -Kphi*(diff(p,x) - 0.0)
darcyz = -Kphi*(diff(p,z) - k_hat )
div_darcy = diff(darcyx,x) + diff(darcyz,z)

fux_stokesdarcy = -diff(p,x) + diff(tauxx,x) + diff(tauxz,z) + diff(xi*divu,x)
fuz_stokesdarcy = -diff(p,z) + diff(tauzz,z) + diff(tauxz,x) + diff(xi*divu,z) + phi*k_hat
fp_stokesdarcy  = divu + R**2*div_darcy

print('MMS solutions:')
print('\n')
write_c_method(p,'p')
write_c_method(ux,'ux')
write_c_method(uz,'uz')
write_c_method(phi,'phi')
write_c_method(Kphi,'Kphi')

# write_c_method(exx,'exx')
# write_c_method(ezz,'ezz')
# write_c_method(exz,'exz')

write_c_method(fux_stokes,'fux_stokes')
write_c_method(fuz_stokes,'fuz_stokes')
write_c_method(fp_stokes,'fp_stokes')

write_c_method(fux_stokesdarcy,'fux_stokesdarcy')
write_c_method(fuz_stokesdarcy,'fuz_stokesdarcy')
write_c_method(fp_stokesdarcy,'fp_stokesdarcy')

print('\n')