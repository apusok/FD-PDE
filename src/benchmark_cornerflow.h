
static void evaluate_CornerFlow_MOR(PetscScalar C1, PetscScalar C4, PetscScalar u0, PetscScalar eta0, PetscScalar x, PetscScalar z, PetscScalar v[], PetscScalar *_p)
/* Input parameters: constants C1, C4 (model specific), coordinates x, z
   Output parameters: velocity v=[vx,vz], pressure p 
   Polar coordinates - defined as x=r*sin_theta, z=-r*cos_theta 
   Note: z-axis inverted from solution in Spiegelman and McKenzie (1987) */
{
  PetscScalar    vr, vth, r, p;
  PetscScalar    sinth, costh, th;

  // polar coordinates
  r  = PetscPowScalar(x*x+z*z,0.5);
  //th = PetscAtanReal(-x/z);
  th = PetscAsinReal(x/r);  //use arcsin instead to avoid INF when z = 0
  sinth = x/r;
  costh = -z/r;

  vr = u0*(C1*costh + C4*costh - C4*th*sinth);
  vth = u0*(-C1*sinth - C4*th*costh);

  // vx velocity
  v[0] = vr*sinth + vth*costh;

  // vz velocity
  v[1] = -vr*costh + vth*sinth;

  // pressure
  p   = 2*C4*eta0*u0*costh/r;
  *_p = p;

}
