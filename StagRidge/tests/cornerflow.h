
static void evaluate_CornerFlow_MOR(PetscScalar A, PetscScalar B, PetscScalar x, PetscScalar z, PetscScalar v[], PetscScalar *_p)
/* Input parameters: constants A, B (model specific), coordinates x, z
   Output parameters: velocity v=[vx,vz], pressure p 
   Polar coordinates - defined as x=r*cos_theta, z=-r*sin_theta */
{
  PetscScalar    vr, vth, r, p;
  PetscScalar    sinth, costh, th;

  // polar coordinates
  r  = PetscPowScalar(x*x+z*z,0.5);
  th = PetscAtanReal(-z/x);
  sinth = -z/r;
  costh = x/r;

  vr = A*costh - B*costh + B*th*sinth;
  vth = -A*sinth + B*th*costh;

  // vx velocity
  v[0] = vr*costh - vth*sinth;

  // vz velocity
  v[1] = -vr*sinth - vth*costh;

  // pressure
  p   = -2*B/r*costh;
  *_p = p;

}
