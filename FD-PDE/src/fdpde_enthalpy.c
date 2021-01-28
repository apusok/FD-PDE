/* Finite Differences-PDE (FD-PDE) object for [ENTHALPY] */

#include "fdpde_enthalpy.h"

const char enthalpy_description[] =
"  << FD-PDE ENTHALPY >> solves the PDEs: \n"
"    dH/dt + A1*div(v *TP) + B1*div(vs*(1-phi)) + (div(C1*grad(TP)) + D1 = 0 \n"
"    dC/dt + A2*div(vs*(1-phi)*Cs) + B1*div(vf*phi*Cf) + (div(C2*phi*grad(Cf)) + D2 = 0 \n"
"  Notes: \n"
"  * Unknowns: H - enthalpy, C[] - composition (n-1 components). \n" 
"  * Other variables: TP - temperature, phi - porosity, Cf[],Cs[] - fluid and solid compositions. \n" 
"  * The coefficients A1, B1, C1, D1, v, vs, vf need to be defined by the user: \n" 
"        A1 - defined in center [dof COEFF_A1 = 0], \n" 
"        B1 - defined in center [dof COEFF_B1 = 1], \n" 
"        D1 - defined in center [dof COEFF_D1 = 2], \n" 
"        A2 - defined in center [dof COEFF_A2 = 3], \n" 
"        B2 - defined in center [dof COEFF_B2 = 4], \n" 
"        D2 - defined in center [dof COEFF_D2 = 5], \n" 
"        C1 - defined on faces [dof COEFF_C1 = 0], \n" 
"        C2 - defined on faces [dof COEFF_C2 = 1], \n" 
"        v  - bulk velocity defined on faces  [dof COEFF_v  = 2], \n" 
"        vf - fluid velocity defined on faces [dof COEFF_vf = 3], \n" 
"        vs - solid velocity defined on faces [dof COEFF_vs = 4], \n" 
"             *velocity coefficients can be solution from Stokes-Darcy. \n";

const char *AdvectSchemeTypeNames_Enthalpy[] = {
  "adv_uninit",
  "adv_none",
  "adv_upwind",
  "adv_upwind2",
  "adv_fromm"
};

const char *TimeStepSchemeTypeNames_Enthalpy[] = {
  "ts_uninit",
  "ts_none",
  "ts_forward_euler",
  "ts_backward_euler",
  "ts_crank_nicholson"
};

// ---------------------------------------
/*@
FDPDECreate_Enthalpy - creates the data structures for FDPDEType = ENTHALPY
Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate_Enthalpy"
PetscErrorCode FDPDECreate_Enthalpy(FDPDE fd)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(enthalpy_description,&fd->description); CHKERRQ(ierr);

  // ENTHALPY Stencil dofs: dmstag - H, C
  fd->dof0  = 0; fd->dof1  = 0; if (!fd->dof2) {fd->dof2 = 2;} 
  fd->dofc0 = 0; fd->dofc1 = 5; fd->dofc2 = 6;

  // Evaluation functions
  fd->ops->form_function      = FormFunction_Enthalpy;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_Enthalpy;
  fd->ops->view               = FDPDEView_Enthalpy;
  fd->ops->destroy            = FDPDEDestroy_Enthalpy;
  fd->ops->setup              = FDPDESetup_Enthalpy;

  // allocate memory to fd-pde context data
  ierr = PetscCalloc1(1,&en);CHKERRQ(ierr);

  en->xprev = NULL;
  en->coeffprev = NULL;

  en->form_enthalpy_method = NULL;
  en->form_TP      = NULL;
  en->form_user_bc = NULL; // PRELIM

  en->user_context   = NULL;
  en->user_context_bc= NULL;
  en->user_context_tp= NULL;
  
  en->ncomponents = fd->dof2;
  en->nreports = 0;
  en->description_enthalpy = NULL;
  en->dmP = NULL;
  en->xP = NULL;
  en->xPprev = NULL;

  // fd-pde context data
  fd->data = en;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDESetup_Enthalpy - setup some structures for FDPDEType = ENTHALPY
Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetup_Enthalpy"
PetscErrorCode FDPDESetup_Enthalpy(FDPDE fd)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  en = fd->data;

  // Create DMStag and vector for pressure/enthalpy
  ierr = DMStagCreateCompatibleDMStag(fd->dmstag,0,0,1,0,&en->dmP); CHKERRQ(ierr);
  ierr = DMSetUp(en->dmP); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(en->dmP,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(en->dmP,&en->xP);CHKERRQ(ierr);
  // initialize zero pressure vector
  ierr = VecSet(en->xP,0.0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEView_Enthalpy - view some info for FDPDEType = ENTHALPY
Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEView_Enthalpy"
PetscErrorCode FDPDEView_Enthalpy(FDPDE fd)
{
  EnthalpyData   *en;
  PetscFunctionBegin;

  en = fd->data;
  PetscPrintf(fd->comm,"[ENTHALPY] FDPDEView:\n");
  PetscPrintf(fd->comm,"  # Advection Scheme type: %s\n",AdvectSchemeTypeNames_Enthalpy[(int)en->advtype]);
  PetscPrintf(fd->comm,"  # Time step Scheme type: %s\n",TimeStepSchemeTypeNames_Enthalpy[(int)en->timesteptype]);
  PetscPrintf(fd->comm,"  # Theta: %g\n",en->theta);
  PetscPrintf(fd->comm,"  # Number chemical components: %d\n",en->ncomponents);
  PetscPrintf(fd->comm,"  # Enthalpy Method description:\n");
  PetscPrintf(fd->comm,"    %s\n",en->description_enthalpy);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEDestroy_Enthalpy - destroys the data structures for FDPDEType = ENTHALPY
Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEDestroy_Enthalpy"
PetscErrorCode FDPDEDestroy_Enthalpy(FDPDE fd)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // enthalpy data
  en = fd->data;
  if (en->xprev)     { ierr = VecDestroy(&en->xprev);CHKERRQ(ierr); }
  if (en->coeffprev) { ierr = VecDestroy(&en->coeffprev);CHKERRQ(ierr); }

  ierr = VecDestroy(&en->xP);CHKERRQ(ierr);
  ierr = DMDestroy(&en->dmP);CHKERRQ(ierr);
  if (en->xPprev) { ierr = VecDestroy(&en->xPprev);CHKERRQ(ierr); }

  en->form_enthalpy_method = NULL;
  en->form_TP        = NULL;
  en->form_user_bc   = NULL; // PRELIM

  en->user_context   = NULL;
  en->user_context_bc= NULL;
  en->user_context_tp= NULL;

  ierr = PetscFree(en->description_enthalpy);CHKERRQ(ierr);
  ierr = PetscFree(en);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
 JacobianCreate_Enthalpy - creates and preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_ENTHALPY
 Use: internal
 @*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianCreate_Enthalpy"
PetscErrorCode JacobianCreate_Enthalpy(FDPDE fd,Mat *J)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(fd->dmstag,J); CHKERRQ(ierr);
  ierr = JacobianPreallocator_Enthalpy(fd,*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
JacobianPreallocator_Enthalpy - preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_ENTHALPY
Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianPreallocator_Enthalpy"
PetscErrorCode JacobianPreallocator_Enthalpy(FDPDE fd,Mat J)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       ii, i, j, sx, sz, nx, nz; // local variables
  PetscInt       nEntries = STENCIL_ENTHALPY_NONZERO_PREALLOC; 
  Mat            preallocator = NULL;
  PetscScalar    *xx;
  DMStagStencil  *point;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Assign pointers and other variables
  Nx = fd->Nx;
  Nz = fd->Nz;

  // MatPreallocate begin
  ierr = MatPreallocatePhaseBegin(J, &preallocator); CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Zero entries
  ierr = PetscCalloc1(nEntries,&xx); CHKERRQ(ierr);
  ierr = PetscCalloc1(nEntries,&point); CHKERRQ(ierr);

  // Get non-zero pattern for preallocator
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      for (ii = 0; ii<fd->dof2; ii++) { // loop over all dofs
        ierr = EnthalpyNonzeroStencil(i,j,ii,Nx,Nz,point);CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  ierr = MatPreallocatePhaseEnd(J); CHKERRQ(ierr);

  // Matrix assembly
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // PetscPrintf(PETSC_COMM_WORLD,"# PREALLOCATOR \n");
  // ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);

  ierr = PetscFree(xx);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpyNonzeroStencil - calculates the non-zero pattern for the enthalpy equations/dof for JacobianPreallocator_Enthalpy()
Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyNonzeroStencil(PetscInt i,PetscInt j, PetscInt ii, PetscInt Nx, PetscInt Nz, DMStagStencil *point)
{
  PetscFunctionBegin;

  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = ii;

  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = ii; // WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = ii; // EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = ii; // SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = ii; // NN

  if (i == 0   ) point[1] = point[0];
  if (i == Nx-1) point[2] = point[0];
  if (j == 0   ) point[3] = point[0];
  if (j == Nz-1) point[4] = point[0];

  if (i <= 1   ) point[5] = point[0];
  if (i >= Nx-2) point[6] = point[0];
  if (j <= 1   ) point[7] = point[0];
  if (j >= Nz-2) point[8] = point[0];

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetAdvectSchemeType - set a method for the advection operator (ENTHALPY)

Input Parameter:
fd - the FD-PDE object
advtype - advection scheme type 

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetAdvectSchemeType"
PetscErrorCode FDPDEEnthalpySetAdvectSchemeType(FDPDE fd, AdvectSchemeType advtype)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine for setting Advection Type works only for FD-PDE Type = ENTHALPY!");
  en = fd->data;
  en->advtype = advtype;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetTimeStepSchemeType - set a method for time stepping (ENTHALPY)

Input Parameter:
fd - the FD-PDE object
timesteptype - time stepping scheme type 

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetTimeStepSchemeType"
PetscErrorCode FDPDEEnthalpySetTimeStepSchemeType(FDPDE fd, TimeStepSchemeType timesteptype)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine for setting TimeStepSchemeType should be used only for FD-PDE Type = ENTHALPY!");
  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  
  en = fd->data;
  en->timesteptype = timesteptype;

  // Assign timestepping algorithm
  switch (en->timesteptype) {
    case TS_UNINIT:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Time stepping scheme for the FD-PDE ENTHALPY was not set! Set with FDPDEEnthalpySetTimeStepSchemeType()");
    case TS_NONE:
      break;
    case TS_FORWARD_EULER:
      en->theta = 0.0;
      break;
    case TS_BACKWARD_EULER:
      en->theta = 1.0;
      break;
    case TS_CRANK_NICHOLSON:
      en->theta = 0.5;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown time stepping scheme for the FD-PDE ENTHALPY! Set with FDPDEEnthalpySetTimeStepSchemeType()");
  }

  if (en->timesteptype != TS_NONE) {
    // Create vectors for time-stepping if required
    ierr = VecDuplicate(fd->x,&en->xprev);CHKERRQ(ierr);
    ierr = VecDuplicate(fd->coeff,&en->coeffprev);CHKERRQ(ierr);
    ierr = VecDuplicate(en->xP,&en->xPprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetTimestep - set a time step size (ENTHALPY)

Input Parameters:
fd - the FD-PDE object
dt - time step size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetTimestep"
PetscErrorCode FDPDEEnthalpySetTimestep(FDPDE fd, PetscScalar dt)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  if (dt) en->dt = dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetTimestep - get the timestep size (ENTHALPY)

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dt - time step size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetTimestep"
PetscErrorCode FDPDEEnthalpyGetTimestep(FDPDE fd, PetscScalar *dt)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  if (dt) *dt = en->dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyComputeExplicitTimestep - function to compute the maximum allowed time step size for the grid for ENTHALPY equations

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dt - computed time step size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyComputeExplicitTimestep"
PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE fd, PetscScalar *dt)
{
  EnthalpyData   *en;
  PetscScalar    domain_dt, global_dt, eps, dx, dz, cell_dt, cell_dt_x, cell_dt_z;
  PetscInt       iprev=-1, inext=-1;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscScalar    **coordx, **coordz;
  DM             dmcoeff;
  Vec            coefflocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Check fd-pde for type and setup
  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  dmcoeff = fd->dmcoeff;

  ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  domain_dt = 1.0e32;
  eps = 1.0e-32; /* small shift to avoid dividing by zero */

  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over elements - max velocity(vs,vf)
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point[4];
      PetscScalar   xx_vs[4], xx_vf[4], vx_max, vz_max;

      point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = COEFF_vs;
      point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = COEFF_vs;
      point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = COEFF_vs;
      point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = COEFF_vs;
      ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,4,point,xx_vs); CHKERRQ(ierr);

      point[0].c = COEFF_vf;
      point[1].c = COEFF_vf;
      point[2].c = COEFF_vf;
      point[3].c = COEFF_vf;
      ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,4,point,xx_vf); CHKERRQ(ierr);

      // check max velocities
      vx_max = PetscMax(PetscMax(PetscAbsScalar(xx_vs[0]),PetscAbsScalar(xx_vs[1])), PetscMax(PetscAbsScalar(xx_vf[0]),PetscAbsScalar(xx_vf[1])));
      vz_max = PetscMax(PetscMax(PetscAbsScalar(xx_vs[2]),PetscAbsScalar(xx_vs[3])), PetscMax(PetscAbsScalar(xx_vf[2]),PetscAbsScalar(xx_vf[3])));

      dx = coordx[0][inext]-coordx[0][iprev];
      dz = coordz[0][inext]-coordz[0][iprev];

      // compute dx, dy for this cell
      cell_dt_x = dx / PetscMax(vx_max, eps);
      cell_dt_z = dz / PetscMax(vz_max, eps);
      cell_dt   = PetscMin(cell_dt_x,cell_dt_z);
      domain_dt = PetscMin(domain_dt,cell_dt);
    }
  }

  // MPI exchange global min/max
  ierr = MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dmcoeff));CHKERRQ(ierr);

  // Return vectors and arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

  // Return value
  *dt = global_dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPrevSolution - retrieves the previous time step solution vector from the FD-PDE object (ENTHALPY). 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
xprev - the previous time step solution vector

Notes:
Reference count on xprev is incremented. User must call VecDestroy() on xprev.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPrevSolution"
PetscErrorCode FDPDEEnthalpyGetPrevSolution(FDPDE fd, Vec *xprev)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  en = fd->data;

  if (xprev) {
    *xprev = en->xprev;
    ierr = PetscObjectReference((PetscObject)en->xprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPrevCoefficient - retrieves the previous time step coefficient vector from the FD-PDE object (ENTHALPY). 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
coeffprev - the previous time step coefficient vector

Notes:
Reference count on coeffprev is incremented. User must call VecDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPrevCoefficient"
PetscErrorCode FDPDEEnthalpyGetPrevCoefficient(FDPDE fd, Vec *coeffprev)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  en = fd->data;

  if (coeffprev) {
    *coeffprev = en->coeffprev;
    ierr = PetscObjectReference((PetscObject)en->coeffprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetNumberComponentsPhaseDiagram - set number of components for composition
Default: 2 components
Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetNumberComponentsPhaseDiagram"
PetscErrorCode FDPDEEnthalpySetNumberComponentsPhaseDiagram(FDPDE fd, PetscInt n)
{
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDEEnthalpySetNumberComponentsPhaseDiagram() before FDPDESetUp()");
  if (n>MAX_COMPONENTS) SETERRQ1(fd->comm,PETSC_ERR_SUP,"Supported only %d maximum chemical components!",MAX_COMPONENTS);
  fd->dof2  = n; 

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetEnthalpyMethod - set an evaluation function for the enthalpy method and phase diagram

Input Parameter:
fd - the FD-PDE object
form_enthalpy_method - name of the evaluation function for enthalpy method and phase diagram
Format: 
    form_enthalpy_method(H,C[],P,&TP,&T,&phi,CS,CF,usr)
description - the user can pass a char string to describe the enthalpy method
data - user context to be passed for evaluation (can be NULL)

Variables inside form_enthalpy_method:
    H  - enthalpy (input)
    C[]- bulk composition (input) (length ncomp)
    P  - pressure (input) needs to be specified by the user
    TP - primary temperature variable (input if TP-primary variable)
    T  - secondary temperature variable
    phi- porosity
    CS[] - solid composition (length ncomp)
    CF[] - fluid composition (length ncomp)
    ncomp - number of components
    user - user context provided

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetEnthalpyMethod"
PetscErrorCode FDPDEEnthalpySetEnthalpyMethod(FDPDE fd, EnthEvalErrorCode(*form_enthalpy_method)(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*), const char description[],void *data)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  en->form_enthalpy_method = form_enthalpy_method;
  en->user_context = data;
  if (description) { ierr = PetscStrallocpy(description,&en->description_enthalpy); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetPotentialTemp - set an evaluation function for potential function
Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetPotentialTemp"
PetscErrorCode FDPDEEnthalpySetPotentialTemp(FDPDE fd, PetscErrorCode(*form_TP)(PetscScalar,PetscScalar,PetscScalar*,void*),void *data)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  en->form_TP = form_TP;
  en->user_context_tp = data;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetUserBC - PRELIM because DMSTAGBCLIST does not support multiple dofs in same location
Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetUserBC"
PetscErrorCode FDPDEEnthalpySetUserBC(FDPDE fd, PetscErrorCode(*form_user_bc)(DM,Vec,PetscScalar***,void*),void *data)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  en->form_user_bc = form_user_bc;
  en->user_context_bc = data;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyUpdateDiagnostics - returns a dm/vector with all enthalpy variables updated

Input Parameter:
  fd - the FD-PDE object
  dm - the default enthalpy DM
  x  - solution vector (H,C)

Output Parameters: 
  dmnew - new DM
  xnew  - associated vector containing the Enthalpy Method variables

Enthalpy variables - should create labels for output
  H  - enthalpy
  C[]- bulk composition
  P  - pressure
  TP - primary temperature variable
  T  - secondary temperature variable
  phi- porosity
  CS[] - solid composition
  CF[] - fluid composition

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyUpdateDiagnostics"
PetscErrorCode FDPDEEnthalpyUpdateDiagnostics(FDPDE fd, DM dm, Vec x, DM *_dmnew, Vec *_xnew)
{
  PetscInt       i, j, ii, sx,sz,nx,nz,idx;
  PetscInt       dof_new, dof_sol;
  DM             dmnew;
  Vec            xnew, xlocal,xnewlocal;
  PetscScalar    H,C[MAX_COMPONENTS],P,phi,T,TP,CS[MAX_COMPONENTS],CF[MAX_COMPONENTS];
  PetscScalar    ***xx, *xE; 
  DMStagStencil  *pointE;
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  en = fd->data;
  if (!en->form_enthalpy_method) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"This routine requires a valid form_enthalpy_method() funtion pointer. Call FDPDEEnthalpySetEnthalpyMethod() first.");

  dof_sol = en->ncomponents;
  dof_new = 5 + 3*en->ncomponents;

  // create new dm with all variables in center
  ierr = DMStagCreateCompatibleDMStag(dm,0,0,dof_new,0,&dmnew); CHKERRQ(ierr);
  ierr = DMSetUp(dmnew); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmnew,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // create global vector
  ierr = DMCreateGlobalVector(dmnew,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmnew, &xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmnew, xnewlocal, &xx); CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  ierr = PetscCalloc1(dof_sol,&xE); CHKERRQ(ierr);
  ierr = PetscCalloc1(dof_sol,&pointE); CHKERRQ(ierr);

  // loop 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point;
      PetscInt      iX, ind;
      PetscScalar   sum_C = 0.0;
      EnthEvalErrorCode thermo_dyn_error_code;

      for (ii = 0; ii<dof_sol; ii++) {
        pointE[ii].i = i; pointE[ii].j = j; pointE[ii].loc = DMSTAG_ELEMENT; pointE[ii].c = ii;
      }
      ierr = DMStagVecGetValuesStencil(dm,xlocal,dof_sol,pointE,xE); CHKERRQ(ierr);
      
      // assign variables
      H = xE[0];
      for (ii = 1; ii<en->ncomponents; ii++) {
        sum_C  += xE[ii];
        C[ii-1] = xE[ii];
      }
      C[en->ncomponents-1] = 1.0 - sum_C;

      // calculate enthalpy method
      thermo_dyn_error_code = en->form_enthalpy_method(H,C,P,&T,&phi,CF,CS,en->ncomponents,en->user_context);CHKERRQ(ierr);
      if (thermo_dyn_error_code != 0) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SIG,"A successful Enthalpy Method is required but has failed! Investigate the enthalpy failure reports for detailed information.");
      }

      // update TP
      if (en->form_TP) { ierr = en->form_TP(T,P,&TP,en->user_context_tp);CHKERRQ(ierr); }
      else TP = T;

      point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; ind = -1;
      ind++; point.c = ind; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr); xx[j][i][iX] = H;
      ind++; point.c = ind; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr); xx[j][i][iX] = T;
      ind++; point.c = ind; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr); xx[j][i][iX] = TP;
      ind++; point.c = ind; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr); xx[j][i][iX] = phi;
      ind++; point.c = ind; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr); xx[j][i][iX] = P;

      // composition
      ind++;
      for (ii = 0; ii<en->ncomponents; ii++) {
        point.c = ind+ii; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr);
        xx[j][i][iX] = C[ii];
      }

      ind += en->ncomponents;
      for (ii = 0; ii<en->ncomponents; ii++) {
        point.c = ind+ii; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr);
        xx[j][i][iX] = CS[ii];
      }

      ind += en->ncomponents;
      for (ii = 0; ii<en->ncomponents; ii++) {
        point.c = ind+ii; ierr = DMStagGetLocationSlot(dmnew, point.loc, point.c, &iX); CHKERRQ(ierr);
        xx[j][i][iX] = CF[ii];
      }
    }
  }

  ierr = PetscFree(xE);CHKERRQ(ierr);
  ierr = PetscFree(pointE);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmnew,xnewlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmnew,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmnew,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  *_dmnew = dmnew;
  *_xnew  = xnew;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPressure - retrieves the pressure DMStag and Vector from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameters (optional):
dmP - the DM object
xP - the vector

Reference count on dmP and xP is incremented. User must call DMDestroy()/VecDestroy()!

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPressure"
PetscErrorCode FDPDEEnthalpyGetPressure(FDPDE fd, DM *dmP, Vec *xP)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  en = fd->data;

  if (dmP) { 
    *dmP = en->dmP;
    ierr = PetscObjectReference((PetscObject)en->dmP);CHKERRQ(ierr); 
  }
  if (xP) { 
    *xP  = en->xP;
    ierr = PetscObjectReference((PetscObject)en->xP);CHKERRQ(ierr); 
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPrevPressure - retrieves the previous time step pressure vector from the FD-PDE object (ENTHALPY). 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
Pprev - the previous time step pressure ector

Notes:
Reference count on Pprev is incremented. User must call VecDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPrevPressure"
PetscErrorCode FDPDEEnthalpyGetPrevPressure(FDPDE fd, Vec *Pprev)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  en = fd->data;

  if (Pprev) {
    *Pprev = en->xPprev;
    ierr = PetscObjectReference((PetscObject)en->xPprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}