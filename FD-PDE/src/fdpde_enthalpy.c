/* Finite Differences-PDE (FD-PDE) object for [ENTHALPY] */

#include "fdpde_enthalpy.h"

const char enthalpy_description[] =
"  << FD-PDE ENTHALPY >> solves the PDEs: \n"
"  Notes: \n"
"  * Unknowns: Q - can be temperature. \n" 
"  * The coefficients A,B,C,D,u need to be defined by the user. \n" 
"        A, B - advection term coefficient, defined in center, \n" 
"        C - diffusion term coefficient, defined on edges, \n" 
"        D - source/sink term, defined in center, \n" 
"        u - velocity, defined on edges (can be solution from Stokes (-Darcy) equations). \n";

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
  fd->dofc0 = 0; fd->dofc1 = 5; fd->dofc2 = 16;

  // Evaluation functions
  fd->ops->form_function      = FormFunction_Enthalpy;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_Enthalpy;
  fd->ops->view               = FDPDEView_Enthalpy;
  fd->ops->destroy            = FDPDEDestroy_Enthalpy;
  fd->ops->setup              = FDPDESetUp_Enthalpy;

  // allocate memory to fd-pde context data
  ierr = PetscCalloc1(1,&en);CHKERRQ(ierr);

  // vectors
  en->xprev = NULL;
  en->coeffprev = NULL;

  // enthalpy other data
  en->dmphiT  = NULL;
  en->xphiT   = NULL;
  en->dmcomp  = NULL;
  en->xCS     = NULL;
  en->xCF     = NULL;
  en->form_CS = NULL;
  en->form_CF = NULL;
  en->user_context = NULL;

  en->ncomponents = fd->dof2;
  en->energy_variable = 0; // default 0-H-enthalpy, 1-TP-temperature

  // fd-pde context data
  fd->data = en;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDESetUp_Enthalpy - set-up additional data structures for FDPDEType = ENTHALPY

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetUp_Enthalpy"
PetscErrorCode FDPDESetUp_Enthalpy(FDPDE fd)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  en = fd->data;

  // Create DMStag object for dmphiT, dmcomp
  ierr = DMStagCreateCompatibleDMStag(fd->dmstag,0,0,2,0,&en->dmphiT); CHKERRQ(ierr);
  ierr = DMSetUp(en->dmphiT); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(en->dmphiT,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);

  ierr = DMStagCreateCompatibleDMStag(fd->dmstag,0,0,(en->ncomponents-1),0,&en->dmcomp); CHKERRQ(ierr);
  ierr = DMSetUp(en->dmcomp); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(en->dmcomp,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);

  // Create global vectors
  ierr = DMCreateGlobalVector(en->dmphiT,&en->xphiT);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(en->dmcomp,&en->xCS);CHKERRQ(ierr);
  ierr = VecDuplicate(en->xCS,&en->xCF);CHKERRQ(ierr);

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

  en = fd->data;
  if (en->xprev)     { ierr = VecDestroy(&en->xprev);CHKERRQ(ierr); }
  if (en->coeffprev) { ierr = VecDestroy(&en->coeffprev);CHKERRQ(ierr); }

  // enthalpy data
  if (en->dmphiT) { ierr = DMDestroy(&en->dmphiT);CHKERRQ(ierr); }
  if (en->xphiT) { ierr = VecDestroy(&en->xphiT);CHKERRQ(ierr); }

  if (en->dmcomp) { ierr = DMDestroy(&en->dmcomp);CHKERRQ(ierr); }
  if (en->form_CS) { en->form_CS = NULL; }
  if (en->form_CF) { en->form_CF = NULL; }
  en->user_context = NULL;
  if (en->xCS)     { ierr = VecDestroy(&en->xCS);CHKERRQ(ierr); }
  if (en->xCF)     { ierr = VecDestroy(&en->xCF);CHKERRQ(ierr); }

  ierr = PetscFree(en);CHKERRQ(ierr);

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
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetTimestep - set a time step size and/or the CFL value (ENTHALPY)

Input Parameters:
fd - the FD-PDE object
dt - time stepping size
dtflg - if true, dt = min(dt,max_dt_grid) where max_dt_grid is max allowed timestep on grid. if false, dt=dt

Note:
The time step size will be checked against the max allowed timestep on grid such that dt <= max_dt_grid.

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
FDPDEEnthalpyGetTimestep - get the timestep (ENTHALPY)

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dt - time stepping size

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

// // ---------------------------------------
// /*@
// FDPDEEnthalpyComputeExplicitTimestep - function to update time step size for ENTHALPY equations

// Input Parameter:
// fd - the FD-PDE object

// Output Parameter:
// dt - time stepping size

// Use: user
// @*/
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "FDPDEEnthalpyComputeExplicitTimestep"
// PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE fd, PetscScalar *dt)
// {
//   EnthalpyData   *en;
//   PetscScalar    domain_dt, global_dt, eps, dx, dz, cell_dt, cell_dt_x, cell_dt_z;
//   PetscInt       iprev=-1, inext=-1;
//   PetscInt       i, j, sx, sz, nx, nz;
//   PetscScalar    **coordx, **coordz;
//   DM             dmcoeff;
//   Vec            coefflocal;
//   PetscErrorCode ierr;
//   PetscFunctionBeginUser;

//   // Check fd-pde for type and setup
//   if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
//   if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

//   en = fd->data;
//   dmcoeff = fd->dmcoeff;

//   ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

//   domain_dt = 1.0e32;
//   eps = 1.0e-32; /* small shift to avoid dividing by zero */

//   ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
//   ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

//   ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev);CHKERRQ(ierr); 
//   ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

//   // Loop over elements - velocity is located on edge and c=1,2,3 - NEED TO UPDATE AS A FUNCTION OF ALL VELOCITY/or just temperature?
//   for (j = sz; j<sz+nz; j++) {
//     for (i = sx; i<sx+nx; i++) {
//       DMStagStencil point[4];
//       PetscScalar   xx[4];

//       point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 1;
//       point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 1;
//       point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 1;
//       point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 1;

//       ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,4,point,xx); CHKERRQ(ierr);

//       dx = coordx[0][inext]-coordx[0][iprev];
//       dz = coordz[0][inext]-coordz[0][iprev];

//       /* compute dx, dy for this cell */
//       cell_dt_x = dx / PetscMax(PetscMax(PetscAbsScalar(xx[0]), PetscAbsScalar(xx[1])), eps);
//       cell_dt_z = dz / PetscMax(PetscMax(PetscAbsScalar(xx[2]), PetscAbsScalar(xx[3])), eps);
//       cell_dt   = PetscMin(cell_dt_x,cell_dt_z);
//       domain_dt = PetscMin(domain_dt,cell_dt);
//     }
//   }

//   // MPI exchange global min/max
//   ierr = MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dmcoeff));CHKERRQ(ierr);

//   // Return vectors and arrays
//   ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

//   // Return value
//   *dt = global_dt;

//   PetscFunctionReturn(0);
// }

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
  PetscInt       ii, i, j, sx, sz, nx, nz, nEntries = 9; // local variables
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
FDPDEEnthalpySetFunctionsPhaseDiagram - set an evaluation functions for the phase diagram

Input Parameter:
fd - the FD-PDE object
form_CF - name of the evaluation function for fluid composition (liquidus)
form_CS - name of the evaluation function for solid composition (solidus)
data - user context to be passed for evaluation (can be NULL)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetFunctionsPhaseDiagram"
PetscErrorCode FDPDEEnthalpySetFunctionsPhaseDiagram(FDPDE fd, PetscErrorCode (*form_CF)(FDPDE fd,DM,Vec,DM,Vec,DM,Vec,void*), PetscErrorCode (*form_CS)(FDPDE fd,DM,Vec,DM,Vec,DM,Vec,void*), void *data)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  en = fd->data;
  en->form_CF = form_CF;
  en->form_CS = form_CS;
  en->user_context = data;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetNumberComponentsPhaseDiagram - set number of components for composition

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
  fd->dof2  = n; 

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpySetEnergyPrimaryVariable - set either H or TP as primary energy variable

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpySetEnergyPrimaryVariable"
PetscErrorCode FDPDEEnthalpySetEnergyPrimaryVariable(FDPDE fd, const char energy_variable)
{
  EnthalpyData   *en;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call this routine after FDPDESetUp()!");
  
  en = fd->data;
  switch (energy_variable) {
    case 'H':
      en->energy_variable = 0;
      break;
    case 'T':
      en->energy_variable = 1;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Energy primary variable supported must be one of {'H','T'}");
      break;
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPorosityTemperature - retrieves the DM and Vector for phi, T from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameters (optional):
dmphiT - the DM object
xphiT - the vector

Reference count on both dm/vector is incremented. User must call DM/VecDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPorosityTemperature"
PetscErrorCode FDPDEEnthalpyGetPorosityTemperature(FDPDE fd, DM *dmphiT, Vec *xphiT)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  en = fd->data;

  if (dmphiT) {
    *dmphiT = en->dmphiT;
    ierr = PetscObjectReference((PetscObject)en->dmphiT);CHKERRQ(ierr);
  }

  if (xphiT) {
    *xphiT = en->xphiT;
    ierr = PetscObjectReference((PetscObject)en->xphiT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEEnthalpyGetPhaseComposition - retrieves the DM and Vector for Cf, Cs from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameters (optional):
dmcomp - the DM object
xCF - the fluid composition vector; indexing goes (Cf)^i, where i is the component index
xCS - the solid composition vector; indexing goes (Cs)^i, where i is the component index

Reference count on both dm/vectors is incremented. User must call DM/VecDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEEnthalpyGetPhaseComposition"
PetscErrorCode FDPDEEnthalpyGetPhaseComposition(FDPDE fd, DM *dmcomp, Vec *xCF, Vec *xCS)
{
  EnthalpyData   *en;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ENTHALPY) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ENTHALPY!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  en = fd->data;

  if (dmcomp) {
    *dmcomp = en->dmcomp;
    ierr = PetscObjectReference((PetscObject)en->dmcomp);CHKERRQ(ierr);
  }

  if (xCF) {
    *xCF = en->xCF;
    ierr = PetscObjectReference((PetscObject)en->xCF);CHKERRQ(ierr);
  }

  if (xCS) {
    *xCS = en->xCS;
    ierr = PetscObjectReference((PetscObject)en->xCS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}