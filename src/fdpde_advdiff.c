/* Finite Differences-PDE (FD-PDE) object for [ADVDIFF] */

#include "fdpde_advdiff.h"

const char advdiff_description[] =
"  << FD-PDE ADVDIFF >> solves the PDEs: \n"
"    A(dQ/dt + div (uQ)) - div(B grad Q) + C = 0 \n"
"    OR \n"
"    dQ/dt + 1/A SD(Q) = 0, where \n"
"    SD(Q) = A(div (uQ)) - div(B grad Q) + C, is the steady state solution. \n"
"  Notes: \n"
"  * Unknowns: Q - can be temperature. \n" 
"  * The coefficients A,B,C,u need to be defined by the user. \n" 
"        A (i.e., density * heat capacity) - defined in center, \n" 
"        B (i.e., thermal conductivity) - defined on edges, \n" 
"        C (i.e., sources of heat production/sink) - defined in center, \n" 
"        u - velocity, defined on edges (can be solution from Stokes (-Darcy) equations). \n";

const char *AdvectSchemeTypeNames[] = {
  "adv_uninit",
  "adv_none",
  "adv_upwind",
  "adv_upwind2",
  "adv_fromm",
  "adv_upwind_minmod"
};

const char *TimeStepSchemeTypeNames[] = {
  "ts_uninit",
  "ts_none",
  "ts_forward_euler",
  "ts_backward_euler",
  "ts_crank_nicholson"
};

// ---------------------------------------
/*@
FDPDECreate_AdvDiff - creates the data structures for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate_AdvDiff"
PetscErrorCode FDPDECreate_AdvDiff(FDPDE fd)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(advdiff_description,&fd->description); CHKERRQ(ierr);

  // ADVDIFF Stencil dofs: dmstag - Q (element)
  fd->dof0  = 0; fd->dof1  = 0; fd->dof2  = 1; 
  fd->dofc0 = 0; fd->dofc1 = 2; fd->dofc2 = 2;

  // Evaluation functions
  fd->ops->form_function      = FormFunction_AdvDiff;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_AdvDiff;
  fd->ops->view               = FDPDEView_AdvDiff;
  fd->ops->destroy            = FDPDEDestroy_AdvDiff;
  fd->ops->setup              = NULL;

  // allocate memory to fd-pde context data
  ierr = PetscCalloc1(1,&ad);CHKERRQ(ierr);

  // vectors
  ad->xprev = NULL;
  ad->coeffprev = NULL;

  // fd-pde context data
  fd->data = ad;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEDestroy_AdvDiff - destroys the data structures for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEDestroy_AdvDiff"
PetscErrorCode FDPDEDestroy_AdvDiff(FDPDE fd)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ad = fd->data;

  if (ad->xprev)     { ierr = VecDestroy(&ad->xprev);CHKERRQ(ierr); }
  if (ad->coeffprev) { ierr = VecDestroy(&ad->coeffprev);CHKERRQ(ierr); }

  ierr = PetscFree(ad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEView_AdvDiff - view some info for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEView_AdvDiff"
PetscErrorCode FDPDEView_AdvDiff(FDPDE fd)
{
  AdvDiffData    *ad;
  PetscFunctionBegin;

  ad = fd->data;

  PetscPrintf(fd->comm,"[ADVDIFF] FDPDEView:\n");
  PetscPrintf(fd->comm,"  # Advection Scheme type: %s\n",AdvectSchemeTypeNames[(int)ad->advtype]);
  PetscPrintf(fd->comm,"  # Time step Scheme type: %s\n",TimeStepSchemeTypeNames[(int)ad->timesteptype]);
  PetscPrintf(fd->comm,"  # Theta: %g\n",ad->theta);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffGetPrevSolution - retrieves the previous time step solution vector from the FD-PDE object (ADVDIFF). 

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
#define __FUNCT__ "FDPDEAdvDiffGetPrevSolution"
PetscErrorCode FDPDEAdvDiffGetPrevSolution(FDPDE fd, Vec *xprev)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  ad = fd->data;

  if (xprev) {
    *xprev = ad->xprev;
    ierr = PetscObjectReference((PetscObject)ad->xprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffGetPrevCoefficient - retrieves the previous time step coefficient vector from the FD-PDE object (ADVDIFF). 

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
#define __FUNCT__ "FDPDEAdvDiffGetPrevCoefficient"
PetscErrorCode FDPDEAdvDiffGetPrevCoefficient(FDPDE fd, Vec *coeffprev)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  ad = fd->data;

  if (coeffprev) {
    *coeffprev = ad->coeffprev;
    ierr = PetscObjectReference((PetscObject)ad->coeffprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffSetAdvectSchemeType - set a method for the advection operator (ADVDIFF)

Input Parameter:
fd - the FD-PDE object
advtype - advection scheme type 

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEAdvDiffSetAdvectSchemeType"
PetscErrorCode FDPDEAdvDiffSetAdvectSchemeType(FDPDE fd, AdvectSchemeType advtype)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"The Advection Type should be set only for FD-PDE Type = ADVDIFF!");
  ad = fd->data;
  ad->advtype = advtype;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffSetTimeStepSchemeType - set a method for time stepping (ADVDIFF)

Input Parameter:
fd - the FD-PDE object
timesteptype - time stepping scheme type 

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEAdvDiffSetTimeStepSchemeType"
PetscErrorCode FDPDEAdvDiffSetTimeStepSchemeType(FDPDE fd, TimeStepSchemeType timesteptype)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"The TimeStepSchemeType should be set only for FD-PDE Type = ADVDIFF!");
  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  
  ad = fd->data;
  ad->timesteptype = timesteptype;

  // Assign timestepping algorithm
  switch (ad->timesteptype) {
    case TS_UNINIT:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Time stepping scheme for the FD-PDE ADVDIFF was not set! Set with FDPDEAdvDiffSetTimeStepSchemeType()");
    case TS_NONE:
      break;
    case TS_FORWARD_EULER:
      ad->theta = 0.0;
      break;
    case TS_BACKWARD_EULER:
      ad->theta = 1.0;
      break;
    case TS_CRANK_NICHOLSON:
      ad->theta = 0.5;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown time stepping scheme for the FD-PDE ADVDIFF! Set with FDPDEAdvDiffSetTimeStepSchemeType()");
  }

  if (ad->timesteptype != TS_NONE) {
    // Create vectors for time-stepping if required
    ierr = VecDuplicate(fd->x,&ad->xprev);CHKERRQ(ierr);
    ierr = VecDuplicate(fd->coeff,&ad->coeffprev);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffSetTimestep - set a time step size (ADVDIFF)

Input Parameters:
fd - the FD-PDE object
dt - time stepping size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEAdvDiffSetTimestep"
PetscErrorCode FDPDEAdvDiffSetTimestep(FDPDE fd, PetscScalar dt)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  if (dt) ad->dt = dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffGetTimestep - get the timestep (ADVDIFF)

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dt - time stepping size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEAdvDiffGetTimestep"
PetscErrorCode FDPDEAdvDiffGetTimestep(FDPDE fd, PetscScalar *dt)
{
  AdvDiffData    *ad;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  if (dt) *dt = ad->dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDEAdvDiffComputeExplicitTimestep - function to update time step size for ADVDIFF equations

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dt - time stepping size

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEAdvDiffComputeExplicitTimestep"
PetscErrorCode FDPDEAdvDiffComputeExplicitTimestep(FDPDE fd, PetscScalar *dt)
{
  AdvDiffData    *ad;
  PetscScalar    domain_dt, global_dt, eps, dx, dz, cell_dt, cell_dt_x, cell_dt_z;
  PetscInt       iprev=-1, inext=-1;
  PetscInt       i, j, sx, sz, nx, nz, v_slot[4];
  PetscScalar    **coordx, **coordz, ***_coeff;
  DM             dmcoeff;
  Vec            coefflocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Check fd-pde for type and setup
  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  dmcoeff = fd->dmcoeff;

  ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmcoeff,coefflocal,&_coeff);CHKERRQ(ierr);

  domain_dt = 1.0e32;
  eps = 1.0e-32; /* small shift to avoid dividing by zero */

  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // get location slots - velocity is located on edge and c=1
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, 1,&v_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,1,&v_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   1,&v_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, 1,&v_slot[3]);CHKERRQ(ierr);

  // Loop over elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar   xx[4];

      xx[0] = _coeff[j][i][v_slot[0]];
      xx[1] = _coeff[j][i][v_slot[1]];
      xx[2] = _coeff[j][i][v_slot[2]];
      xx[3] = _coeff[j][i][v_slot[3]];

      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      /* compute dx, dy for this cell */
      cell_dt_x = dx / PetscMax(PetscMax(PetscAbsScalar(xx[0]), PetscAbsScalar(xx[1])), eps);
      cell_dt_z = dz / PetscMax(PetscMax(PetscAbsScalar(xx[2]), PetscAbsScalar(xx[3])), eps);
      cell_dt   = PetscMin(cell_dt_x,cell_dt_z);
      domain_dt = PetscMin(domain_dt,cell_dt);
    }
  }

  // MPI exchange global min/max
  ierr = MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dmcoeff));CHKERRQ(ierr);

  // Return vectors and arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmcoeff,coefflocal,&_coeff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

  // Return value
  *dt = global_dt;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
 JacobianCreate_AdvDiff - creates and preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_ADVDIFF
 
 Use: internal
 @*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianCreate_AdvDiff"
PetscErrorCode JacobianCreate_AdvDiff(FDPDE fd,Mat *J)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(fd->dmstag,J); CHKERRQ(ierr);
  ierr = JacobianPreallocator_AdvDiff(fd,*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
JacobianPreallocator_AdvDiff - preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_ADVDIFF

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianPreallocator_AdvDiff"
PetscErrorCode JacobianPreallocator_AdvDiff(FDPDE fd,Mat J)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       i, j, sx, sz, nx, nz, nEntries = 5; // local variables, extended stencil = 9  
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

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      ierr = EnergyStencil(i,j,Nx,Nz,fd->dm_btype0,fd->dm_btype1,point);CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries,point,xx,INSERT_VALUES); CHKERRQ(ierr);
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
EnergyStencil - calculates the non-zero pattern for the advdiff equation/dof for JacobianPreallocator_AdvDiff()
Use: internal
Note: 9-point extended stencil seems to cause convergence issues!
@*/
// ---------------------------------------
PetscErrorCode EnergyStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1, DMStagStencil *point)
{ 
  PetscFunctionBegin;

  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0;
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = 0;
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = 0;
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = 0;
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = 0;

  // point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = 0; // WW
  // point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = 0; // EE
  // point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = 0; // SS
  // point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = 0; // NN

  if (dm_btype0!=DM_BOUNDARY_PERIODIC) {
    if (i == 0   ) point[1] = point[0];
    if (i == Nx-1) point[2] = point[0];
    // if (i <= 1   ) point[5] = point[0];
    // if (i >= Nx-2) point[6] = point[0];
  }
  
  if (dm_btype1!=DM_BOUNDARY_PERIODIC) {
    if (j == 0   ) point[3] = point[0];
    if (j == Nz-1) point[4] = point[0];
    // if (j <= 1   ) point[7] = point[0];
    // if (j >= Nz-2) point[8] = point[0];
  }

  PetscFunctionReturn(0);
}
