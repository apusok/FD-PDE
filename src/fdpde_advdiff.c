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
  PetscFunctionBegin;

  // Initialize data
  PetscCall(PetscStrallocpy(advdiff_description,&fd->description)); 

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
  PetscCall(PetscCalloc1(1,&ad));

  // vectors
  ad->xprev = NULL;
  ad->coeffprev = NULL;

  // fd-pde context data
  fd->data = ad;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  ad = fd->data;

  if (ad->xprev)     { PetscCall(VecDestroy(&ad->xprev)); }
  if (ad->coeffprev) { PetscCall(VecDestroy(&ad->coeffprev)); }

  PetscCall(PetscFree(ad));

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(PetscPrintf(fd->comm,"[ADVDIFF] FDPDEView:\n"));
  PetscCall(PetscPrintf(fd->comm,"  # Advection Scheme type: %s\n",AdvectSchemeTypeNames[(int)ad->advtype]));
  PetscCall(PetscPrintf(fd->comm,"  # Time step Scheme type: %s\n",TimeStepSchemeTypeNames[(int)ad->timesteptype]));
  PetscCall(PetscPrintf(fd->comm,"  # Theta: %g\n",ad->theta));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  ad = fd->data;

  if (xprev) {
    *xprev = ad->xprev;
    PetscCall(PetscObjectReference((PetscObject)ad->xprev));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");
  
  ad = fd->data;

  if (coeffprev) {
    *coeffprev = ad->coeffprev;
    PetscCall(PetscObjectReference((PetscObject)ad->coeffprev));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"The Advection Type should be set only for FD-PDE Type = ADVDIFF!");
  ad = fd->data;
  ad->advtype = advtype;

  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCall(VecDuplicate(fd->x,&ad->xprev));
    PetscCall(VecDuplicate(fd->coeff,&ad->coeffprev));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  if (dt) ad->dt = dt;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  if (dt) *dt = ad->dt;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;

  // Check fd-pde for type and setup
  if (fd->type != FDPDE_ADVDIFF) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONG,"This routine is only valid for FD-PDE Type = ADVDIFF!");
  if (!fd->data) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"The FD-PDE context data has not been set up. Call FDPDESetUp() first.");

  ad = fd->data;
  dmcoeff = fd->dmcoeff;

  PetscCall(DMGetLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal)); 
  PetscCall(DMStagVecGetArrayRead(dmcoeff,coefflocal,&_coeff));

  domain_dt = 1.0e32;
  eps = 1.0e-32; /* small shift to avoid dividing by zero */

  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext)); 

  // get location slots - velocity is located on edge and c=1
  PetscCall(DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, 1,&v_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,1,&v_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   1,&v_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, 1,&v_slot[3]));

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
  PetscCall(MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dmcoeff)));

  // Return vectors and arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dmcoeff,coefflocal,&_coeff));
  PetscCall(DMRestoreLocalVector(dmcoeff,&coefflocal)); 

  // Return value
  *dt = global_dt;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;
  PetscCall(DMCreateMatrix(fd->dmstag,J)); 
  PetscCall(JacobianPreallocator_AdvDiff(fd,*J));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = fd->Nx;
  Nz = fd->Nz;

  // MatPreallocate begin
  PetscCall(MatPreallocatePhaseBegin(J, &preallocator)); 
  
  // Get local domain
  PetscCall(DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
  // Zero entries
  PetscCall(PetscCalloc1(nEntries,&xx)); 
  PetscCall(PetscCalloc1(nEntries,&point)); 

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscCall(EnergyStencil(i,j,Nx,Nz,fd->dm_btype0,fd->dm_btype1,point));
      PetscCall(DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries,point,xx,INSERT_VALUES)); 
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  PetscCall(MatPreallocatePhaseEnd(J)); 

  // Matrix assembly
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY)); 
  PetscCall(MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY)); 

  PetscCall(PetscFree(xx));
  PetscCall(PetscFree(point));

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionReturn(PETSC_SUCCESS);
}
