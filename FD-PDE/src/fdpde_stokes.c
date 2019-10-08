/* Finite differences staggered grid context for STOKES equations */

#include "fdpde_stokes.h"

const char stokes_description[] =
"  << FD-PDE Stokes >> solves the PDEs: \n"
"    div ( 2 A symgrad(u) ) - grad(p) = B \n"
"                            - div(u) = C \n"
"  Notes: \n"
"  * Unknowns: p - pressure, u - velocity. \n" 
"  * The coefficients A,B,C need to be defined by the user. \n" 
"        A - effective viscosity, \n" 
"        B - right-hand side for the momentum equation, \n" 
"        C - right-hand side for the continuity equation. \n" 
"  * A,B,C are not colocated on DMStag! Effective viscosity (A) has to be specified:\n" 
"        A = [eta_c, eta_n], with eta_c in center points, eta_n in corner points.\n"
"  * The momentum rhs (B) is defined in velocity points, and the continuity rhs (C) \n" 
"    is defined in center points. For DMStag the following right-hand-side coefficients  \n" 
"    need to be specified: \n" 
"        B = [fux, fuz], with fux in Vx-points, fuz in Vz-points (edges)\n" 
"        C = fp, in P-points (element). \n";

// ---------------------------------------
/*@
FDPDECreate_Stokes - creates the data structures for FDPDEType = STOKES

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate_Stokes"
PetscErrorCode FDPDECreate_Stokes(FDPDE fd)
{
  DM             dmstag;
  PetscScalar    pval = -0.00001;
  PetscInt       dofPV0, dofPV1, dofPV2, stencilWidth;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(stokes_description,&fd->description); CHKERRQ(ierr);

  // stencil dofs
  dofPV0 = 0; dofPV1 = 1; dofPV2 = 1; // dmstag: Vx, Vz (edges), P (element)
  stencilWidth = 1;

  // Create DMStag object for Stokes unknowns: dmstag (P-element, v-vertex)
  ierr = DMStagCreate2d(fd->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, fd->Nx, fd->Nz, 
            PETSC_DECIDE, PETSC_DECIDE, dofPV0, dofPV1, dofPV2, 
            DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmstag); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmstag); CHKERRQ(ierr);
  ierr = DMSetUp         (dmstag); CHKERRQ(ierr);

  // Create default coordinates (user can change them before calling FDSolve)
  ierr = DMStagSetUniformCoordinatesProduct(dmstag,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);

  // Assign pointers
  fd->dmstag  = dmstag;

  // Create global vectors
  ierr = DMCreateGlobalVector(fd->dmstag, &fd->x); CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->r); CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->xold); CHKERRQ(ierr);

  // Set initial values for xguess
  ierr = VecSet(fd->xold,pval);CHKERRQ(ierr);

  // Create Jacobian
  ierr = DMCreateMatrix(fd->dmstag, &fd->J); CHKERRQ(ierr);

  // Evaluation functions
  fd->ops->form_function      = FormFunction_Stokes;
  fd->ops->jacobian_prealloc  = JacobianPreallocator_Stokes;
  fd->ops->create_coefficient = CreateCoefficient_Stokes;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
CreateCoefficient_Stokes - creates the coefficient data (dmcoeff, coeff) for FDPDEType = FDPDE_STOKES

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CreateCoefficient_Stokes"
PetscErrorCode CreateCoefficient_Stokes(FDPDE fd)
{
  DM             dmCoeff;
  PetscInt       dofCf0, dofCf1, dofCf2;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Stencil dofs
  dofCf0 = 1; dofCf1 = 1; dofCf2 = 2;

  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"DM object for FD-PDE is NULL. The constructor for FDPDE-STOKES is required to create a valid DM of type DMSTAG");

  // Create DMStag object for Stokes coefficients: dmCoeff 
  ierr = DMStagCreateCompatibleDMStag(fd->dmstag, dofCf0, dofCf1, dofCf2, 0, &dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(dmCoeff); CHKERRQ(ierr);

  // Set coordinates - should mimic the same method as dmstag
  ierr = DMStagSetUniformCoordinatesProduct(dmCoeff,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);

  // Assign pointers
  fd->dmcoeff = dmCoeff;

  // Create global vector
  ierr = DMCreateGlobalVector(fd->dmcoeff,&fd->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
JacobianPreallocator_Stokes - preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_STOKES

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianPreallocator_Stokes"
PetscErrorCode JacobianPreallocator_Stokes(FDPDE fd)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       i, j, sx, sz, nx, nz; // local variables
  Mat            preallocator = NULL;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Assign pointers and other variables
  Nx = fd->Nx;
  Nz = fd->Nz;

  // MatPreallocate begin
  ierr = MatPreallocatePhaseBegin(fd->J, &preallocator); CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Zero entries
  PetscScalar   xx[11];
  DMStagStencil point[11];
  ierr = PetscMemzero(xx,sizeof(PetscScalar)*11); CHKERRQ(ierr);

  // NOTE: Should take into account fd->bclist for BC
  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {

      // Top boundary velocity Dirichlet
      if (j == Nz-1) {
        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_UP; point[0].c = 0;
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,1,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      // Bottom boundary velocity Dirichlet
      if (j == 0) {
        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_DOWN; point[0].c = 0;
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,1,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      } 

      // Right Boundary velocity Dirichlet
      if (i == Nx-1) {
        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_RIGHT; point[0].c = 0;
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,1,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      // Left velocity Dirichlet
      if (i == 0) {
        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT; point[0].c = 0;
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,1,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      } 

      // Continuity equation (P) : V_x + V_z = 0
      ierr = ContinuityStencil(i,j,point); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,5,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      // X-momentum equation : (u_xx + u_zz) - p_x = rhog^x (rhog_x=0)
      if (i > 0) {
        ierr = XMomentumStencil(i,j,Nz,point); CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,11,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }

      // Z-momentum equation : (u_xx + u_zz) - p_z = rhog^z
      if (j > 0) {
        ierr = ZMomentumStencil(i,j,Nx,point); CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,11,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  ierr = MatPreallocatePhaseEnd(fd->J); CHKERRQ(ierr);
  
  // View preallocated struct of the Jacobian
  //ierr = MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Matrix assembly
  ierr = MatAssemblyBegin(fd->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (fd->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityStencil - calculates the non-zero pattern for the continuity equation/dof for JacobianPreallocator_Stokes()

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityStencil(PetscInt i,PetscInt j, DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // for P Dirichlet BC
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = 0;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = 0;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = 0;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = 0;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
XMomentumStencil - calculates the non-zero pattern for the X-momentum equation/dof for JacobianPreallocator_Stokes()

Use: internal
@*/
// ---------------------------------------
PetscErrorCode XMomentumStencil(PetscInt i,PetscInt j,PetscInt N, DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_LEFT;    point[0].c   = 0;
  point[1].i  = i  ; point[1].j  = j-1; point[1].loc  = DMSTAG_LEFT;    point[1].c   = 0;
  point[2].i  = i  ; point[2].j  = j+1; point[2].loc  = DMSTAG_LEFT;    point[2].c   = 0;
  point[3].i  = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_LEFT;    point[3].c   = 0;
  point[4].i  = i  ; point[4].j  = j  ; point[4].loc  = DMSTAG_RIGHT;   point[4].c   = 0;
  point[5].i  = i-1; point[5].j  = j  ; point[5].loc  = DMSTAG_DOWN;    point[5].c   = 0;
  point[6].i  = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_DOWN;    point[6].c   = 0;
  point[7].i  = i-1; point[7].j  = j  ; point[7].loc  = DMSTAG_UP;      point[7].c   = 0;
  point[8].i  = i  ; point[8].j  = j  ; point[8].loc  = DMSTAG_UP;      point[8].c   = 0;
  point[9].i  = i-1; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0;
  point[10].i = i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c  = 0;

  if (j == 0) {
    point[1] = point[0];
  } else if (j == N-1) {
    point[2] = point[0];
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ZMomentumStencil - calculates the non-zero pattern for the Z-momentum equation/dof for JacobianPreallocator_Stokes()

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ZMomentumStencil(PetscInt i,PetscInt j,PetscInt N, DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_DOWN;    point[0].c  = 0;
  point[1].i = i  ; point[1].j  = j-1; point[1].loc  = DMSTAG_DOWN;    point[1].c  = 0;
  point[2].i = i  ; point[2].j  = j+1; point[2].loc  = DMSTAG_DOWN;    point[2].c  = 0;
  point[3].i = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_DOWN;    point[3].c  = 0;
  point[4].i = i+1; point[4].j  = j  ; point[4].loc  = DMSTAG_DOWN;    point[4].c  = 0;
  point[5].i = i  ; point[5].j  = j-1; point[5].loc  = DMSTAG_LEFT;    point[5].c  = 0;
  point[6].i = i  ; point[6].j  = j-1; point[6].loc  = DMSTAG_RIGHT;   point[6].c  = 0;
  point[7].i = i  ; point[7].j  = j  ; point[7].loc  = DMSTAG_LEFT;    point[7].c  = 0;
  point[8].i = i  ; point[8].j  = j  ; point[8].loc  = DMSTAG_RIGHT;   point[8].c  = 0;
  point[9].i = i  ; point[9].j  = j-1; point[9].loc  = DMSTAG_ELEMENT; point[9].c  = 0;
  point[10].i= i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c = 0;

  if (i == 0) {
    point[3] = point[0]; 
  } else if (i == N-1) {
    point[4] = point[0];
  }
  PetscFunctionReturn(0);
}
