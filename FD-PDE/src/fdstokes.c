/* Finite differences staggered grid context for STOKES equations */

#include "fdstokes.h"


const char stokes_description[] =
"  << FD-PDE Stokes >> solves the PDEs: \n"
"    div ( 2 eta symgrad(u) ) - grad(p) = f(x) \n"
"                              - div(u) = g(x) \n"
"  [User notes] \n"
"  * The function f(x) is defined in velocity points, and g(x) is defined \n" 
"    in center points. For DMStag the following right-hand-side coefficients  \n" 
"    need to be specified: \n" 
"        f(x) = [fux, fuz], fux in Vx-points, fuz in Vz-points \n" 
"        g(x) = fp, in P-points (element) \n"
"  * The viscosity has to be specified:\n" 
"       eta = [eta_c, eta_n], eta_c in center points, eta_n in corner points.\n";

// ---------------------------------------
// FDCreate_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDCreate_Stokes"
PetscErrorCode FDCreate_Stokes(FD fd)
{
  DM             dmPV;
  PetscScalar    pval = -0.00001;
  PetscInt       dofPV0, dofPV1, dofPV2, stencilWidth;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(stokes_description,&fd->description); CHKERRQ(ierr);

  // stencil dofs
  dofPV0 = 0; dofPV1 = 1; dofPV2 = 1; // dmstag: Vx, Vz (edges), P (element)
  stencilWidth = 1;

  // Create DMStag object for Stokes unknowns: dmPV (P-element, v-vertex)
  ierr = DMStagCreate2d(fd->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, fd->Nx, fd->Nz, 
            PETSC_DECIDE, PETSC_DECIDE, dofPV0, dofPV1, dofPV2, 
            DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmPV); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmPV); CHKERRQ(ierr);
  ierr = DMSetUp         (dmPV); CHKERRQ(ierr);

  // Assign pointers
  fd->dmstag  = dmPV;

  // Create global vectors
  ierr = DMCreateGlobalVector(fd->dmstag, &fd->x    ); CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->r     ); CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->xguess); CHKERRQ(ierr);

  // Set initial values for xguess
  ierr = VecSet(fd->xguess,pval);CHKERRQ(ierr);

  // Create Jacobian
  ierr = DMCreateMatrix(fd->dmstag, &fd->J); CHKERRQ(ierr);

  // Evaluation functions
  fd->ops->form_function      = FormFunction_Stokes;
  fd->ops->jacobian_prealloc  = FDJacobianPreallocator_Stokes;
  fd->ops->create_coefficient = FDCreateCoefficient_Stokes;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDCreateCoefficient_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDCreateCoefficient_Stokes"
PetscErrorCode FDCreateCoefficient_Stokes(FD fd)
{
  DM             dmCoeff;
  PetscScalar    xmin, xmax, zmin, zmax;
  PetscInt       dofCf0, dofCf1, dofCf2;
  PetscInt       iprev, inext;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Stencil dofs
  dofCf0 = 1; dofCf1 = 1; dofCf2 = 2;

  // Create DMStag object for Stokes coefficients: dmCoeff 
  ierr = DMStagCreateCompatibleDMStag(fd->dmstag, dofCf0, dofCf1, dofCf2, 0, &dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(dmCoeff); CHKERRQ(ierr);

  // Get start and end coordinates
  ierr = DMStagGet1dCoordinateArraysDOFRead(fd->dmstag,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(fd->dmstag,DMSTAG_LEFT,&iprev);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(fd->dmstag,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  xmin = coordx[0][iprev]; xmax = coordx[fd->Nx-1][inext];
  zmin = coordz[0][iprev]; zmax = coordz[fd->Nz-1][inext];

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(fd->dmstag,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Set coordinates - should mimic the same method as dmstag
  ierr = DMStagSetUniformCoordinatesProduct(dmCoeff, xmin, xmax, zmin, zmax, 0.0, 0.0);CHKERRQ(ierr);

  // Assign pointers
  fd->dmcoeff = dmCoeff;

  // Create global vectors
  ierr = DMCreateGlobalVector(fd->dmcoeff,&fd->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDJacobianPreallocator_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDJacobianPreallocator_Stokes"
PetscErrorCode FDJacobianPreallocator_Stokes(FD fd)
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

  // NOTE: Should take into account fd->bc_list for BC
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
// ContinuityStencil
// ---------------------------------------
PetscErrorCode ContinuityStencil(PetscInt i,PetscInt j, DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = 0;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = 0;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = 0;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = 0;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// XMomentumStencil
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
// ZMomentumStencil
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