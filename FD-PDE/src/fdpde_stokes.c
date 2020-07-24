/* Finite differences staggered grid context for STOKES equations */

#include "fdpde_stokes.h"

const char stokes_description[] =
"  << FD-PDE Stokes >> solves the PDEs: \n"
"    - grad(p) + div ( 2 A symgrad(u) ) - B = 0\n"
"                                div(u) - C = 0\n"
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(stokes_description,&fd->description); CHKERRQ(ierr);

  // STOKES Stencil dofs: dmstag - Vx, Vz (edges), P (element)
  fd->dof0  = 0; fd->dof1  = 1; fd->dof2  = 1; 
  fd->dofc0 = 1; fd->dofc1 = 1; fd->dofc2 = 2;

  // Evaluation functions
  fd->ops->form_function      = FormFunction_Stokes;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_Stokes;
  // fd->ops->setup              = NULL;
  fd->ops->view               = NULL;
  fd->ops->destroy            = NULL;

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
PetscErrorCode JacobianPreallocator_Stokes(FDPDE fd,Mat J)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       i, j, sx, sz, nx, nz; // local variables
  Mat            preallocator = NULL;
  // Zero entries
  PetscInt       nEntries_true, nEntries=23;
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
  
  ierr = PetscCalloc1(nEntries,&xx); CHKERRQ(ierr);
  ierr = PetscCalloc1(nEntries,&point); CHKERRQ(ierr);

  if (!fd->linearsolve) nEntries_true = 23;
  else                  nEntries_true = 11;

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {

      // Continuity equation (P) : V_x + V_z = 0
      ierr = ContinuityStencil(i,j,point); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,5,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      // X-momentum equation : (u_xx + u_zz) - p_x = rhog^x (rhog_x=0)
      ierr = XMomentumStencil(i,j,Nx,Nz,point,0); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      if (i==Nx-1){
        ierr = XMomentumStencil(i,j,Nx,Nz,point,1); CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }

      // Z-momentum equation : (u_xx + u_zz) - p_z = rhog^z
      ierr = ZMomentumStencil(i,j,Nx,Nz,point,0); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      if (j==Nz-1){
        ierr = ZMomentumStencil(i,j,Nx,Nz,point,1); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);
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
 JacobianCreate_Stokes - creates and preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_STOKES
 
 Use: internal
 @*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianCreate_Stokes"
PetscErrorCode JacobianCreate_Stokes(FDPDE fd,Mat *J)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(fd->dmstag,J); CHKERRQ(ierr);
  ierr = JacobianPreallocator_Stokes(fd,*J);CHKERRQ(ierr);
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
PetscErrorCode XMomentumStencil(PetscInt i,PetscInt j,PetscInt Nx, PetscInt Nz, DMStagStencil *point, PetscInt iloc)
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

  // non-linear stencil
  point[11].i = i-2; point[11].j = j  ; point[11].loc = DMSTAG_DOWN;    point[11].c   = 0;
  point[12].i = i-2; point[12].j = j  ; point[12].loc = DMSTAG_UP;      point[12].c   = 0;
  point[13].i = i-1; point[13].j = j-1; point[13].loc = DMSTAG_LEFT;    point[13].c   = 0;
  point[14].i = i-1; point[14].j = j-1; point[14].loc = DMSTAG_DOWN;    point[14].c   = 0;
  point[15].i = i  ; point[15].j = j-1; point[15].loc = DMSTAG_RIGHT;   point[15].c   = 0;
  point[16].i = i  ; point[16].j = j-1; point[16].loc = DMSTAG_DOWN;    point[16].c   = 0;
  point[17].i = i+1; point[17].j = j  ; point[17].loc = DMSTAG_DOWN;    point[17].c   = 0;
  point[18].i = i+1; point[18].j = j  ; point[18].loc = DMSTAG_UP;      point[18].c   = 0;
  point[19].i = i  ; point[19].j = j+1; point[19].loc = DMSTAG_UP;      point[19].c   = 0;
  point[20].i = i  ; point[20].j = j+1; point[20].loc = DMSTAG_RIGHT;   point[20].c   = 0;
  point[21].i = i-1; point[21].j = j+1; point[21].loc = DMSTAG_LEFT;    point[21].c   = 0;
  point[22].i = i-1; point[22].j = j+1; point[22].loc = DMSTAG_UP;      point[22].c   = 0;

  // left
  if (i == 0) {
    point[3] = point[0];
    point[5] = point[6];
    point[7] = point[8];
    point[9] = point[10];

    point[11] = point[0];
    point[12] = point[0];
    point[13] = point[0];
    point[14] = point[0];
    point[21] = point[0];
    point[22] = point[0];
  } 

  if (i == 1) {
    point[11] = point[0];
    point[12] = point[0];
  } 

  if (i == Nx-1) {
    point[17] = point[0];
    point[18] = point[0];
  } 

  
  if (iloc) { // last right - different stencil
    point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_RIGHT;   point[0].c   = 0;
    point[1].i  = i  ; point[1].j  = j-1; point[1].loc  = DMSTAG_RIGHT;   point[1].c   = 0;
    point[2].i  = i  ; point[2].j  = j+1; point[2].loc  = DMSTAG_RIGHT;   point[2].c   = 0;
    point[3].i  = i  ; point[3].j  = j  ; point[3].loc  = DMSTAG_LEFT;    point[3].c   = 0;
    point[4] = point[0];
    point[5].i  = i  ; point[5].j  = j  ; point[5].loc  = DMSTAG_DOWN;    point[5].c   = 0;
    point[6] = point[5];
    point[7].i  = i  ; point[7].j  = j  ; point[7].loc  = DMSTAG_UP;      point[7].c   = 0;
    point[8] = point[7];
    point[9].i  = i  ; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0;
    point[10] = point[9];

    point[11].i = i-1; point[11].j = j  ; point[11].loc = DMSTAG_DOWN;    point[11].c   = 0;
    point[12].i = i-1; point[12].j = j  ; point[12].loc = DMSTAG_UP;      point[12].c   = 0;
    point[13].i = i  ; point[13].j = j-1; point[13].loc = DMSTAG_LEFT;    point[13].c   = 0;
    point[14].i = i  ; point[14].j = j-1; point[14].loc = DMSTAG_DOWN;    point[14].c   = 0;
    point[15] = point[0];
    point[16] = point[0];
    point[17] = point[0];
    point[18] = point[0];
    point[19] = point[0];
    point[20] = point[0];
    point[21].i = i  ; point[21].j = j+1; point[21].loc = DMSTAG_LEFT;    point[21].c   = 0;
    point[22].i = i  ; point[22].j = j+1; point[22].loc = DMSTAG_UP;      point[22].c   = 0;
  }

  // down/up boundary
  if (j == 0) {
    point[1]  = point[0];
    point[13] = point[0];
    point[14] = point[0];
    point[15] = point[0];
    point[16] = point[0];
  } else if (j == Nz-1) {
    point[2]  = point[0];
    point[19] = point[0];
    point[20] = point[0];
    point[21] = point[0];
    point[22] = point[0];
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ZMomentumStencil - calculates the non-zero pattern for the Z-momentum equation/dof for JacobianPreallocator_Stokes()

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ZMomentumStencil(PetscInt i,PetscInt j,PetscInt Nx, PetscInt Nz, DMStagStencil *point, PetscInt iloc)
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

  // non-linear stencil
  point[11].i = i-1; point[11].j = j  ; point[11].loc = DMSTAG_UP;     point[11].c   = 0;
  point[12].i = i-1; point[12].j = j  ; point[12].loc = DMSTAG_LEFT;   point[12].c   = 0;
  point[13].i = i-1; point[13].j = j-1; point[13].loc = DMSTAG_LEFT;   point[13].c   = 0;
  point[14].i = i-1; point[14].j = j-1; point[14].loc = DMSTAG_DOWN;   point[14].c   = 0;
  point[15].i = i  ; point[15].j = j-2; point[15].loc = DMSTAG_LEFT;   point[15].c   = 0;
  point[16].i = i  ; point[16].j = j-2; point[16].loc = DMSTAG_RIGHT;  point[16].c   = 0;
  point[17].i = i+1; point[17].j = j-1; point[17].loc = DMSTAG_DOWN;   point[17].c   = 0;
  point[18].i = i+1; point[18].j = j-1; point[18].loc = DMSTAG_RIGHT;  point[18].c   = 0;
  point[19].i = i+1; point[19].j = j  ; point[19].loc = DMSTAG_RIGHT;  point[19].c   = 0;
  point[20].i = i+1; point[20].j = j  ; point[20].loc = DMSTAG_UP;     point[20].c   = 0;
  point[21].i = i  ; point[21].j = j+1; point[21].loc = DMSTAG_RIGHT;  point[21].c   = 0;
  point[22].i = i  ; point[22].j = j+1; point[22].loc = DMSTAG_LEFT;   point[22].c   = 0;

  if (j == 0) { // down
    point[1] = point[0]; 
    point[5] = point[7];
    point[6] = point[8];
    point[9] = point[10];

    point[13] = point[0]; 
    point[14] = point[0]; 
    point[15] = point[0]; 
    point[16] = point[0]; 
    point[17] = point[0]; 
    point[18] = point[0]; 
  } 

  if (j == 1) {
    point[15] = point[0]; 
    point[16] = point[0]; 
  } 

  if (j == Nz-1) {
    point[21] = point[0]; 
    point[22] = point[0]; 
  } 

  if (iloc) { // up - different stencil
    point[0].i = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_UP;      point[0].c  = 0;
    point[1].i = i  ; point[1].j  = j  ; point[1].loc  = DMSTAG_DOWN;    point[1].c  = 0;
    point[2] = point[0];
    point[3].i = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_UP;      point[3].c  = 0;
    point[4].i = i+1; point[4].j  = j  ; point[4].loc  = DMSTAG_UP;      point[4].c  = 0;
    point[5].i = i  ; point[5].j  = j  ; point[5].loc  = DMSTAG_LEFT;    point[5].c  = 0;
    point[6].i = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_RIGHT;   point[6].c  = 0;
    point[7] = point[5];
    point[8] = point[6];
    point[9].i = i  ; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c  = 0;
    point[10] = point[9];

    point[11] = point[0]; 
    point[12] = point[0]; 
    point[13].i = i-1; point[13].j = j  ; point[13].loc = DMSTAG_LEFT;   point[13].c   = 0;
    point[14].i = i-1; point[14].j = j  ; point[14].loc = DMSTAG_DOWN;   point[14].c   = 0;
    point[15].i = i  ; point[15].j = j-1; point[15].loc = DMSTAG_LEFT;   point[15].c   = 0;
    point[16].i = i  ; point[16].j = j-1; point[16].loc = DMSTAG_RIGHT;  point[16].c   = 0;
    point[17].i = i+1; point[17].j = j  ; point[17].loc = DMSTAG_DOWN;   point[17].c   = 0;
    point[18].i = i+1; point[18].j = j  ; point[18].loc = DMSTAG_RIGHT;  point[18].c   = 0;
    point[19] = point[0]; 
    point[20] = point[0]; 
    point[21] = point[0]; 
    point[22] = point[0]; 
  }

  // left/right boundary
  if (i == 0) {
    point[3]  = point[0]; 
    point[11] = point[0]; 
    point[12] = point[0];
    point[13] = point[0]; 
    point[14] = point[0]; 
  } else if (i == Nx-1) {
    point[4] = point[0];
    point[17] = point[0]; 
    point[18] = point[0];
    point[19] = point[0]; 
    point[20] = point[0]; 
  }
  PetscFunctionReturn(0);
}
