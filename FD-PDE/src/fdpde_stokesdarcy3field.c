/* Finite differences staggered grid context for STOKES-DARCY-3FIELD equations */
/* THREE-FIELD FORMULATION */

#include "fdpde_stokesdarcy3field.h"
#include "fdpde_stokes.h"

const char stokesdarcy3_description[] =
"  << FD-PDE Stokes-Darcy 2 Field Formulation >> solves the PDEs: \n"
"    - grad(p) + div ( 2 A symgrad(u) ) - B = 0\n"
"      div(u) + div( D2 grad(p) + D3 + D4 grad(P)) - C = 0\n"
"      div(u) + D1*P - DC = 0\n"
"  Notes: \n"
"  * Unknowns: p - dynamic pressure, P - compaction pressure, u - velocity (solid). \n" 
"  * The coefficients A,B,C are defined as in STOKES questions: \n" 
"        A - effective viscosity (center, corner), \n" 
"        B - right-hand side for the momentum equation (edges), \n" 
"        C - right-hand side for the continuity equation (center). \n" 
"  * The D1, D2, D3 Stokes-Darcy coupling coefficients are defined: \n"
"        D1 - compaction viscosity coefficient (center), \n" 
"        D2 - coefficient for grad(p) in Darcy flux (edges), \n" 
"        D3 - fluid buoyancy in Darcy flux: D3 = -K/mu*rho_f*g_vec (edges), \n"
"        D4 - coefficient for grad(P) in Darcy flux (edges),  \n"
"        DC - right-hand side for the compaction pressure equation (center). \n" ;

// ---------------------------------------
/*@
FDPDECreate_StokesDarcy3Field - creates the data structures for FDPDEType = STOKESDARCY3FIELD

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate_StokesDarcy3Field"
PetscErrorCode FDPDECreate_StokesDarcy3Field(FDPDE fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(stokesdarcy3_description,&fd->description); CHKERRQ(ierr);

  // STOKESDARCY2FIELD Stencil dofs: dmstag - edges (v), element (p,P)
  fd->dof0  = 0; fd->dof1  = 1; fd->dof2  = 2; 
  fd->dofc0 = 1; fd->dofc1 = 4; fd->dofc2 = 4; // corner (A), edges (B,D2,D3,D4), element (A,C,D1,DC)

  // Evaluation functions
  fd->ops->form_function      = FormFunction_StokesDarcy3Field;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_StokesDarcy3Field;
  fd->ops->setup              = NULL;
  fd->ops->view               = NULL;
  fd->ops->destroy            = NULL;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
JacobianPreallocator_StokesDarcy3Field - preallocates the non-zero pattern into the Jacobian for FDPDEType = STOKESDARCY3FIELD

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianPreallocator_StokesDarcy3Field"
PetscErrorCode JacobianPreallocator_StokesDarcy3Field(FDPDE fd,Mat J)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       i, j, sx, sz, nx, nz; // local variables
  Mat            preallocator = NULL;
  PetscInt       nEntries_true;
  const PetscInt nEntries=STENCIL_STOKES_MOMENTUM_NONLIN;
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

  if (!fd->linearsolve) nEntries_true = STENCIL_STOKES_MOMENTUM_NONLIN;
  else                  nEntries_true = STENCIL_STOKES_MOMENTUM_LIN;

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {

      // Continuity equation - add terms for Darcy
      ierr = ContinuityStencil_StokesDarcy3Field(i,j,Nx,Nz,point); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,13,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      // Compaction equation
      ierr = CompactionStencil_StokesDarcy3Field(i,j,Nx,Nz,point); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,5,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      // Momentum equations - the same as for Stokes
      ierr = XMomentumStencil(i,j,Nx,Nz,point,0); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      if (i==Nx-1){
        ierr = XMomentumStencil(i,j,Nx,Nz,point,1); CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }

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
 JacobianCreate_StokesDarcy3Field - creates and preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_STOKESDARCY3FIELD
 
 Use: internal
 @*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianCreate_StokesDarcy3Field"
PetscErrorCode JacobianCreate_StokesDarcy3Field(FDPDE fd,Mat *J)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(fd->dmstag,J); CHKERRQ(ierr);
  ierr = JacobianPreallocator_StokesDarcy3Field(fd,*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityStencil_StokesDarcy3Field - calculates the non-zero pattern for the continuity equation

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityStencil_StokesDarcy3Field(PetscInt i,PetscInt j,PetscInt Nx,PetscInt Nz,DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = SD3_DOF_P; 
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = SD3_DOF_V;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = SD3_DOF_V;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = SD3_DOF_V;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = SD3_DOF_V;

  // Darcy extra terms
  point[5].i = i+1; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = SD3_DOF_P;
  point[6].i = i-1; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = SD3_DOF_P;
  point[7].i = i  ; point[7].j = j+1; point[7].loc = DMSTAG_ELEMENT; point[7].c = SD3_DOF_P;
  point[8].i = i  ; point[8].j = j-1; point[8].loc = DMSTAG_ELEMENT; point[8].c = SD3_DOF_P;

  point[9].i  = i+1; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c  = SD3_DOF_PC;
  point[10].i = i-1; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c = SD3_DOF_PC;
  point[11].i = i  ; point[11].j = j+1; point[11].loc = DMSTAG_ELEMENT; point[11].c = SD3_DOF_PC;
  point[12].i = i  ; point[12].j = j-1; point[12].loc = DMSTAG_ELEMENT; point[12].c = SD3_DOF_PC;

  // correct for boundaries
  if (i == Nx-1) { point[5] = point[0]; point[9]  = point[0]; }
  if (i == 0   ) { point[6] = point[0]; point[10] = point[0]; }
  if (j == Nz-1) { point[7] = point[0]; point[11] = point[0]; }
  if (j == 0   ) { point[8] = point[0]; point[12] = point[0]; }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
CompactionStencil_StokesDarcy3Field - calculates the non-zero pattern for the compaction pressure equation

Use: internal
@*/
// ---------------------------------------
PetscErrorCode CompactionStencil_StokesDarcy3Field(PetscInt i,PetscInt j,PetscInt Nx,PetscInt Nz,DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = SD3_DOF_PC; 
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = SD3_DOF_V;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = SD3_DOF_V;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = SD3_DOF_V;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = SD3_DOF_V;

  PetscFunctionReturn(0);
}