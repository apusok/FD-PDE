/* Finite differences staggered grid context for STOKES-DARCY-2FIELD equations */

#include "fdpde_stokesdarcy2field.h"
#include "fdpde_stokes.h"

const char stokesdarcy2_description[] =
"  << FD-PDE Stokes-Darcy 2 Field Formulation >> solves the PDEs: \n"
"    - grad(p) + div ( 2 A symgrad(u) ) + grad( D1 div(u)) - B = 0\n"
"                           div(u) + div( D2 grad(p) + D3) - C = 0\n"
"  Notes: \n"
"  * Unknowns: p - pressure (fluid), u - velocity (solid). \n" 
"  * The coefficients A,B,C are defined as in STOKES questions: \n" 
"        A - effective viscosity (center, corner), \n" 
"        B - right-hand side for the momentum equation (edges), \n" 
"        C - right-hand side for the continuity equation (center). \n" 
"  * The D1, D2, D3 Stokes-Darcy coupling coefficients are defined: \n"
"        D1 - bulk and shear viscosity (center), \n" 
"        D2 - coefficient for grad(p) in Darcy flux: D2 = -K/mu (edges), \n" 
"        D3 - fluid buoyancy in Darcy flux: D3 = -K/mu*rho_f*g_vec (edges). \n" ;

// ---------------------------------------
/*@
FDPDECreate_StokesDarcy2Field - creates the data structures for FDPDEType = STOKESDARCY2FIELD

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate_StokesDarcy2Field"
PetscErrorCode FDPDECreate_StokesDarcy2Field(FDPDE fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(stokesdarcy2_description,&fd->description); CHKERRQ(ierr);

  // STOKESDARCY2FIELD Stencil dofs: dmstag - Vx, Vz (edges), P (element)
  fd->dof0  = 0; fd->dof1  = 1; fd->dof2  = 1; 
  fd->dofc0 = 1; fd->dofc1 = 3; fd->dofc2 = 3; // corner (A), edges (B,D2,D3), element (A,C,D1)

  // Evaluation functions
  fd->ops->form_function      = FormFunction_StokesDarcy2Field;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_StokesDarcy2Field;
  // fd->ops->setup              = NULL;
  fd->ops->view               = NULL;
  fd->ops->destroy            = NULL;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
JacobianPreallocator_StokesDarcy2Field - preallocates the non-zero pattern into the Jacobian for FDPDEType = STOKESDARCY2FIELD

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianPreallocator_StokesDarcy2Field"
PetscErrorCode JacobianPreallocator_StokesDarcy2Field(FDPDE fd,Mat J)
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
  ierr = MatPreallocatePhaseBegin(J, &preallocator); CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Zero entries
  PetscInt      nEntries_true, nEntries=23;
  PetscScalar   xx[nEntries];
  DMStagStencil point[nEntries];
  ierr = PetscMemzero(xx,sizeof(PetscScalar)*nEntries); CHKERRQ(ierr);

  if (fd->linearsolve) nEntries_true = 23;
  else                 nEntries_true = 11;

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {

      // Continuity equation - add terms for Darcy
      ierr = ContinuityStencil_StokesDarcy2Field(i,j,Nx,Nz,point); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,9,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      // X-momentum equation - stencil remains the same as Stokes
      ierr = XMomentumStencil(i,j,Nx,Nz,point,0); CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);

      if (i==Nx-1){
        ierr = XMomentumStencil(i,j,Nx,Nz,point,1); CHKERRQ(ierr);
        ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,nEntries_true,point,xx,INSERT_VALUES); CHKERRQ(ierr);
      }

      // Z-momentum equation - stencil remains the same as Stokes
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

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
 JacobianCreate_StokesDarcy2Field - creates and preallocates the non-zero pattern into the Jacobian for FDPDEType = FDPDE_STOKESDARCY2FIELD
 
 Use: internal
 @*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "JacobianCreate_StokesDarcy2Field"
PetscErrorCode JacobianCreate_StokesDarcy2Field(FDPDE fd,Mat *J)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(fd->dmstag,J); CHKERRQ(ierr);
  ierr = JacobianPreallocator_StokesDarcy2Field(fd,*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityStencil_StokesDarcy2Field - calculates the non-zero pattern for the continuity equation

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityStencil_StokesDarcy2Field(PetscInt i,PetscInt j,PetscInt Nx,PetscInt Nz,DMStagStencil *point)
{
  PetscFunctionBegin;
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // for P Dirichlet BC
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = 0;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = 0;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = 0;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = 0;

  // Darcy extra terms
  point[5].i = i+1; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = 0;
  point[6].i = i-1; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = 0;
  point[7].i = i  ; point[7].j = j+1; point[7].loc = DMSTAG_ELEMENT; point[7].c = 0;
  point[8].i = i  ; point[8].j = j-1; point[8].loc = DMSTAG_ELEMENT; point[8].c = 0;

  // correct for boundaries
  if (i == Nx-1) point[5] = point[0];
  if (i == 0   ) point[6] = point[0];
  if (j == Nz-1) point[7] = point[0];
  if (j == 0   ) point[8] = point[0];

  PetscFunctionReturn(0);
}