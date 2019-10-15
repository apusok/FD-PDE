/* Finite Differences-PDE (FD-PDE) object for [ADVDIFF] */

#include "fdpde_advdiff.h"

const char advdiff_description[] =
"  << FD-PDE ADVDIFF >> solves the PDEs: \n"
"    A(dQ/dt + div (uT)) - div(B grad Q) + C = 0 \n"
"    OR \n"
"    dQ/dt + 1/A SD(Q) = 0, where \n"
"    SD(Q) = A(div (uT)) - div(B grad Q) + C, is the steady state solution. \n"
"  Notes: \n"
"  * Unknowns: Q - can be temperature. \n" 
"  * The coefficients A,B,C,u need to be defined by the user. \n" 
"        A = rho*cp (Density * Heat capacity) - defined in center, \n" 
"        B = k (Thermal conductivity) - defined on edges, \n" 
"        C = sources of heat production/sink - defined in center, \n" 
"        u = velocity - defined on edges (can be solution from Stokes equations). \n";

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
  DM             dmstag;
  PetscInt       dof0, dof1, dof2, stencilWidth;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(advdiff_description,&fd->description); CHKERRQ(ierr);

  // stencil dofs
  dof0 = 0; dof1 = 0; dof2 = 1; // dmstag: Q (element)
  stencilWidth = 1;

  // Create DMStag object for Stokes unknowns: dmstag (P-element, v-vertex)
  ierr = DMStagCreate2d(fd->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, fd->Nx, fd->Nz, 
            PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2, 
            DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmstag); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmstag); CHKERRQ(ierr);
  ierr = DMSetUp         (dmstag); CHKERRQ(ierr);

  // Create default coordinates (user can change them before calling FDSolve)
  ierr = DMStagSetUniformCoordinatesProduct(dmstag,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0);CHKERRQ(ierr);

  // Assign pointers
  fd->dmstag  = dmstag;

  // Evaluation functions
  fd->ops->form_function      = FormFunction_AdvDiff;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_AdvDiff;
  fd->ops->create_coefficient = CreateCoefficient_AdvDiff;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
CreateCoefficient_AdvDiff - creates the coefficient data (dmcoeff, coeff) for FDPDEType = FDPDE_ADVDIFF

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CreateCoefficient_AdvDiff"
PetscErrorCode CreateCoefficient_AdvDiff(FDPDE fd)
{
  DM             dmCoeff;
  PetscInt       dofCf0, dofCf1, dofCf2;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Stencil dofs
  dofCf0 = 0; dofCf1 = 2; dofCf2 = 2;

  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"DM object for FD-PDE is NULL. The constructor for FDPDE-ADVDIFF is required to create a valid DM of type DMSTAG");

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
  PetscScalar   xx[5];
  DMStagStencil point[5];
  ierr = PetscMemzero(xx,sizeof(PetscScalar)*5); CHKERRQ(ierr);

  // Get non-zero pattern for preallocator - Loop over all local elements 
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {

      // For boundary dofs - adapt the 5-point stencil
      ierr = EnergyStencil(i,j,Nx,Nz,point);CHKERRQ(ierr);
      ierr = DMStagMatSetValuesStencil(fd->dmstag,preallocator,1,point,5,point,xx,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  ierr = MatPreallocatePhaseEnd(J); CHKERRQ(ierr);
  
  // View preallocated struct of the Jacobian
  //ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Matrix assembly
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnergyStencil - calculates the non-zero pattern for the advdiff equation/dof for JacobianPreallocator_AdvDiff()

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnergyStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMStagStencil *point)
{
  PetscFunctionBegin;

  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0;
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = 0;
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = 0;
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = 0;
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = 0;

  if (i == 0) {
    point[1] = point[0];
  } else if (i == Nx-1) {
    point[2] = point[0];
  }

  if (j == 0) {
    point[3] = point[0];
  } else if (j == Nz-1) {
    point[4] = point[0];
  }

  PetscFunctionReturn(0);
}