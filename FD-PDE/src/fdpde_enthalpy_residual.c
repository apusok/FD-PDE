#include "fdpde_enthalpy.h"

// ---------------------------------------
/*@
FormFunction_Enthalpy - (ENTHALPY) Residual evaluation function

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_Enthalpy"
PetscErrorCode FormFunction_Enthalpy(SNES snes, Vec x, Vec f, void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  EnthalpyData   *en;
  DM             dm, dmcoeff;
  Vec            xlocal, coefflocal, flocal, xCFlocal, xCSlocal;
  Vec            xprevlocal, coeffprevlocal;
  PetscInt       Nx, Nz, sx, sz, nx, nz;
  PetscInt       i,j,icenter,iH,iTP,iT,iC,iCF,iCS,iPHI;
  PetscScalar    fval;
  DMStagBCList   bclist;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***ff;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!fd->ops->form_coefficient) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Form coefficient function pointer is NULL. Must call FDPDESetFunctionCoefficient() and provide a non-NULL function pointer.");
  en = fd->data;
  if ((!en->form_CF) || (!en->form_CS)) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Form CF-Phase Diagram function pointer is NULL. Must call FDPDEEnthalpySetFunctionsPhaseDiagram() and provide a non-NULL function pointer.");

  // Assign pointers and other variables
  dm    = fd->dmstag;
  dmcoeff = fd->dmcoeff;
  
  xprevlocal     = NULL;
  coeffprevlocal = NULL;

  Nx = fd->Nx;
  Nz = fd->Nz;

  // Update BC list
  bclist = fd->bclist;
  if (fd->bclist->evaluate) {
    ierr = fd->bclist->evaluate(dm,x,bclist,bclist->data);CHKERRQ(ierr);
  }

  // Update coefficients
  ierr = fd->ops->form_coefficient(fd,dm,x,dmcoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);

  // Update solid and liquid compositions
  ierr = en->form_CF(fd,dm,x,en->dmphase,en->xCF,en->user_context);CHKERRQ(ierr);
  ierr = en->form_CS(fd,dm,x,en->dmphase,en->xCS,en->user_context);CHKERRQ(ierr);

  ierr = DMGetLocalVector(en->dmphase, &xCSlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (en->dmphase, en->xCS, INSERT_VALUES, xCSlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(en->dmphase, &xCFlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (en->dmphase, en->xCF, INSERT_VALUES, xCFlocal); CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Map the previous time step vectors
  if (en->timesteptype != TS_NONE) {
    ierr = DMGetLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm, en->xprev, INSERT_VALUES, xprevlocal); CHKERRQ(ierr);

    ierr = DMGetLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dmcoeff, en->coeffprev, INSERT_VALUES, coeffprevlocal); CHKERRQ(ierr);

    // Check time step
    if (!en->dt) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A valid time step size for FD-PDE ENTHALPY was not set! Set with FDPDEEnthalpySetTimestep()");
    }
  }

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dm, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, flocal, &ff); CHKERRQ(ierr);

  // Get dof locations
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_H,  &iH  ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_TP, &iTP ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_T,  &iT  ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_C,  &iC  ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_CF, &iCF ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_CS, &iCS ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,DOF_PHI,&iPHI); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,en,i,j,-1,0.0,&fval); CHKERRQ(ierr);
      ff[j][i][iH] = fval;
      ierr = PotentialTempResidual(dm,xlocal,dmcoeff,coefflocal,i,j,&fval);CHKERRQ(ierr);
      ff[j][i][iTP] = fval;
      ierr = TemperatureResidual(dm,xlocal,dmcoeff,coefflocal,i,j,&fval);CHKERRQ(ierr);
      ff[j][i][iT] = fval;
      ierr = BulkCompositionResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,en,i,j,-1,0.0,&fval); CHKERRQ(ierr);
      ff[j][i][iC] = fval;
      ierr = PorosityResidual(dm,xlocal,dmcoeff,coefflocal,i,j,&fval);CHKERRQ(ierr);
      ff[j][i][iPHI] = fval;

      // phase diagram: CF, CS - need to adapt to multi-component
      PetscScalar    xx[2], xxCF, xxCS;
      DMStagStencil  point[2], pointC;
      point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_CF;
      point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_CS;
      ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);

      pointC.i = i; pointC.j = j; pointC.loc = DMSTAG_ELEMENT; pointC.c = 0;
      ierr = DMStagVecGetValuesStencil(en->dmphase,xCFlocal,1,&pointC,&xxCF); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(en->dmphase,xCSlocal,1,&pointC,&xxCS); CHKERRQ(ierr);
      
      ff[j][i][iCF] = xx[0]-xxCF;
      ff[j][i][iCS] = xx[1]-xxCS;
    }
  }

  // Boundary conditions - only element dofs
  ierr = DMStagBCListApply_Enthalpy(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,en,Nx,Nz,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(en->dmphase,&xCFlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(en->dmphase,&xCSlocal); CHKERRQ(ierr);

  if (en->timesteptype != TS_NONE) {
    ierr = DMRestoreLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
  }

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpyResidual - (ENTHALPY) calculates the residual for enthalpy per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscInt bc_type, PetscScalar bc_val, PetscScalar *_fval)
{
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  DMStagStencil  point;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = EnthalpySteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = EnthalpySteadyStateOperator(dm,xprevlocal,dmcoeff,coeffprevlocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval0); CHKERRQ(ierr);
    ierr = EnthalpySteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval1); CHKERRQ(ierr);

    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = DOF_H;
    ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);

    fval = xx - xxprev + en->dt*(en->theta*fval1 + (1-en->theta)*fval0);
  }
  *_fval = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpySteadyStateOperator - (ENTHALPY) calculates the steady state enthalpy residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpySteadyStateOperator(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscInt bc_type, PetscScalar bc_val,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter;
  PetscScalar    xxTP[9], xxPHI[9], xxPHIs[9], cx[15];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A1, B1, D1, C1_Left, C1_Right, C1_Down, C1_Up, u1[5], u2[5];
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  DMStagStencil  point[15];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = COEFF_A1; // A1
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = COEFF_B1; // B1
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = COEFF_D1; // D1
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_LEFT;    point[3].c = COEFF_C1; // C1_left
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_RIGHT;   point[4].c = COEFF_C1; // C1_right
  point[5].i = i; point[5].j = j; point[5].loc = DMSTAG_DOWN;    point[5].c = COEFF_C1; // C1_down
  point[6].i = i; point[6].j = j; point[6].loc = DMSTAG_UP;      point[6].c = COEFF_C1; // C1_up
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_LEFT;    point[7].c = COEFF_u1; // u1_left
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_RIGHT;   point[8].c = COEFF_u1; // u1_right
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_DOWN;    point[9].c = COEFF_u1; // u1_down
  point[10].i= i; point[10].j= j; point[10].loc= DMSTAG_UP;      point[10].c= COEFF_u1; // u1_up
  point[11].i= i; point[11].j= j; point[11].loc= DMSTAG_LEFT;    point[11].c= COEFF_u2; // u2_left
  point[12].i= i; point[12].j= j; point[12].loc= DMSTAG_RIGHT;   point[12].c= COEFF_u2; // u2_right
  point[13].i= i; point[13].j= j; point[13].loc= DMSTAG_DOWN;    point[13].c= COEFF_u2; // u2_down
  point[14].i= i; point[14].j= j; point[14].loc= DMSTAG_UP;      point[14].c= COEFF_u2; // u2_up
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,15,point,cx); CHKERRQ(ierr);

  // Assign variables
  A1 = cx[0];
  B1 = cx[1];
  D1 = cx[2];

  C1_Left  = cx[3];
  C1_Right = cx[4];
  C1_Down  = cx[5];
  C1_Up    = cx[6];

  u1[0] = 0.0;
  u1[1] = cx[7]; // u1_left
  u1[2] = cx[8]; // u1_right
  u1[3] = cx[9]; // u1_down
  u1[4] = cx[10];// u1_up

  u2[0] = 0.0;
  u2[1] = cx[11]; // u2_left
  u2[2] = cx[12]; // u2_right
  u2[3] = cx[13]; // u2_down
  u2[4] = cx[14]; // u2_up

  // Grid spacings
  if (i == Nx-1) dx[0] = coordx[i  ][icenter]-coordx[i-1][icenter];
  else           dx[0] = coordx[i+1][icenter]-coordx[i  ][icenter];

  if (i == 0) dx[1] = coordx[i+1][icenter]-coordx[i  ][icenter];
  else        dx[1] = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx[2]  = (dx[0]+dx[1])*0.5;

  if (j == Nz-1) dz[0] = coordz[j  ][icenter]-coordz[j-1][icenter];
  else           dz[0] = coordz[j+1][icenter]-coordz[j  ][icenter];

  if (j == 0) dz[1] = coordz[j+1][icenter]-coordz[j  ][icenter];
  else        dz[1] = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz[2] = (dz[0]+dz[1])*0.5;

  // Get stencil values - TP
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_TP; // i,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_TP; // i-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = DOF_TP; // i+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = DOF_TP; // i,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = DOF_TP; // i,j+1 -N
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = DOF_TP; // i-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = DOF_TP; // i+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = DOF_TP; // i,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = DOF_TP; // i,j+2 -NN

  if (i == 1) point[5] = point[2];
  if (j == 1) point[7] = point[4];
  if (i == Nx-2) point[6] = point[1];
  if (j == Nz-2) point[8] = point[3];

  if (i == 0) { point[1] = point[0]; point[5] = point[2]; }
  if (j == 0) { point[3] = point[0]; point[7] = point[4]; }

  if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }

  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxTP); CHKERRQ(ierr);

  // Get stencil values for solid porosity - PHI
  for (ii = 0; ii<9; ii++) { point[ii].c = DOF_PHI; }
  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxPHI); CHKERRQ(ierr);
  for (ii = 0; ii<9; ii++) { xxPHIs[ii] = 1.0 - xxPHI[ii]; }

  // // add Neumann BC
  // if (bc_type == 0) { xx[1] = xx[0] - bc_val*dx[1]; xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left 
  // if (bc_type == 1) { xx[2] = xx[0] + bc_val*dx[0]; xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right
  // if (bc_type == 2) { xx[3] = xx[0] - bc_val*dz[1]; xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down
  // if (bc_type == 3) { xx[4] = xx[0] + bc_val*dz[0]; xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up
  // if (bc_type == 4) { xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left+1 
  // if (bc_type == 5) { xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right-1
  // if (bc_type == 6) { xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down+1
  // if (bc_type == 7) { xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up-1

  // Calculate diff residual
  dQ2dx = C1_Right*(xxTP[2]-xxTP[0])/dx[0] - C1_Left*(xxTP[0]-xxTP[1])/dx[1];
  dQ2dz = C1_Up   *(xxTP[4]-xxTP[0])/dz[0] - C1_Down*(xxTP[0]-xxTP[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];

  // Calculate adv residual
  ierr = AdvectionResidual(u1,xxTP,  dx,dz,advtype,&adv1); CHKERRQ(ierr);
  ierr = AdvectionResidual(u2,xxPHIs,dx,dz,advtype,&adv2); CHKERRQ(ierr);

  ffi  = A1*adv1 +B1*adv2 + diff + D1;
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
PotentialTempResidual - (ENTHALPY) calculates the potential temperature residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode PotentialTempResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscInt i, PetscInt j, PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscScalar    xx[4], cx[5];
  PetscScalar    M1, N1, O1, P1, Q1;
  DMStagStencil  point[5];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = COEFF_M1; // M1
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = COEFF_N1; // N1
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = COEFF_O1; // O1
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_ELEMENT; point[3].c = COEFF_P1; // P1
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_ELEMENT; point[4].c = COEFF_Q1; // Q1
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,5,point,cx); CHKERRQ(ierr);

  // Assign variables
  M1 = cx[0];
  N1 = cx[1];
  O1 = cx[2];
  P1 = cx[1];
  Q1 = cx[2];

  // Get stencil values H, TP, T, PHI
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_H;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_TP;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = DOF_T;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_ELEMENT; point[3].c = DOF_PHI;
  ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

  ffi = M1*xx[0] + N1*xx[1] + O1*xx[2] + P1*xx[3] + Q1; 
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
TemperatureResidual - (ENTHALPY) calculates the temperature residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode TemperatureResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscInt i, PetscInt j, PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscScalar    xx[4], cx[5];
  PetscScalar    M2, N2, O2, P2, Q2;
  DMStagStencil  point[5];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = COEFF_M2; // M2
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = COEFF_N2; // N2
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = COEFF_O2; // O2
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_ELEMENT; point[3].c = COEFF_P2; // P2
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_ELEMENT; point[4].c = COEFF_Q2; // Q2
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,5,point,cx); CHKERRQ(ierr);

  // Assign variables
  M2 = cx[0];
  N2 = cx[1];
  O2 = cx[2];
  P2 = cx[1];
  Q2 = cx[2];

  // Get stencil values H, TP, T, PHI
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_H;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_TP;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = DOF_T;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_ELEMENT; point[3].c = DOF_PHI;
  ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

  ffi = M2*xx[0] + N2*xx[1] + O2*xx[2] + P2*xx[3] + Q2; 
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
BulkCompositionResidual - (ENTHALPY) calculates the residual for bulk composition per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode BulkCompositionResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscInt bc_type, PetscScalar bc_val, PetscScalar *_fval)
{
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  DMStagStencil  point;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = BulkCompositionSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = BulkCompositionSteadyStateOperator(dm,xprevlocal,dmcoeff,coeffprevlocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval0); CHKERRQ(ierr);
    ierr = BulkCompositionSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,en->advtype,bc_type,bc_val,&fval1); CHKERRQ(ierr);

    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = DOF_C;
    ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);

    fval = xx - xxprev + en->dt*(en->theta*fval1 + (1-en->theta)*fval0);
  }
  *_fval = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
BulkCompositionSteadyStateOperator - (ENTHALPY) calculates the steady state bulk composition residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode BulkCompositionSteadyStateOperator(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscInt bc_type, PetscScalar bc_val,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter;
  PetscScalar    xxCF[9], xxCS[9], xxPHI[9], xxPHIs[9], cx[15], f1[9], f2[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A2, B2, D2, C2_Left, C2_Right, C2_Down, C2_Up, u2[5], u3[5];
  PetscScalar    phi_Left, phi_Right, phi_Down, phi_Up;
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  DMStagStencil  point[15];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = COEFF_A2; // A2
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = COEFF_B2; // B2
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = COEFF_D2; // D2
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_LEFT;    point[3].c = COEFF_C2; // C2_left
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_RIGHT;   point[4].c = COEFF_C2; // C2_right
  point[5].i = i; point[5].j = j; point[5].loc = DMSTAG_DOWN;    point[5].c = COEFF_C2; // C2_down
  point[6].i = i; point[6].j = j; point[6].loc = DMSTAG_UP;      point[6].c = COEFF_C2; // C2_up
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_LEFT;    point[7].c = COEFF_u2; // u2_left
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_RIGHT;   point[8].c = COEFF_u2; // u2_right
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_DOWN;    point[9].c = COEFF_u2; // u2_down
  point[10].i= i; point[10].j= j; point[10].loc= DMSTAG_UP;      point[10].c= COEFF_u2; // u2_up
  point[11].i= i; point[11].j= j; point[11].loc= DMSTAG_LEFT;    point[11].c= COEFF_u3; // u3_left
  point[12].i= i; point[12].j= j; point[12].loc= DMSTAG_RIGHT;   point[12].c= COEFF_u3; // u3_right
  point[13].i= i; point[13].j= j; point[13].loc= DMSTAG_DOWN;    point[13].c= COEFF_u3; // u3_down
  point[14].i= i; point[14].j= j; point[14].loc= DMSTAG_UP;      point[14].c= COEFF_u3; // u3_up
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,15,point,cx); CHKERRQ(ierr);

  // Assign variables
  A2 = cx[0];
  B2 = cx[1];
  D2 = cx[2];

  C2_Left  = cx[3];
  C2_Right = cx[4];
  C2_Down  = cx[5];
  C2_Up    = cx[6];

  u2[0] = 0.0;
  u2[1] = cx[7]; // u2_left
  u2[2] = cx[8]; // u2_right
  u2[3] = cx[9]; // u2_down
  u2[4] = cx[10];// u2_up

  u3[0] = 0.0;
  u3[1] = cx[11]; // u3_left
  u3[2] = cx[12]; // u3_right
  u3[3] = cx[13]; // u3_down
  u3[4] = cx[14]; // u3_up

  // Grid spacings
  if (i == Nx-1) dx[0] = coordx[i  ][icenter]-coordx[i-1][icenter];
  else           dx[0] = coordx[i+1][icenter]-coordx[i  ][icenter];

  if (i == 0) dx[1] = coordx[i+1][icenter]-coordx[i  ][icenter];
  else        dx[1] = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx[2]  = (dx[0]+dx[1])*0.5;

  if (j == Nz-1) dz[0] = coordz[j  ][icenter]-coordz[j-1][icenter];
  else           dz[0] = coordz[j+1][icenter]-coordz[j  ][icenter];

  if (j == 0) dz[1] = coordz[j+1][icenter]-coordz[j  ][icenter];
  else        dz[1] = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz[2] = (dz[0]+dz[1])*0.5;

  // Get stencil values - CF
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_CF; // i,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_CF; // i-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = DOF_CF; // i+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = DOF_CF; // i,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = DOF_CF; // i,j+1 -N
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = DOF_CF; // i-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = DOF_CF; // i+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = DOF_CF; // i,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = DOF_CF; // i,j+2 -NN

  if (i == 1) point[5] = point[2];
  if (j == 1) point[7] = point[4];
  if (i == Nx-2) point[6] = point[1];
  if (j == Nz-2) point[8] = point[3];
  if (i == 0) { point[1] = point[0]; point[5] = point[2]; }
  if (j == 0) { point[3] = point[0]; point[7] = point[4]; }
  if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }
  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxCF); CHKERRQ(ierr);

  // Get stencil values for solid composition
  for (ii = 0; ii<9; ii++) { point[ii].c = DOF_CS; }
  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxCS); CHKERRQ(ierr);

  // Get stencil values for porosity and solid porosity
  for (ii = 0; ii<9; ii++) { point[ii].c = DOF_PHI; }
  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxPHI); CHKERRQ(ierr);
  for (ii = 0; ii<9; ii++) { xxPHIs[ii] = 1.0 - xxPHI[ii]; }

  // Calculate solid and fluid fluxes
  for (ii = 0; ii<9; ii++) { 
    f1[ii] = xxPHIs[ii]*xxCS[ii]; 
    f2[ii] = xxPHI[ii] *xxCF[ii]; 
  }

  // // add Neumann BC
  // if (bc_type == 0) { xx[1] = xx[0] - bc_val*dx[1]; xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left 
  // if (bc_type == 1) { xx[2] = xx[0] + bc_val*dx[0]; xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right
  // if (bc_type == 2) { xx[3] = xx[0] - bc_val*dz[1]; xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down
  // if (bc_type == 3) { xx[4] = xx[0] + bc_val*dz[0]; xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up
  // if (bc_type == 4) { xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left+1 
  // if (bc_type == 5) { xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right-1
  // if (bc_type == 6) { xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down+1
  // if (bc_type == 7) { xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up-1

  // calculate porosity on edges - assume constant grid spacing
  phi_Left  = (xxPHI[1]+xxPHI[0])*0.5;
  phi_Right = (xxPHI[2]+xxPHI[0])*0.5;
  phi_Down  = (xxPHI[3]+xxPHI[0])*0.5;
  phi_Up    = (xxPHI[4]+xxPHI[0])*0.5;

  // Calculate diff residual
  dQ2dx = C2_Right*phi_Right*(xxCF[2]-xxCF[0])/dx[0] - C2_Left*phi_Left*(xxCF[0]-xxCF[1])/dx[1];
  dQ2dz = C2_Up   *phi_Up   *(xxCF[4]-xxCF[0])/dz[0] - C2_Down*phi_Down*(xxCF[0]-xxCF[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];

  // Calculate adv residual
  ierr = AdvectionResidual(u2,f1,dx,dz,advtype,&adv1); CHKERRQ(ierr);
  ierr = AdvectionResidual(u3,f2,dx,dz,advtype,&adv2); CHKERRQ(ierr);

  ffi  = A2*adv1 + B2*adv2 + diff + D2;
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
PorosityResidual - (ENTHALPY) calculates the porosity residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode PorosityResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscInt i, PetscInt j, PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscScalar    xx[4];
  DMStagStencil  point[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get stencil values C, CF, CS, PHI
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = DOF_C;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = DOF_CF;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = DOF_CS;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_ELEMENT; point[3].c = DOF_PHI;
  ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

  ffi = xx[0] - xx[3]*xx[1] - (1.0-xx[3])*xx[2];
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListApply_Enthalpy - function to apply boundary conditions for ENTHALPY equations

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApply_Enthalpy"
PetscErrorCode DMStagBCListApply_Enthalpy(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *ad, PetscInt Nx, PetscInt Nz, PetscScalar ***ff)
{
  PetscScalar    xx, fval;
  PetscInt       i, j, ibc, idx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {
    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      // Add flux terms - for first and second points (needed for second order advection schemes)
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // if (i == 0) { // left
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i  ,j,0,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j][i  ][idx] = fval;
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i+1,j,4,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j][i+1][idx] = fval;
      // }
      // if (i == Nx-1) { // right
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i  ,j,1,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j][i  ][idx] = fval;
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i-1,j,5,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j][i-1][idx] = fval;
      // }
      // if (j == 0) { // down
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j  ,2,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j  ][i][idx] = fval;
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j+1,6,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j+1][i][idx] = fval;
      // }
      // if (j == Nz-1) { // up
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j  ,3,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j  ][i][idx] = fval;
      //   ierr = EnthalpyResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j-1,7,bclist[ibc].val,&fval); CHKERRQ(ierr);
      //   ff[j-1][i][idx] = fval;
      // }
    }
  }

  PetscFunctionReturn(0);
}