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
  Vec            xlocal, coefflocal, flocal, xphiTlocal,xCFlocal,xCSlocal;
  Vec            xprevlocal, coeffprevlocal;
  PetscInt       Nx, Nz, sx, sz, nx, nz;
  PetscInt       i,j,ii,icenter,idx;
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
  ierr = en->form_CF(fd,dm,x,dmcoeff,fd->coeff,en->dmcomp,en->xCF,en->user_context);CHKERRQ(ierr);
  ierr = en->form_CS(fd,dm,x,dmcoeff,fd->coeff,en->dmcomp,en->xCS,en->user_context);CHKERRQ(ierr);

  ierr = DMGetLocalVector(en->dmcomp, &xCSlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (en->dmcomp, en->xCS, INSERT_VALUES, xCSlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(en->dmcomp, &xCFlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (en->dmcomp, en->xCF, INSERT_VALUES, xCFlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(en->dmphiT, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (en->dmphiT, en->xphiT, INSERT_VALUES, xphiTlocal); CHKERRQ(ierr);

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

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      if (en->energy_variable == 0) {
        ierr = EnthalpyResidual_H(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,en->dmphiT,xphiTlocal,coordx,coordz,en,i,j,&fval); CHKERRQ(ierr);
      } 
      if (en->energy_variable == 1) {
        ierr = EnthalpyResidual_TP(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,en->dmphiT,xphiTlocal,coordx,coordz,en,i,j,&fval); CHKERRQ(ierr);
      }
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      for (ii = 0; ii<en->ncomponents-1; ii++) {
        ierr = BulkCompositionResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,en->dmphiT,xphiTlocal,en->dmcomp,xCFlocal,xCSlocal,coordx,coordz,en,i,j,ii,&fval); CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii+1,&idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions - only element dofs
  ierr = DMStagBCListApply_Enthalpy(dm,xlocal,bclist->bc_e,bclist->nbc_element,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(en->dmcomp,&xCFlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(en->dmcomp,&xCSlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(en->dmphiT,&xphiTlocal); CHKERRQ(ierr);

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
EnthalpyResidual_H - (ENTHALPY) calculates the residual for enthalpy per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyResidual_H(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal,DM dmphiT,Vec xphiTlocal, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscScalar *_fval)
{
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  DMStagStencil  point;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = EnthalpySteadyStateOperator_H(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = EnthalpySteadyStateOperator_H(dm,xprevlocal,dmcoeff,coeffprevlocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval0); CHKERRQ(ierr);
    ierr = EnthalpySteadyStateOperator_H(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval1); CHKERRQ(ierr);

    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
    ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);

    fval = xx - xxprev + en->dt*(en->theta*fval1 + (1-en->theta)*fval0);
  }
  *_fval = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpySteadyStateOperator_H - (ENTHALPY) calculates the steady state enthalpy residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpySteadyStateOperator_H(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal,DM dmphiT,Vec xphiTlocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter;
  PetscScalar    cx[25], xx[9], xxTP[9], xxPHI[9], xxPHIs[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A1, B1, D1, C1_Left, C1_Right, C1_Down, C1_Up, v[5], vs[5];
  PetscScalar    M1, N1, O1, P1, Q1, M2, N2, O2, P2, Q2, M, N, P,Q;
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  DMStagStencil  point[25];
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
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_LEFT;    point[7].c = COEFF_v; // v_left
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_RIGHT;   point[8].c = COEFF_v; // v_right
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_DOWN;    point[9].c = COEFF_v; // v_down
  point[10].i= i; point[10].j= j; point[10].loc= DMSTAG_UP;      point[10].c= COEFF_v; // v_up
  point[11].i= i; point[11].j= j; point[11].loc= DMSTAG_LEFT;    point[11].c= COEFF_vs; // vs_left
  point[12].i= i; point[12].j= j; point[12].loc= DMSTAG_RIGHT;   point[12].c= COEFF_vs; // vs_right
  point[13].i= i; point[13].j= j; point[13].loc= DMSTAG_DOWN;    point[13].c= COEFF_vs; // vs_down
  point[14].i= i; point[14].j= j; point[14].loc= DMSTAG_UP;      point[14].c= COEFF_vs; // vs_up

  point[15].i = i; point[15].j = j; point[15].loc = DMSTAG_ELEMENT; point[15].c = COEFF_M1;
  point[16].i = i; point[16].j = j; point[16].loc = DMSTAG_ELEMENT; point[16].c = COEFF_N1;
  point[17].i = i; point[17].j = j; point[17].loc = DMSTAG_ELEMENT; point[17].c = COEFF_O1;
  point[18].i = i; point[18].j = j; point[18].loc = DMSTAG_ELEMENT; point[18].c = COEFF_P1;
  point[19].i = i; point[19].j = j; point[19].loc = DMSTAG_ELEMENT; point[19].c = COEFF_Q1;
  point[20].i = i; point[20].j = j; point[20].loc = DMSTAG_ELEMENT; point[20].c = COEFF_M2;
  point[21].i = i; point[21].j = j; point[21].loc = DMSTAG_ELEMENT; point[21].c = COEFF_N2;
  point[22].i = i; point[22].j = j; point[22].loc = DMSTAG_ELEMENT; point[22].c = COEFF_O2;
  point[23].i = i; point[23].j = j; point[23].loc = DMSTAG_ELEMENT; point[23].c = COEFF_P2;
  point[24].i = i; point[24].j = j; point[24].loc = DMSTAG_ELEMENT; point[24].c = COEFF_Q2;
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,25,point,cx); CHKERRQ(ierr);

  // Assign variables
  A1 = cx[0];
  B1 = cx[1];
  D1 = cx[2];

  C1_Left  = cx[3];
  C1_Right = cx[4];
  C1_Down  = cx[5];
  C1_Up    = cx[6];

  v[0] = 0.0;
  v[1] = cx[7]; // v_left
  v[2] = cx[8]; // v_right
  v[3] = cx[9]; // v_down
  v[4] = cx[10];// v_up

  vs[0] = 0.0;
  vs[1] = cx[11]; // vs_left
  vs[2] = cx[12]; // vs_right
  vs[3] = cx[13]; // vs_down
  vs[4] = cx[14]; // vs_up

  M1 = cx[15];  M2 = cx[20];
  N1 = cx[16];  N2 = cx[21];
  O1 = cx[17];  O2 = cx[22];
  P1 = cx[18];  P2 = cx[23];
  Q1 = cx[19];  Q2 = cx[24];

  M = M1 - O1/O2*M2;
  N = N1 - O1/O2*N2;
  P = P1 - O1/O2*P2;
  Q = Q1 - O1/O2*Q2;

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

  // Get stencil values - H, phi
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // i,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = 0; // i-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = 0; // i+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = 0; // i,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = 0; // i,j+1 -N
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = 0; // i-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = 0; // i+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = 0; // i,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = 0; // i,j+2 -NN

  if (i == 1) point[5] = point[2];
  if (j == 1) point[7] = point[4];
  if (i == Nx-2) point[6] = point[1];
  if (j == Nz-2) point[8] = point[3];

  if (i == 0) { point[1] = point[0]; point[5] = point[2]; }
  if (j == 0) { point[3] = point[0]; point[7] = point[4]; }

  if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }

  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xx); CHKERRQ(ierr);
  ierr = DMStagVecGetValuesStencil(dm,xphiTlocal,9,point,xxPHI); CHKERRQ(ierr);

  // calculate TP, solid porosity
  for (ii = 0; ii<9; ii++) { 
    xxPHIs[ii] = 1.0 - xxPHI[ii]; 
    xxTP[ii] = -1.0/N * (M*xx[ii] + P*xxPHI[ii] + Q);
  }

  // Calculate diff residual
  dQ2dx = C1_Right*(xxTP[2]-xxTP[0])/dx[0] - C1_Left*(xxTP[0]-xxTP[1])/dx[1];
  dQ2dz = C1_Up   *(xxTP[4]-xxTP[0])/dz[0] - C1_Down*(xxTP[0]-xxTP[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];

  // Calculate adv residual
  ierr = AdvectionResidual(v, xxTP,  dx,dz,advtype,&adv1); CHKERRQ(ierr);
  ierr = AdvectionResidual(vs,xxPHIs,dx,dz,advtype,&adv2); CHKERRQ(ierr);

  ffi  = A1*adv1 +B1*adv2 + diff + D1;
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpyResidual_TP - (ENTHALPY) calculates the residual for TP per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyResidual_TP(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal,DM dmphiT,Vec xphiTlocal, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscScalar *_fval)
{
  PetscScalar    xx, xxprev, xxphi, xxH, xxHprev, cx[10];
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  DMStagStencil  point, pointC[10];
  PetscScalar    M1, N1, O1, P1, Q1, M2, N2, O2, P2, Q2, M, N,P,Q;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = EnthalpySteadyStateOperator_TP(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = EnthalpySteadyStateOperator_TP(dm,xprevlocal,dmcoeff,coeffprevlocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval0); CHKERRQ(ierr);
    ierr = EnthalpySteadyStateOperator_TP(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,coordx,coordz,i,j,en->advtype,&fval1); CHKERRQ(ierr);

    // get enthalpy
    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
    ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xphiTlocal,1,&point,&xxphi); CHKERRQ(ierr);

    // transform to TP
    pointC[0].i = i; pointC[0].j = j; pointC[0].loc = DMSTAG_ELEMENT; pointC[0].c = COEFF_M1;
    pointC[1].i = i; pointC[1].j = j; pointC[1].loc = DMSTAG_ELEMENT; pointC[1].c = COEFF_N1;
    pointC[2].i = i; pointC[2].j = j; pointC[2].loc = DMSTAG_ELEMENT; pointC[2].c = COEFF_O1;
    pointC[3].i = i; pointC[3].j = j; pointC[3].loc = DMSTAG_ELEMENT; pointC[3].c = COEFF_P1;
    pointC[4].i = i; pointC[4].j = j; pointC[4].loc = DMSTAG_ELEMENT; pointC[4].c = COEFF_Q1;
    pointC[5].i = i; pointC[5].j = j; pointC[5].loc = DMSTAG_ELEMENT; pointC[5].c = COEFF_M2;
    pointC[6].i = i; pointC[6].j = j; pointC[6].loc = DMSTAG_ELEMENT; pointC[6].c = COEFF_N2;
    pointC[7].i = i; pointC[7].j = j; pointC[7].loc = DMSTAG_ELEMENT; pointC[7].c = COEFF_O2;
    pointC[8].i = i; pointC[8].j = j; pointC[8].loc = DMSTAG_ELEMENT; pointC[8].c = COEFF_P2;
    pointC[9].i = i; pointC[9].j = j; pointC[9].loc = DMSTAG_ELEMENT; pointC[9].c = COEFF_Q2;
    ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,10,pointC,cx); CHKERRQ(ierr);

    M1 = cx[0];  M2 = cx[5];
    N1 = cx[1];  N2 = cx[6];
    O1 = cx[2];  O2 = cx[7];
    P1 = cx[3];  P2 = cx[8];
    Q1 = cx[4];  Q2 = cx[9];

    M = M1 - O1/O2*M2;
    N = N1 - O1/O2*N2;
    P = P1 - O1/O2*P2;
    Q = Q1 - O1/O2*Q2;

    xxH     = -1.0/M*(N*xx     + P*xxphi + Q);
    xxHprev = -1.0/M*(N*xxprev + P*xxphi + Q);

    fval = xxH - xxHprev + en->dt*(en->theta*fval1 + (1-en->theta)*fval0);
  }
  *_fval = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpySteadyStateOperator_TP - (ENTHALPY) calculates the steady state enthalpy residual per dof for TP

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpySteadyStateOperator_TP(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal,DM dmphiT,Vec xphiTlocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter;
  PetscScalar    cx[15], xxTP[9], xxPHI[9], xxPHIs[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A1, B1, D1, C1_Left, C1_Right, C1_Down, C1_Up, v[5], vs[5];
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
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_LEFT;    point[7].c = COEFF_v; // v_left
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_RIGHT;   point[8].c = COEFF_v; // v_right
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_DOWN;    point[9].c = COEFF_v; // v_down
  point[10].i= i; point[10].j= j; point[10].loc= DMSTAG_UP;      point[10].c= COEFF_v; // v_up
  point[11].i= i; point[11].j= j; point[11].loc= DMSTAG_LEFT;    point[11].c= COEFF_vs; // vs_left
  point[12].i= i; point[12].j= j; point[12].loc= DMSTAG_RIGHT;   point[12].c= COEFF_vs; // vs_right
  point[13].i= i; point[13].j= j; point[13].loc= DMSTAG_DOWN;    point[13].c= COEFF_vs; // vs_down
  point[14].i= i; point[14].j= j; point[14].loc= DMSTAG_UP;      point[14].c= COEFF_vs; // vs_up
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,15,point,cx); CHKERRQ(ierr);

  // Assign variables
  A1 = cx[0];
  B1 = cx[1];
  D1 = cx[2];

  C1_Left  = cx[3];
  C1_Right = cx[4];
  C1_Down  = cx[5];
  C1_Up    = cx[6];

  v[0] = 0.0;
  v[1] = cx[7]; // v_left
  v[2] = cx[8]; // v_right
  v[3] = cx[9]; // v_down
  v[4] = cx[10];// v_up

  vs[0] = 0.0;
  vs[1] = cx[11]; // vs_left
  vs[2] = cx[12]; // vs_right
  vs[3] = cx[13]; // vs_down
  vs[4] = cx[14]; // vs_up

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

  // Get stencil values - TP, phi
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // i,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = 0; // i-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = 0; // i+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = 0; // i,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = 0; // i,j+1 -N
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = 0; // i-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = 0; // i+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = 0; // i,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = 0; // i,j+2 -NN

  if (i == 1) point[5] = point[2];
  if (j == 1) point[7] = point[4];
  if (i == Nx-2) point[6] = point[1];
  if (j == Nz-2) point[8] = point[3];

  if (i == 0) { point[1] = point[0]; point[5] = point[2]; }
  if (j == 0) { point[3] = point[0]; point[7] = point[4]; }

  if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }

  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xxTP); CHKERRQ(ierr);
  ierr = DMStagVecGetValuesStencil(dm,xphiTlocal,9,point,xxPHI); CHKERRQ(ierr);

  // calculate solid porosity
  for (ii = 0; ii<9; ii++) { 
    xxPHIs[ii] = 1.0 - xxPHI[ii];
  }

  // Calculate diff residual
  dQ2dx = C1_Right*(xxTP[2]-xxTP[0])/dx[0] - C1_Left*(xxTP[0]-xxTP[1])/dx[1];
  dQ2dz = C1_Up   *(xxTP[4]-xxTP[0])/dz[0] - C1_Down*(xxTP[0]-xxTP[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];

  // Calculate adv residual
  ierr = AdvectionResidual(v, xxTP,  dx,dz,advtype,&adv1); CHKERRQ(ierr);
  ierr = AdvectionResidual(vs,xxPHIs,dx,dz,advtype,&adv2); CHKERRQ(ierr);

  ffi  = A1*adv1 +B1*adv2 + diff + D1;
  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
BulkCompositionResidual - (ENTHALPY) calculates the residual for bulk composition per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode BulkCompositionResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, DM dmphiT,Vec xphiTlocal,DM dmcomp,Vec xCFlocal,Vec xCSlocal, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscInt ii, PetscScalar *_fval)
{
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  DMStagStencil  point;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = BulkCompositionSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,dmcomp,xCFlocal,xCSlocal,coordx,coordz,i,j,ii,en->advtype,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = BulkCompositionSteadyStateOperator(dm,xprevlocal,dmcoeff,coeffprevlocal,dmphiT,xphiTlocal,dmcomp,xCFlocal,xCSlocal,coordx,coordz,i,j,ii,en->advtype,&fval0); CHKERRQ(ierr);
    ierr = BulkCompositionSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,dmphiT,xphiTlocal,dmcomp,xCFlocal,xCSlocal,coordx,coordz,i,j,ii,en->advtype,&fval1); CHKERRQ(ierr);

    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = ii+1;
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
PetscErrorCode BulkCompositionSteadyStateOperator(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, DM dmphiT,Vec xphiTlocal, DM dmcomp,Vec xCFlocal,Vec xCSlocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt icomp, AdvectSchemeType advtype,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter;
  PetscScalar    xxCF[9], xxCS[9], xxPHI[9], xxPHIs[9], cx[15], f1[9], f2[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A2, B2, D2, C2_Left, C2_Right, C2_Down, C2_Up, vs[5], vf[5];
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
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_LEFT;    point[7].c = COEFF_vs; // vs_left
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_RIGHT;   point[8].c = COEFF_vs; // vs_right
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_DOWN;    point[9].c = COEFF_vs; // vs_down
  point[10].i= i; point[10].j= j; point[10].loc= DMSTAG_UP;      point[10].c= COEFF_vs; // vs_up
  point[11].i= i; point[11].j= j; point[11].loc= DMSTAG_LEFT;    point[11].c= COEFF_vf; // vf_left
  point[12].i= i; point[12].j= j; point[12].loc= DMSTAG_RIGHT;   point[12].c= COEFF_vf; // vf_right
  point[13].i= i; point[13].j= j; point[13].loc= DMSTAG_DOWN;    point[13].c= COEFF_vf; // vf_down
  point[14].i= i; point[14].j= j; point[14].loc= DMSTAG_UP;      point[14].c= COEFF_vf; // vf_up
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,15,point,cx); CHKERRQ(ierr);

  // Assign variables
  A2 = cx[0];
  B2 = cx[1];
  D2 = cx[2];

  C2_Left  = cx[3];
  C2_Right = cx[4];
  C2_Down  = cx[5];
  C2_Up    = cx[6];

  vs[0] = 0.0;
  vs[1] = cx[7]; // vs_left
  vs[2] = cx[8]; // vs_right
  vs[3] = cx[9]; // vs_down
  vs[4] = cx[10];// vs_up

  vf[0] = 0.0;
  vf[1] = cx[11]; // vf_left
  vf[2] = cx[12]; // vf_right
  vf[3] = cx[13]; // vf_down
  vf[4] = cx[14]; // vf_up

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
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = icomp; // i,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = icomp; // i-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = icomp; // i+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = icomp; // i,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = icomp; // i,j+1 -N
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = icomp; // i-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = icomp; // i+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = icomp; // i,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = icomp; // i,j+2 -NN

  if (i == 1) point[5] = point[2];
  if (j == 1) point[7] = point[4];
  if (i == Nx-2) point[6] = point[1];
  if (j == Nz-2) point[8] = point[3];
  if (i == 0) { point[1] = point[0]; point[5] = point[2]; }
  if (j == 0) { point[3] = point[0]; point[7] = point[4]; }
  if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }
  ierr = DMStagVecGetValuesStencil(dmcomp,xCFlocal,9,point,xxCF); CHKERRQ(ierr);
  ierr = DMStagVecGetValuesStencil(dmcomp,xCSlocal,9,point,xxCS); CHKERRQ(ierr);

  // Get stencil values for porosity and solid porosity
  for (ii = 0; ii<9; ii++) { point[ii].c = 0; }
  ierr = DMStagVecGetValuesStencil(dmphiT,xphiTlocal,9,point,xxPHI); CHKERRQ(ierr);

  // Calculate solid and fluid fluxes
  for (ii = 0; ii<9; ii++) { 
    xxPHIs[ii] = 1.0 - xxPHI[ii];
    f1[ii] = xxPHIs[ii]*xxCS[ii]; 
    f2[ii] = xxPHI[ii] *xxCF[ii]; 
  }

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
  ierr = AdvectionResidual(vs,f1,dx,dz,advtype,&adv1); CHKERRQ(ierr);
  ierr = AdvectionResidual(vf,f2,dx,dz,advtype,&adv2); CHKERRQ(ierr);

  ffi  = A2*adv1 + B2*adv2 + diff + D2;
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
PetscErrorCode DMStagBCListApply_Enthalpy(DM dm, Vec xlocal,DMStagBC *bclist, PetscInt nbc, PetscScalar ***ff)
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
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if (bclist[ibc].val) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Non-zero BC type NEUMANN for FDPDE_ENTHALPY [ELEMENT] is not yet implemented.");
      }
    }
  }

  PetscFunctionReturn(0);
}