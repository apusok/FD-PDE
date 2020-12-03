#include "fdpde_enthalpy.h"

static PetscInt SingleDimIndex(PetscInt i, PetscInt j, PetscInt nz) { return i*nz+j; }
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
  ThermoState    *thm, *thm_prev;
  CoeffState     *cff, *cff_prev;
  DM             dm, dmcoeff, dmP;
  Vec            xlocal, coefflocal, flocal;
  Vec            Plocal, Pprevlocal, xprevlocal, coeffprevlocal;
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
  if (!en->form_enthalpy_method) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"This routine requires a valid form_enthalpy_method() funtion pointer. Call FDPDEEnthalpySetEnthalpyMethod() first.");

  // Assign pointers and other variables
  dm    = fd->dmstag;
  dmcoeff = fd->dmcoeff;
  dmP = en->dmP;

  xprevlocal     = NULL;
  coeffprevlocal = NULL;

  Nx = fd->Nx;
  Nz = fd->Nz;

  // Update BC list - PRELIM 
  bclist = fd->bclist;
  if (fd->bclist->evaluate) {
    // ierr = fd->bclist->evaluate(dm,x,bclist,bclist->data);CHKERRQ(ierr);
  }

  // Update coefficients
  ierr = fd->ops->form_coefficient(fd,dm,x,dmcoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dm, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, flocal, &ff); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmP, &Plocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmP, en->xP, INSERT_VALUES, Plocal); CHKERRQ(ierr);

  // Map the previous time step vectors
  if (en->timesteptype != TS_NONE) {
    ierr = DMGetLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm, en->xprev, INSERT_VALUES, xprevlocal); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dmcoeff, en->coeffprev, INSERT_VALUES, coeffprevlocal); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmP, &Pprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dmP, en->xPprev, INSERT_VALUES, Pprevlocal); CHKERRQ(ierr);

    // Check time step
    if (!en->dt) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A valid time step size for FD-PDE ENTHALPY was not set! Set with FDPDEEnthalpySetTimestep()");
    }
  }

  // update enthalpy and coeff cell data
  ierr = PetscCalloc1((size_t)(nx*nz)*sizeof(ThermoState),&thm);CHKERRQ(ierr); 
  ierr = PetscCalloc1((size_t)(nx*nz)*sizeof(CoeffState),&cff);CHKERRQ(ierr);
  ierr = ApplyEnthalpyMethod(fd,dm,xlocal,dmcoeff,coefflocal,dmP,Plocal,en,thm,cff); CHKERRQ(ierr);

  if (en->timesteptype != TS_NONE) {
    ierr = PetscCalloc1((size_t)(nx*nz)*sizeof(ThermoState),&thm_prev);CHKERRQ(ierr);
    ierr = PetscCalloc1((size_t)(nx*nz)*sizeof(CoeffState),&cff_prev);CHKERRQ(ierr);
    ierr = ApplyEnthalpyMethod(fd,dm,xprevlocal,dmcoeff,coeffprevlocal,dmP,Pprevlocal,en,thm_prev,cff_prev); CHKERRQ(ierr);
  }

  // Residual evaluation
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      ierr = EnthalpyResidual(dm,thm,cff,thm_prev,cff_prev,coordx,coordz,en,i,j,&fval); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      for (ii = 0; ii<en->ncomponents-1; ii++) {
        ierr = BulkCompositionResidual(dm,thm,cff,thm_prev,cff_prev,coordx,coordz,en,i,j,ii,&fval); CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii+1,&idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions - only element dofs // PRELIM
  ierr = en->form_user_bc(dm,x,ff,en->user_context);CHKERRQ(ierr);
  // ierr = DMStagBCListApply_Enthalpy(dm,xlocal,bclist->bc_e,bclist->nbc_element,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmP, &Plocal); CHKERRQ(ierr);

  ierr = PetscFree(thm);CHKERRQ(ierr);
  ierr = PetscFree(cff);CHKERRQ(ierr);

  if (en->timesteptype != TS_NONE) {
    ierr = PetscFree(thm_prev);CHKERRQ(ierr);
    ierr = PetscFree(cff_prev);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmP, &Pprevlocal); CHKERRQ(ierr);
  }

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  // ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ApplyEnthalpyMethod - apply enthalpy method during each solver iteration; it collects the coefficients and enthalpy variables for the entire domain
Use: internal
@*/
// ---------------------------------------
PetscErrorCode ApplyEnthalpyMethod(FDPDE fd, DM dm,Vec xlocal,DM dmcoeff,Vec coefflocal,DM dmP, Vec Plocal,EnthalpyData *en,ThermoState *thm,CoeffState *cff)
{
  PetscInt       ii,i,j,sx,sz,nx,nz,idx;
  PetscScalar    X,C[MAX_COMPONENTS],P,phi,H,T,TP,CS[MAX_COMPONENTS],CF[MAX_COMPONENTS];
  DMStagStencil  point;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  X = 0.0;
  for (ii = 0; ii<en->ncomponents-1; ii++) { C[ii] = 0.0; CF[ii] = 0.0; CS[ii] = 0.0;}

  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      idx = SingleDimIndex(i-sx,j-sz,nz);
      ierr = CoeffCellData(dmcoeff,coefflocal,i,j,&cff[idx]);CHKERRQ(ierr);
      ierr = SolutionCellData(dm,xlocal,i,j,&X,C);CHKERRQ(ierr);

      point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
      ierr = DMStagVecGetValuesStencil(dmP,Plocal,1,&point,&P); CHKERRQ(ierr);

      if (en->energy_variable == 0) {
        H = X;
        ierr = en->form_enthalpy_method(H,C,P,&TP,&T,&phi,CF,CS,en->ncomponents,en->user_context);CHKERRQ(ierr);
      }
      if (en->energy_variable == 1) {
        TP = X;
        ierr = en->form_enthalpy_method(TP,C,P,&H,&T,&phi,CF,CS,en->ncomponents,en->user_context);CHKERRQ(ierr);
      }
      
      thm[idx].P  = P;
      thm[idx].TP = TP;
      thm[idx].T  = T;
      thm[idx].H  = H;
      thm[idx].phi = phi;
      for (ii = 0; ii<en->ncomponents-1; ii++) {
        thm[idx].C[ii]  = C[ii];
        thm[idx].CS[ii] = CS[ii];
        thm[idx].CF[ii] = CF[ii];
      }
    }
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
CoeffCellData - get cell data for coefficients
Use: internal
@*/
// ---------------------------------------
PetscErrorCode CoeffCellData(DM dmcoeff, Vec coefflocal, PetscInt i,PetscInt j, CoeffState *_cff)
{
  CoeffState     cff;
  PetscInt       ii,dof0,dof1,dof2;
  DMStagStencil  pointE[6], pointF[20];
  PetscScalar    cE[6],cF[20];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dmcoeff,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);

  for (ii = 0; ii<dof2; ii++) { // element
    pointE[ii].i = i; pointE[ii].j = j; pointE[ii].loc = DMSTAG_ELEMENT; pointE[ii].c = ii;
  }

  for (ii = 0; ii<dof1; ii++) { // faces
    pointF[4*ii+0].i= i; pointF[4*ii+0].j= j; pointF[4*ii+0].loc= DMSTAG_LEFT;  pointF[4*ii+0].c= ii;
    pointF[4*ii+1].i= i; pointF[4*ii+1].j= j; pointF[4*ii+1].loc= DMSTAG_RIGHT; pointF[4*ii+1].c= ii;
    pointF[4*ii+2].i= i; pointF[4*ii+2].j= j; pointF[4*ii+2].loc= DMSTAG_DOWN;  pointF[4*ii+2].c= ii;
    pointF[4*ii+3].i= i; pointF[4*ii+3].j= j; pointF[4*ii+3].loc= DMSTAG_UP;    pointF[4*ii+3].c= ii;
  }
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,  dof2,pointE,cE); CHKERRQ(ierr);
  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,4*dof1,pointF,cF); CHKERRQ(ierr);

  // assign values
  cff.A1 = cE[COEFF_A1]; cff.A2 = cE[COEFF_A2];
  cff.B1 = cE[COEFF_B1]; cff.B2 = cE[COEFF_B2];
  cff.D1 = cE[COEFF_D1]; cff.D2 = cE[COEFF_D2];

  for (ii = 0; ii<4; ii++) { 
    cff.C1[ii] = cF[4*COEFF_C1+ii];
    cff.C2[ii] = cF[4*COEFF_C2+ii];
    cff.v[ii]  = cF[4*COEFF_v +ii];
    cff.vf[ii] = cF[4*COEFF_vf+ii];
    cff.vs[ii] = cF[4*COEFF_vs+ii];
  }

  *_cff = cff;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
SolutionCellData - get cell data for solution H/TP,C
Use: internal
@*/
// ---------------------------------------
PetscErrorCode SolutionCellData(DM dm, Vec xlocal, PetscInt i,PetscInt j, PetscScalar *_X, PetscScalar *C)
{
  PetscInt       ii,dof0,dof1,dof2;
  DMStagStencil  *pointE;
  PetscScalar    *xE, X;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(dof2,&xE); CHKERRQ(ierr);
  ierr = PetscCalloc1(dof2,&pointE); CHKERRQ(ierr);

  for (ii = 0; ii<dof2; ii++) { // element
    pointE[ii].i = i; pointE[ii].j = j; pointE[ii].loc = DMSTAG_ELEMENT; pointE[ii].c = ii;
  }
  ierr = DMStagVecGetValuesStencil(dm,xlocal,dof2,pointE,xE); CHKERRQ(ierr);

  // assign values
  X = xE[0];
  for (ii = 0; ii<dof2-1; ii++) {
    C[ii] = xE[ii+1];
  }

  *_X = X;

  ierr = PetscFree(xE);CHKERRQ(ierr);
  ierr = PetscFree(pointE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
EnthalpyResidual - (ENTHALPY) calculates the residual for H/TP per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyResidual(DM dm,ThermoState *thm, CoeffState *cff, ThermoState *thm_prev, CoeffState *cff_prev, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscScalar *_fval)
{
  PetscInt      idx,sx,sz,nx,nz;
  PetscScalar   xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = EnthalpySteadyStateOperator(dm,thm,cff,coordx,coordz,i,j,en->advtype,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = EnthalpySteadyStateOperator(dm,thm_prev,cff_prev,coordx,coordz,i,j,en->advtype,&fval0); CHKERRQ(ierr);
    ierr = EnthalpySteadyStateOperator(dm,thm,cff,coordx,coordz,i,j,en->advtype,&fval1); CHKERRQ(ierr);

    idx = SingleDimIndex(i-sx,j-sz,nz);
    xx     = thm[idx].H;
    xxprev = thm_prev[idx].H;

    fval = xx - xxprev + en->dt*(en->theta*fval1 + (1.0-en->theta)*fval0);
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
PetscErrorCode EnthalpySteadyStateOperator(DM dm, ThermoState *thm, CoeffState *cff, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, nx,nz,sx,sz,icenter, idx[9];
  PetscScalar    xxTP[9], xxPHIs[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A1, B1, D1, C1_Left, C1_Right, C1_Down, C1_Up, v[5], vs[5];
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  idx[0] = SingleDimIndex(i-sx,j-sz,nz);

  // Coefficients
  A1 = cff[idx[0]].A1;
  B1 = cff[idx[0]].B1;
  D1 = cff[idx[0]].D1;

  C1_Left  = cff[idx[0]].C1[0];
  C1_Right = cff[idx[0]].C1[1];
  C1_Down  = cff[idx[0]].C1[2];
  C1_Up    = cff[idx[0]].C1[3];

  v[0] = 0.0;
  v[1] = cff[idx[0]].v[0]; // v_left
  v[2] = cff[idx[0]].v[1]; // v_right
  v[3] = cff[idx[0]].v[2]; // v_down
  v[4] = cff[idx[0]].v[3]; // v_up

  vs[0] = 0.0;
  vs[1] = cff[idx[0]].vs[0]; // vs_left
  vs[2] = cff[idx[0]].vs[1]; // vs_right
  vs[3] = cff[idx[0]].vs[2]; // vs_down
  vs[4] = cff[idx[0]].vs[3]; // vs_up

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
  idx[0] = SingleDimIndex(i  -sx,j  -sz,nz); // C
  idx[1] = SingleDimIndex(i-1-sx,j  -sz,nz); // W
  idx[2] = SingleDimIndex(i+1-sx,j  -sz,nz); // E
  idx[3] = SingleDimIndex(i  -sx,j-1-sz,nz); // S
  idx[4] = SingleDimIndex(i  -sx,j+1-sz,nz); // N
  idx[5] = SingleDimIndex(i-2-sx,j  -sz,nz); // WW
  idx[6] = SingleDimIndex(i+2-sx,j  -sz,nz); // EE
  idx[7] = SingleDimIndex(i  -sx,j-2-sz,nz); // SS
  idx[8] = SingleDimIndex(i  -sx,j+2-sz,nz); // NN

  if (i == 1) idx[5] = idx[2];
  if (j == 1) idx[7] = idx[4];
  if (i == Nx-2) idx[6] = idx[1];
  if (j == Nz-2) idx[8] = idx[3];

  if (i == 0) { idx[1] = idx[0]; idx[5] = idx[2]; }
  if (j == 0) { idx[3] = idx[0]; idx[7] = idx[4]; }
  if (i == Nx-1) { idx[2] = idx[0]; idx[6] = idx[1]; }
  if (j == Nz-1) { idx[4] = idx[0]; idx[8] = idx[3]; }

  for (ii = 0; ii<9; ii++) { 
    xxTP[ii] = thm[idx[ii]].TP;
    xxPHIs[ii] = 1.0 - thm[idx[ii]].phi;
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
PetscErrorCode BulkCompositionResidual(DM dm,ThermoState *thm, CoeffState *cff, ThermoState *thm_prev, CoeffState *cff_prev, PetscScalar **coordx, PetscScalar **coordz, EnthalpyData *en, PetscInt i, PetscInt j, PetscInt ii, PetscScalar *_fval)
{
  PetscInt       idx, sx,sz,nx,nz;
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  if (en->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = BulkCompositionSteadyStateOperator(dm,thm,cff,coordx,coordz,i,j,ii,en->advtype,&fval); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = BulkCompositionSteadyStateOperator(dm,thm_prev,cff_prev,coordx,coordz,i,j,ii,en->advtype,&fval0); CHKERRQ(ierr);
    ierr = BulkCompositionSteadyStateOperator(dm,thm,cff,coordx,coordz,i,j,ii,en->advtype,&fval1); CHKERRQ(ierr);

    idx = SingleDimIndex(i-sx,j-sz,nz);
    xx     = thm[idx].C[ii];
    xxprev = thm_prev[idx].C[ii];

    fval = xx - xxprev + en->dt*(en->theta*fval1 + (1.0-en->theta)*fval0);
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
PetscErrorCode BulkCompositionSteadyStateOperator(DM dm, ThermoState *thm, CoeffState *cff, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt icomp, AdvectSchemeType advtype,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter, sx,sz,nx,nz,idx[9];
  PetscScalar    xxCF[9], xxCS[9], xxPHI[9], xxPHIs[9], f1[9], f2[9];
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
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Coefficients
  idx[0] = SingleDimIndex(i-sx,j-sz,nz);
  A2 = cff[idx[0]].A2;
  B2 = cff[idx[0]].B2;
  D2 = cff[idx[0]].D2;

  C2_Left  = cff[idx[0]].C2[0];
  C2_Right = cff[idx[0]].C2[1];
  C2_Down  = cff[idx[0]].C2[2];
  C2_Up    = cff[idx[0]].C2[3];

  vs[0] = 0.0;
  vs[1] = cff[idx[0]].vs[0];
  vs[2] = cff[idx[0]].vs[1]; 
  vs[3] = cff[idx[0]].vs[2];
  vs[4] = cff[idx[0]].vs[3];

  vf[0] = 0.0;
  vf[1] = cff[idx[0]].vf[0];
  vf[2] = cff[idx[0]].vf[1];
  vf[3] = cff[idx[0]].vf[2];
  vf[4] = cff[idx[0]].vf[3];

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

  // Get stencil values - CF, CS, phi, phis
  idx[0] = SingleDimIndex(i  -sx,j  -sz,nz); // C
  idx[1] = SingleDimIndex(i-1-sx,j  -sz,nz); // W
  idx[2] = SingleDimIndex(i+1-sx,j  -sz,nz); // E
  idx[3] = SingleDimIndex(i  -sx,j-1-sz,nz); // S
  idx[4] = SingleDimIndex(i  -sx,j+1-sz,nz); // N
  idx[5] = SingleDimIndex(i-2-sx,j  -sz,nz); // WW
  idx[6] = SingleDimIndex(i+2-sx,j  -sz,nz); // EE
  idx[7] = SingleDimIndex(i  -sx,j-2-sz,nz); // SS
  idx[8] = SingleDimIndex(i  -sx,j+2-sz,nz); // NN

  if (i == 1) idx[5] = idx[2];
  if (j == 1) idx[7] = idx[4];
  if (i == Nx-2) idx[6] = idx[1];
  if (j == Nz-2) idx[8] = idx[3];

  if (i == 0) { idx[1] = idx[0]; idx[5] = idx[2]; }
  if (j == 0) { idx[3] = idx[0]; idx[7] = idx[4]; }
  if (i == Nx-1) { idx[2] = idx[0]; idx[6] = idx[1]; }
  if (j == Nz-1) { idx[4] = idx[0]; idx[8] = idx[3]; }

  // Get local data
  for (ii = 0; ii<9; ii++) { 
    xxCF[ii] = thm[idx[ii]].CF[icomp];
    xxCS[ii] = thm[idx[ii]].CS[icomp];
    xxPHI[ii] = thm[idx[ii]].phi;
    xxPHIs[ii] = 1.0 - thm[idx[ii]].phi;
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

// // ---------------------------------------
// /*@
// DMStagBCListApply_Enthalpy - function to apply boundary conditions for ENTHALPY equations

// Use: internal
// @*/
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagBCListApply_Enthalpy"
// PetscErrorCode DMStagBCListApply_Enthalpy(DM dm, Vec xlocal,DMStagBC *bclist, PetscInt nbc, PetscScalar ***ff)
// {
//   PetscScalar    xx, fval;
//   PetscInt       i, j, ibc, idx;
//   PetscErrorCode ierr;
//   PetscFunctionBeginUser;

//   // Loop over all boundaries
//   for (ibc = 0; ibc<nbc; ibc++) {
//     if (bclist[ibc].type == BC_DIRICHLET) {
//       i   = bclist[ibc].point.i;
//       j   = bclist[ibc].point.j;
//       idx = bclist[ibc].idx;

//       // Get residual value
//       ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
//       ff[j][i][idx] = xx - bclist[ibc].val;
//     }

//     if (bclist[ibc].type == BC_NEUMANN) {
//       i   = bclist[ibc].point.i;
//       j   = bclist[ibc].point.j;
//       idx = bclist[ibc].idx;

//       if (bclist[ibc].val) {
//         SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Non-zero BC type NEUMANN for FDPDE_ENTHALPY [ELEMENT] is not yet implemented.");
//       }
//     }
//   }

//   PetscFunctionReturn(0);
// }