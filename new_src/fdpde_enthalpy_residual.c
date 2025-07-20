#include "fdpde_enthalpy.h"

static char * EnthalpyErrorTypeNames(int err) {
  switch (err) {
    case STATE_VALID:
    return "STATE_VALID";
    break;
  case PHI_STATE_INVALID:
    return "PHI_STATE_INVALID";
    break;
  case ERR_PHI_DIVIDE_BY_ZERO:
    return "ERR_PHI_DIVIDE_BY_ZERO";
    break;
  case ERR_SOLID_PHI_DIVIDE_BY_ZERO:
    return "ERR_SOLID_PHI_DIVIDE_BY_ZERO";
    break;
  case ERR_DIVIDE_BY_ZERO:
    return "ERR_DIVIDE_BY_ZERO";
    break;
  case ERR_INF_NAN_VALUE:
    return "ERR_INF_NAN_VALUE";
    break;
  case DIM_T_KELVIN_STATE_INVALID:
    return "DIM_T_KELVIN_STATE_INVALID";
    break;
  case DIM_T_CELSIUS_STATE_INVALID:
    return "DIM_T_CELSIUS_STATE_INVALID";
    break;
  case DIM_STATE_INVALID:
    return "DIM_STATE_INVALID";
    break;
  case DIM_C_STATE_INVALID:
    return "DIM_C_STATE_INVALID";
    break;
  case DIM_CF_STATE_INVALID:
    return "DIM_CF_STATE_INVALID";
    break;
  case DIM_CS_STATE_INVALID:
    return "DIM_CS_STATE_INVALID";
    break;
  case STATE_INVALID_IERR:
    return "STATE_INVALID_IERR";
    break;
  case STATE_INVALID:
    return "STATE_INVALID";
    break;
  default:
    return "UNKNOWN_INVALID_STATE";
    break;
  }
};

#define SingleDimIndex(i,j,nz) (i)*(nz)+(j)

typedef struct {
  PetscInt     i,j,icomp;
  const ThermoState  *thm,*thm_prev;
  const CoeffState   *cff,*cff_prev;
  const EnthalpyData *en;
  DM                 dm;
  DMBoundaryType     dm_btype0,dm_btype1;
  const PetscScalar  **coordx,**coordz;
  PetscInt           icenter;
  PetscInt           sx,sz,nx,nz;
  PetscInt           Nx,Nz;
} EnthalpyPackCtx;

PetscErrorCode EnthalpyPackCtx_InitDM(DM dm, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1, PetscScalar **cx,PetscScalar **cz,EnthalpyPackCtx *ctx)
{
  PetscFunctionBegin;
  ctx->dm = dm;
  ctx->dm_btype0 = dm_btype0;
  ctx->dm_btype1 = dm_btype1;
  ctx->coordx = (const PetscScalar**)cx;
  ctx->coordz = (const PetscScalar**)cz;
  PetscCall(DMStagGetCorners(dm, &ctx->sx, &ctx->sz, NULL, &ctx->nx, &ctx->nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dm,&ctx->Nx,&ctx->Nz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&ctx->icenter));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnthalpyPackCtx_InitState(ThermoState *thm,ThermoState *thm_k,
                                            CoeffState *cff,CoeffState *cff_k,
                                            EnthalpyData *en,
                                            EnthalpyPackCtx *ctx)
{
  PetscFunctionBegin;
  ctx->thm      = (const ThermoState*)thm;
  ctx->thm_prev = (const ThermoState*)thm_k;
  ctx->cff      = (const CoeffState*)cff;
  ctx->cff_prev = (const CoeffState*)cff_k;
  ctx->en       = (const EnthalpyData*)en;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnthalpyPackCtx_Init_IJ(PetscInt i,PetscInt j,PetscInt icomp,EnthalpyPackCtx *ctx)
{
  PetscFunctionBegin;
  ctx->i = i;
  ctx->j = j;
  ctx->icomp = icomp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnthalpyResidual_pack(EnthalpyPackCtx*,PetscScalar*);
PetscErrorCode BulkCompositionResidual_pack(EnthalpyPackCtx*,PetscScalar*);
PetscErrorCode DMStagBCListApply_Enthalpy(DM,Vec,DMStagBC*,PetscInt,PetscScalar***,EnthalpyPackCtx);

PetscBool ghost_point_enthalpy(PetscBool flgx, PetscBool flgz, PetscInt i, PetscInt j, PetscInt Nx, PetscInt Nz) {
  if ((flgx  && flgz) || (!flgx && flgz && (i>=0) && (i<Nx)) || (flgx && !flgz && (j>=0) && (j<Nz)) || 
     (!flgx && !flgz && (i>=0) && (i<Nx) && (j>=0) && (j<Nz))) {
    return PETSC_TRUE;
  } else return PETSC_FALSE;
}

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
  PetscLogDouble tlog[11];
  PetscInt       slot_h,slot_c[20];
  EnthalpyPackCtx pack;
  
  PetscFunctionBegin;
  PetscTime(&tlog[0]);
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

  // Update BC list
  bclist = fd->bclist;
  if (fd->bclist->evaluate) {
    PetscCall(fd->bclist->evaluate(dm,x,bclist,bclist->data));
  }
  PetscTime(&tlog[6]);

  // Get local domain
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Create residual local vector
  PetscCall(DMCreateLocalVector(dm, &flocal)); 
  PetscCall(DMStagVecGetArray(dm, flocal, &ff)); 

  // Map global vectors to local domain
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMGetLocalVector(dmP, &Plocal)); 
  PetscCall(DMGlobalToLocal (dmP, en->xP, INSERT_VALUES, Plocal)); 

  // Map the previous time step vectors
  if (en->timesteptype != TS_NONE) {
    PetscCall(DMGetLocalVector(dm, &xprevlocal)); 
    PetscCall(DMGlobalToLocal (dm, en->xprev, INSERT_VALUES, xprevlocal)); 
    PetscCall(DMGetLocalVector(dmcoeff, &coeffprevlocal)); 
    PetscCall(DMGlobalToLocal (dmcoeff, en->coeffprev, INSERT_VALUES, coeffprevlocal)); 
    PetscCall(DMGetLocalVector(dmP, &Pprevlocal)); 
    PetscCall(DMGlobalToLocal (dmP, en->xPprev, INSERT_VALUES, Pprevlocal)); 

    // Check time step
    if (!en->dt) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A valid time step size for FD-PDE ENTHALPY was not set! Set with FDPDEEnthalpySetTimestep()");
    }
  }
  PetscTime(&tlog[1]);

  if (en->timesteptype != TS_NONE) {
    PetscCall(PetscCalloc1((size_t)((nx+4)*(nz+4)),&thm_prev));
    PetscCall(PetscCalloc1((size_t)((nx+4)*(nz+4)),&cff_prev));
    PetscCall(ApplyEnthalpyMethod(fd,dm,xprevlocal,dmP,Pprevlocal,fd->dm_btype0,fd->dm_btype1,en,thm_prev,"prev")); 
    PetscCall(UpdateCoeffStructure(fd,dmcoeff,coeffprevlocal,cff_prev));
  }
  PetscTime(&tlog[2]);

  // update enthalpy and coeff cell data
  PetscCall(PetscCalloc1((size_t)((nx+4)*(nz+4)),&thm));
  PetscCall(PetscCalloc1((size_t)((nx+4)*(nz+4)),&cff));
  PetscCall(ApplyEnthalpyMethod(fd,dm,xlocal,dmP,Plocal,fd->dm_btype0,fd->dm_btype1,en,thm,NULL)); 
  PetscTime(&tlog[3]);

  // Update coefficients after enthalpy (for dependency of coeff on porosity)
  PetscCall(fd->ops->form_coefficient(fd,dm,x,dmcoeff,fd->coeff,fd->user_context));
  PetscTime(&tlog[4]);
  
  PetscCall(DMGetLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal)); 
  PetscCall(UpdateCoeffStructure(fd,dmcoeff,coefflocal,cff));
  PetscTime(&tlog[5]);

  // Residual evaluation
  PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&slot_h)); 
  for (ii = 0; ii<en->ncomponents-1; ii++) { // solve only for the first N-1 components
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii+1,&slot_c[ii])); 
  }
  
  PetscCall(EnthalpyPackCtx_InitDM(dm,fd->dm_btype0,fd->dm_btype1,coordx,coordz,&pack));
  PetscCall(EnthalpyPackCtx_InitState(thm,thm_prev,cff,cff_prev,en,&pack));

  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      // enthalpy
      PetscCall(EnthalpyPackCtx_Init_IJ(i,j,-1,&pack));
      PetscCall(EnthalpyResidual_pack(&pack,&fval)); 
      ff[j][i][slot_h] = fval;

      // bulk composition
      for (ii = 0; ii<en->ncomponents-1; ii++) { // solve only for the first N-1 components
        PetscCall(EnthalpyPackCtx_Init_IJ(i,j,ii,&pack));
        PetscCall(BulkCompositionResidual_pack(&pack,&fval)); 
        ff[j][i][slot_c[ii]] = fval;
      }
    }
  }
  PetscTime(&tlog[7]);

  // Boundary conditions - only element dofs
  if (en->form_user_bc) { // internal BC
    PetscCall(en->form_user_bc(dm,x,ff,en->user_context));
  }
  PetscTime(&tlog[8]);
  PetscCall(DMStagBCListApply_Enthalpy(dm,xlocal,bclist->bc_e,bclist->nbc_element,ff,pack));
  PetscTime(&tlog[9]);

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,flocal,&ff)); 
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  PetscCall(DMRestoreLocalVector(dmcoeff,&coefflocal)); 
  PetscCall(DMRestoreLocalVector(dmP, &Plocal)); 

  PetscCall(PetscFree(thm));
  PetscCall(PetscFree(cff));

  if (en->timesteptype != TS_NONE) {
    PetscCall(PetscFree(thm_prev));
    PetscCall(PetscFree(cff_prev));
    PetscCall(DMRestoreLocalVector(dm, &xprevlocal)); 
    PetscCall(DMRestoreLocalVector(dmcoeff, &coeffprevlocal)); 
    PetscCall(DMRestoreLocalVector(dmP, &Pprevlocal)); 
  }

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,f)); 
  PetscCall(DMLocalToGlobalEnd  (dm,flocal,INSERT_VALUES,f)); 

  PetscCall(VecDestroy(&flocal)); 
  PetscTime(&tlog[10]);

  if (fd->log_info) {
    printf("  FormFunction_Enthalpy: total                       %1.2e\n",tlog[9]-tlog[0]);
    printf("  FormFunction_Enthalpy: g2l(input)                  %1.2e\n",tlog[1]-tlog[0]);
    printf("  FormFunction_Enthalpy: en->timesteptype != TS_NONE %1.2e\n",tlog[2]-tlog[1]);
    printf("  FormFunction_Enthalpy: ApplyEnthalpyMethod         %1.2e\n",tlog[3]-tlog[2]);
    printf("  FormFunction_Enthalpy: form_coefficient            %1.2e\n",tlog[4]-tlog[3]);
    printf("  FormFunction_Enthalpy: g2l+UpdateCoeffStructure    %1.2e\n",tlog[5]-tlog[4]);
    printf("  FormFunction_Enthalpy: bclist->eval                %1.2e\n",tlog[6]-tlog[5]);
    printf("  FormFunction_Enthalpy: cell-loop                   %1.2e\n",tlog[7]-tlog[6]);
    printf("  FormFunction_Enthalpy: form_user_bc                %1.2e\n",tlog[8]-tlog[7]);
    printf("  FormFunction_Enthalpy: DMStagBCListApply_Enthalpy  %1.2e\n",tlog[9]-tlog[8]);
    printf("  FormFunction_Enthalpy: g2l(output)                 %1.2e\n",tlog[10]-tlog[9]);
    printf("----------------------------------------------------------------------\n");
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
UpdateCoeffStructure - collects the coefficients variables for the entire domain
Use: internal
@*/
// ---------------------------------------
PetscErrorCode UpdateCoeffStructure(FDPDE fd, DM dmcoeff,Vec coefflocal,CoeffState *cff)
{
  PetscInt       i,j,sx,sz,nx,nz,idx;
  
  const PetscReal ***_coeff;
  PetscInt       ii,dof0,dof1,dof2;
  PetscScalar    cE[6],cF[20];
  PetscInt       slot_cell[20];
  PetscInt       slot_face[4*20];
  
  PetscFunctionBegin;
  
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetDOF(dmcoeff,&dof0,&dof1,&dof2,NULL));
  PetscCall(DMStagVecGetArrayRead(dmcoeff,coefflocal,&_coeff));

  for (ii = 0; ii<dof2; ii++) { // element
    DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,ii,&slot_cell[ii]);
  }
  for (ii = 0; ii<dof1; ii++) { // faces
    DMStagGetLocationSlot(dmcoeff, DMSTAG_LEFT, ii, &slot_face[4*ii+0]);
    DMStagGetLocationSlot(dmcoeff, DMSTAG_RIGHT,ii, &slot_face[4*ii+1]);
    DMStagGetLocationSlot(dmcoeff, DMSTAG_DOWN, ii, &slot_face[4*ii+2]);
    DMStagGetLocationSlot(dmcoeff, DMSTAG_UP,   ii, &slot_face[4*ii+3]);
  }
  
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      idx = SingleDimIndex(i-sx+2,j-sz+2,nz+4);
      
      for (ii = 0; ii<dof2; ii++) { // element
        cE[ii] = _coeff[ j ][ i ][ slot_cell[ii] ];
      }

      for (ii = 0; ii<4*dof1; ii++) { // faces
        cF[ii] = _coeff[ j ][ i ][ slot_face[ii] ];
      }

      // assign values
      cff[idx].A1 = cE[COEFF_A1]; cff[idx].A2 = cE[COEFF_A2];
      cff[idx].B1 = cE[COEFF_B1]; cff[idx].B2 = cE[COEFF_B2];
      cff[idx].D1 = cE[COEFF_D1]; cff[idx].D2 = cE[COEFF_D2];
      
      for (ii = 0; ii<4; ii++) {
        cff[idx].C1[ii] = cF[4*COEFF_C1+ii];
        cff[idx].C2[ii] = cF[4*COEFF_C2+ii];
        cff[idx].v[ii]  = cF[4*COEFF_v +ii];
        cff[idx].vf[ii] = cF[4*COEFF_vf+ii];
        cff[idx].vs[ii] = cF[4*COEFF_vs+ii];
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dmcoeff,coefflocal,&_coeff));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
ApplyEnthalpyMethod - apply enthalpy method during each solver iteration; it collects enthalpy variables for the entire domain
Use: internal
@*/
// ---------------------------------------
PetscErrorCode ApplyEnthalpyMethod(FDPDE fd, DM dm,Vec xlocal,DM dmP, Vec Plocal,DMBoundaryType dm_btype0,DMBoundaryType dm_btype1,EnthalpyData *en,ThermoState *thm, const char prefix[])
{
  PetscInt       ii,i,j,sx,sz,nx,nz,idx,nreports, gnreports;
  PetscInt       Nx, Nz;
  PetscScalar    H,C[MAX_COMPONENTS],P,phi,T,TP,CS[MAX_COMPONENTS],CF[MAX_COMPONENTS];
  PetscBool      passed = PETSC_TRUE;
  PetscMPIInt    rank;
  PetscBool      flgx = PETSC_FALSE, flgz=PETSC_FALSE;
  
  PetscInt       dof0,dof1,dof2,*dm_slot,dmP_slot;
  const PetscScalar ***_xlocal,***_Plocal;
  
  PetscFunctionBegin;
  
  Nx = fd->Nx;
  Nz = fd->Nz;
  
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  
  PetscCall(DMStagVecGetArrayRead(dm,xlocal,&_xlocal));
  PetscCall(DMStagVecGetArrayRead(dmP,Plocal,&_Plocal));

  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
  PetscCall(PetscCalloc1(dof2,&dm_slot)); 
  for (ii = 0; ii<dof2; ii++) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii,&dm_slot[ii])); 
  }
  PetscCall(DMStagGetLocationSlot(dmP,DMSTAG_ELEMENT,0,&dmP_slot)); 
  
  if (dm_btype0==DM_BOUNDARY_PERIODIC) flgx = PETSC_TRUE;
  if (dm_btype1==DM_BOUNDARY_PERIODIC) flgz = PETSC_TRUE;

  // compute ghosted enthalpy data
  for (j = sz-2; j<sz+nz+2; j++) {
    for (i = sx-2; i<sx+nx+2; i++) {
      PetscBool flg;
      flg = ghost_point_enthalpy(flgx,flgz,i,j,Nx,Nz);
      if (flg) {
        EnthEvalErrorCode thermo_dyn_error_code = 0;
        
        H = 0.0; phi = 0.0; T = 0.0; P = 0.0; TP = 0.0;
        for (ii = 0; ii<en->ncomponents; ii++) { C[ii] = 0.0; CF[ii] = 0.0; CS[ii] = 0.0;}
        
        idx = SingleDimIndex(i-sx+2,j-sz+2,nz+4);
        {
          PetscScalar sum_C = 0.0;
          
          // assign values H, C
          H = _xlocal[j][i][dm_slot[0]];
          for (ii = 1; ii<dof2; ii++) {
            sum_C  += _xlocal[j][i][dm_slot[ii]];
            C[ii-1] = _xlocal[j][i][dm_slot[ii]];
          }
          C[dof2-1] = 1.0 - sum_C;
        }
        
        P = _Plocal[j][i][dmP_slot];
        
        thermo_dyn_error_code = en->form_enthalpy_method(H,C,P,&T,&phi,CF,CS,en->ncomponents,en->user_context);
        if (thermo_dyn_error_code != 0) passed = PETSC_FALSE;
        
        if (en->form_TP) { PetscCall(en->form_TP(T,P,&TP,en->user_context_tp)); }
        else TP = T;
        
        thm[idx].P  = P;
        thm[idx].TP = TP;
        thm[idx].T  = T;
        thm[idx].H  = H;
        thm[idx].phi = phi;
        for (ii = 0; ii<en->ncomponents; ii++) {
          thm[idx].C[ii]  = C[ii];
          thm[idx].CS[ii] = CS[ii];
          thm[idx].CF[ii] = CF[ii];
        }
        thm[idx].err = thermo_dyn_error_code;
      }
    }
  }
  
  PetscCall(PetscFree(dm_slot));
  PetscCall(DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal));
  PetscCall(DMStagVecRestoreArrayRead(dmP,Plocal,&_Plocal));

  // output failure report to file per rank
  if (!passed) {
    char        fname[PETSC_MAX_PATH_LEN];
    PetscBool   stop_failed = PETSC_FALSE;
    PetscViewer viewer;
    PetscMPIInt rank;
    
    PetscCall(MPI_Comm_rank(fd->comm,&rank));
    if (prefix) PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"enthalpy_failure_%s_%D.rank%D.report",prefix,en->nreports,rank);
    else PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"enthalpy_failure_%D.rank%D.report",en->nreports,rank);
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,fname,&viewer));
    PetscCall(ApplyEnthalpyReport_Failure(fd,viewer,en,thm));
    PetscCall(PetscViewerDestroy(&viewer));
    en->nreports++;
    
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-stop_enthalpy_failed",&stop_failed,NULL));
    if (stop_failed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SIG,"The Enthalpy Method has failed! Investigate the enthalpy failure reports for detailed information.");
  }
  PetscCall(MPI_Allreduce(&en->nreports,&gnreports,1,MPI_INT,MPI_MAX,fd->comm));
  en->nreports = gnreports;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
ApplyEnthalpyReport_Failure - report failure of enthalpy data to file
Use: internal
@*/
// ---------------------------------------
PetscErrorCode ApplyEnthalpyReport_Failure(FDPDE fd,PetscViewer viewer, EnthalpyData *en,ThermoState *thm)
{
  PetscInt   ii,i,j,sx,sz,nx,nz,idx, its;
  const char *vname;

  PetscFunctionBegin;

  PetscCall(PetscViewerFileGetName(viewer,&vname));
  PetscPrintf(PETSC_COMM_SELF,"=====================================================================\n");
  PetscPrintf(PETSC_COMM_SELF,"====  ENTHALPY METHOD has failed! \n");
  PetscPrintf(PETSC_COMM_SELF,"====  Please inspect the following file to diagnose the problem\n");
  PetscPrintf(PETSC_COMM_SELF,"====  %s\n",vname);
  PetscPrintf(PETSC_COMM_SELF,"=====================================================================\n");

  PetscViewerASCIIPrintf(viewer,"ENTHALPY METHOD FAILURE REPORT\n");
  PetscViewerASCIIPrintf(viewer,"[PDE summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"pde: Enthalpy\n");
  PetscViewerASCIIPrintf(viewer,"description: %s\n",fd->description);
  PetscViewerASCIIPopTab(viewer);

  PetscViewerASCIIPrintf(viewer,"[ENTHALPY METHOD summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"description: %s\n",en->description_enthalpy);
  PetscViewerASCIIPopTab(viewer);

  PetscViewerASCIIPrintf(viewer,"[SNES summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscCall(SNESGetIterationNumber(fd->snes,&its));
  PetscViewerASCIIPrintf(viewer,"iterations performed: %D\n",its);
  PetscViewerASCIIPopTab(viewer);

  // output enthalpy data cell wise
  PetscCall(DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  PetscViewerASCIIPrintf(viewer,"[ENTHALPY ERRORS]\n");
  PetscViewerASCIIPushTab(viewer);
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      const char *err_message;
      idx = SingleDimIndex(i-sx+2,j-sz+2,nz+4);;
      err_message = EnthalpyErrorTypeNames(thm[idx].err);
      PetscViewerASCIIPrintf(viewer," Error %s encountered in cell [i=%d j=%d]  \n",err_message,i,j);
    }
  }
  PetscViewerASCIIPopTab(viewer);

  PetscViewerASCIIPrintf(viewer,"[ENTHALPY data]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"i  j  H            ");
  for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"C[%d]         ",ii);}
  PetscViewerASCIIPrintf(viewer,"P             T             PHI          ");
  for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"CF[%d]        ",ii);}
  for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"CS[%d]        ",ii);}
  PetscViewerASCIIPrintf(viewer,"\n");

  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      idx = SingleDimIndex(i-sx+2,j-sz+2,nz+4);
      PetscViewerASCIIPrintf(viewer,"%d  %d  %1.6e ",i,j,thm[idx].H);
      for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"%1.6e ",thm[idx].C[ii]);}
      PetscViewerASCIIPrintf(viewer,"%1.6e  %1.6e  %1.6e ",thm[idx].P,thm[idx].T,thm[idx].phi);
      for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"%1.6e ",thm[idx].CF[ii]);}
      for (ii = 0; ii<en->ncomponents; ii++) { PetscViewerASCIIPrintf(viewer,"%1.6e ",thm[idx].CS[ii]);}
      PetscViewerASCIIPrintf(viewer,"\n");
    }
  }
  PetscViewerASCIIPopTab(viewer);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
EnthalpySteadyStateOperator - (ENTHALPY) calculates the steady state enthalpy residual per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpySteadyStateOperator_pack(EnthalpyPackCtx *pack,AdvectSchemeType advtype,PetscInt state,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, nx,nz,sx,sz,icenter, idx[9];
  PetscScalar    xxTP[9], xxPHIs[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A1, B1, D1, C1_Left, C1_Right, C1_Down, C1_Up, v[5], vs[5];
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  
  DM dm = pack->dm;
  const ThermoState *thm = pack->thm;
  const CoeffState *cff = pack->cff;
  const PetscScalar **coordx = pack->coordx;
  const PetscScalar **coordz = pack->coordz;
  PetscInt i = pack->i;
  PetscInt j = pack->j;
  
  PetscFunctionBegin;
  
  if (state == 0) {
    thm = pack->thm_prev;
    cff = pack->cff_prev;
  }
  sx = pack->sx;
  sz = pack->sz;
  nx = pack->nx;
  nz = pack->nz;
  Nx = pack->Nx;
  Nz = pack->Nz;
  icenter = pack->icenter;
  
  idx[0] = SingleDimIndex(i-sx+2,j-sz+2,nz+4);
  
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
  idx[0] = SingleDimIndex(i  -sx+2,j  -sz+2,nz+4); // C
  idx[1] = SingleDimIndex(i-1-sx+2,j  -sz+2,nz+4); // W
  idx[2] = SingleDimIndex(i+1-sx+2,j  -sz+2,nz+4); // E
  idx[3] = SingleDimIndex(i  -sx+2,j-1-sz+2,nz+4); // S
  idx[4] = SingleDimIndex(i  -sx+2,j+1-sz+2,nz+4); // N
  idx[5] = SingleDimIndex(i-2-sx+2,j  -sz+2,nz+4); // WW
  idx[6] = SingleDimIndex(i+2-sx+2,j  -sz+2,nz+4); // EE
  idx[7] = SingleDimIndex(i  -sx+2,j-2-sz+2,nz+4); // SS
  idx[8] = SingleDimIndex(i  -sx+2,j+2-sz+2,nz+4); // NN
  
  if (pack->dm_btype0!=DM_BOUNDARY_PERIODIC) {
    if (i == 1   ) idx[5] = idx[2];
    if (i == Nx-2) idx[6] = idx[1];
    if (i == 0   ) { idx[1] = idx[0]; idx[5] = idx[2]; }
    if (i == Nx-1) { idx[2] = idx[0]; idx[6] = idx[1]; }
  }

  if (pack->dm_btype1!=DM_BOUNDARY_PERIODIC) {
    if (j == 1   ) idx[7] = idx[4];
    if (j == Nz-2) idx[8] = idx[3];
    if (j == 0   ) { idx[3] = idx[0]; idx[7] = idx[4]; }
    if (j == Nz-1) { idx[4] = idx[0]; idx[8] = idx[3]; }
  }
  
  for (ii = 0; ii<9; ii++) {
    xxTP[ii] = thm[idx[ii]].TP;
    xxPHIs[ii] = 1.0 - thm[idx[ii]].phi;
  }
  
  // Calculate diff residual
  dQ2dx = C1_Right*(xxTP[2]-xxTP[0])/dx[0] - C1_Left*(xxTP[0]-xxTP[1])/dx[1];
  dQ2dz = C1_Up   *(xxTP[4]-xxTP[0])/dz[0] - C1_Down*(xxTP[0]-xxTP[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];
  
  // Calculate adv residual
  PetscCall(AdvectionResidual(v, xxTP,  dx,dz,pack->en->dt,advtype,&adv1)); 
  PetscCall(AdvectionResidual(vs,xxPHIs,dx,dz,pack->en->dt,advtype,&adv2)); 
  
  ffi  = A1*adv1 +B1*adv2 + diff + D1;
  *ff = ffi;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
EnthalpyResidual - (ENTHALPY) calculates the residual for H/TP per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnthalpyResidual_pack(EnthalpyPackCtx *pack, PetscScalar *_fval)
{
  PetscInt       idx;
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;

  PetscFunctionBegin;
  if (pack->en->timesteptype == TS_NONE) {
    // steady-state operator
    PetscCall(EnthalpySteadyStateOperator_pack(pack,pack->en->advtype,1,&fval)); 
  } else {
    // time-dependent solution
    PetscCall(EnthalpySteadyStateOperator_pack(pack,pack->en->advtype,0,&fval0)); 
    PetscCall(EnthalpySteadyStateOperator_pack(pack,pack->en->advtype,1,&fval1)); 
    
    idx = SingleDimIndex(pack->i-pack->sx+2,pack->j-pack->sz+2,pack->nz+4);
    xx     = pack->thm[idx].H;
    xxprev = pack->thm_prev[idx].H;
    
    fval = xx - xxprev + pack->en->dt*(pack->en->theta*fval1 + (1.0-pack->en->theta)*fval0);
  }
  *_fval = fval;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
BulkCompositionSteadyStateOperator - (ENTHALPY) calculates the steady state bulk composition residual per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode BulkCompositionSteadyStateOperator_pack(EnthalpyPackCtx *pack,AdvectSchemeType advtype,PetscInt state,PetscScalar *ff)
{
  PetscScalar    ffi;
  PetscInt       ii, Nx, Nz, icenter, sx,sz,nx,nz,idx[9];
  PetscScalar    xxCF[9], xxCS[9], xxPHI[9], xxPHIs[9], f1[9], f2[9];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A2, B2, D2, C2_Left, C2_Right, C2_Down, C2_Up, vs[5], vf[5];
  PetscScalar    phi_Left, phi_Right, phi_Down, phi_Up;
  PetscScalar    dQ2dx, dQ2dz, diff, adv1, adv2;
  DMStagStencil  point[15];
  
  DM dm = pack->dm;
  const ThermoState *thm = pack->thm;
  const CoeffState *cff = pack->cff;
  const PetscScalar **coordx = pack->coordx;
  const PetscScalar **coordz = pack->coordz;
  PetscInt i = pack->i;
  PetscInt j = pack->j;
  PetscInt icomp = pack->icomp;
  
  PetscFunctionBegin;

  if (state == 0) {
    thm = pack->thm_prev;
    cff = pack->cff_prev;
  }
  sx = pack->sx;
  sz = pack->sz;
  nx = pack->nx;
  nz = pack->nz;
  Nx = pack->Nx;
  Nz = pack->Nz;
  icenter = pack->icenter;

  // Coefficients
  idx[0] = SingleDimIndex(i-sx+2,j-sz+2,nz+4);
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
  idx[0] = SingleDimIndex(i  -sx+2,j  -sz+2,nz+4); // C
  idx[1] = SingleDimIndex(i-1-sx+2,j  -sz+2,nz+4); // W
  idx[2] = SingleDimIndex(i+1-sx+2,j  -sz+2,nz+4); // E
  idx[3] = SingleDimIndex(i  -sx+2,j-1-sz+2,nz+4); // S
  idx[4] = SingleDimIndex(i  -sx+2,j+1-sz+2,nz+4); // N
  idx[5] = SingleDimIndex(i-2-sx+2,j  -sz+2,nz+4); // WW
  idx[6] = SingleDimIndex(i+2-sx+2,j  -sz+2,nz+4); // EE
  idx[7] = SingleDimIndex(i  -sx+2,j-2-sz+2,nz+4); // SS
  idx[8] = SingleDimIndex(i  -sx+2,j+2-sz+2,nz+4); // NN
  
  if (pack->dm_btype0!=DM_BOUNDARY_PERIODIC) {
    if (i == 1   ) idx[5] = idx[2];
    if (i == Nx-2) idx[6] = idx[1];
    if (i == 0   ) { idx[1] = idx[0]; idx[5] = idx[2]; }
    if (i == Nx-1) { idx[2] = idx[0]; idx[6] = idx[1]; }
  }

  if (pack->dm_btype1!=DM_BOUNDARY_PERIODIC) {
    if (j == 1   ) idx[7] = idx[4];
    if (j == Nz-2) idx[8] = idx[3];
    if (j == 0   ) { idx[3] = idx[0]; idx[7] = idx[4]; }
    if (j == Nz-1) { idx[4] = idx[0]; idx[8] = idx[3]; }
  }
  
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
  PetscCall(AdvectionResidual(vs,f1,dx,dz,pack->en->dt,advtype,&adv1)); 
  PetscCall(AdvectionResidual(vf,f2,dx,dz,pack->en->dt,advtype,&adv2)); 
  
  ffi  = A2*adv1 + B2*adv2 + diff + D2;
  *ff = ffi;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
BulkCompositionResidual - (ENTHALPY) calculates the residual for bulk composition per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode BulkCompositionResidual_pack(EnthalpyPackCtx *pack, PetscScalar *_fval)
{
  PetscInt       idx;
  PetscScalar    xx, xxprev;
  PetscScalar    fval=0.0, fval0=0.0, fval1=0.0;
  PetscInt       ii;
  
  PetscFunctionBegin;
  
  ii = pack->icomp;
  if (pack->en->timesteptype == TS_NONE) {
    // steady-state operator
    PetscCall(BulkCompositionSteadyStateOperator_pack(pack,pack->en->advtype,1,&fval)); 
  } else {
    // time-dependent solution
    PetscCall(BulkCompositionSteadyStateOperator_pack(pack,pack->en->advtype,0,&fval0)); 
    PetscCall(BulkCompositionSteadyStateOperator_pack(pack,pack->en->advtype,1,&fval1)); 
    
    idx = SingleDimIndex(pack->i-pack->sx+2,pack->j-pack->sz+2,pack->nz+4);
    xx     = pack->thm[idx].C[ii];
    xxprev = pack->thm_prev[idx].C[ii];
    
    fval = xx - xxprev + pack->en->dt*(pack->en->theta*fval1 + (1.0-pack->en->theta)*fval0);
  }
  *_fval = fval;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagBCListApply_Enthalpy - function to apply boundary conditions for ENTHALPY equations

Use: internal
@*/
// ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagBCListApply_Enthalpy"
PetscErrorCode DMStagBCListApply_Enthalpy(DM dm, Vec xlocal,DMStagBC *bclist, PetscInt nbc, PetscScalar ***ff,EnthalpyPackCtx pack)
{
  PetscScalar    xx, fval;
  PetscInt       i, j, ii, ibc, idx, Nx, Nz;
  PetscScalar    ***_xlocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMStagVecGetArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {
    if (bclist[ibc].type == BC_PERIODIC) { // normal stencil for i,j - should come before other BCs are set
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if (bclist[ibc].point.c==0) {
        ierr = EnthalpyPackCtx_Init_IJ(i,j,-1,&pack);CHKERRQ(ierr);
        ierr = EnthalpyResidual_pack(&pack,&fval); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      } else {
        ii = bclist[ibc].point.c - 1;
        ierr = EnthalpyPackCtx_Init_IJ(i,j,ii,&pack);CHKERRQ(ierr);
        ierr = BulkCompositionResidual_pack(&pack,&fval); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }

    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      xx = _xlocal[j][i][idx];
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if (j == 0) { // down
        ff[j][i][idx] = _xlocal[j+1][i][idx] - _xlocal[j][i][idx] - bclist[ibc].val;
      }

      if (j == Nz-1) { // up
        ff[j][i][idx] = _xlocal[j][i][idx] - _xlocal[j-1][i][idx] - bclist[ibc].val;
      }

      if (i == 0) { // left
        ff[j][i][idx] = _xlocal[j][i+1][idx] - _xlocal[j][i][idx] - bclist[ibc].val;
      }

      if (i == Nx-1) { // right
        ff[j][i][idx] = _xlocal[j][i][idx] - _xlocal[j][i-1][idx] - bclist[ibc].val;
      }
    }
  }

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
