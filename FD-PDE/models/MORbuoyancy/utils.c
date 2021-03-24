#include "MORbuoyancy.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// ---------------------------------------
// DoOutput
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DoOutput"
PetscErrorCode DoOutput(FDPDE fdPV, FDPDE fdHC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  DM             dmPVcoeff, dmHCcoeff,dmP;
  Vec            xPVcoeff, xHCcoeff, xP, xPprev;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  if ((usr->par->restart) && (usr->nd->istep==usr->par->restart)) {
    ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d_r",usr->nd->istep);
  } else {
    ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);
  }
  ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);

  // Output bag and parameters
  ierr = OutputParameters(usr);CHKERRQ(ierr); 

  // Output solution vectors
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xphiT_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xphiT,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmVel,usr->xVel,fout);CHKERRQ(ierr);

  if (usr->nd->istep > 0) {
    // coefficients
    ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(dmHCcoeff,xHCcoeff,fout);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdPV,&dmPVcoeff,&xPVcoeff);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPVcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(dmPVcoeff,xPVcoeff,fout);CHKERRQ(ierr);

    // material properties eta, permeability, density
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_matProp_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(usr->dmmatProp,usr->xmatProp,fout);CHKERRQ(ierr);
  }

  // residuals
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPV,fdPV->r,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,fdHC->r,fout);CHKERRQ(ierr);

  // lithostatic pressure
  ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressure_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmP,xP,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressurePrev_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmP,xPprev,fout);CHKERRQ(ierr);

  ierr = VecDestroy(&xP);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// CreateDirectory
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CreateDirectory"
PetscErrorCode CreateDirectory(const char *name)
{
  PetscMPIInt rank;
  int         status;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // create a new directory if it doesn't exist on rank zero
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if(rank==0) {
    status = mkdir(name,0777);
    if(!status) PetscPrintf(PETSC_COMM_WORLD,"# New directory created: %s \n",name);
    else        PetscPrintf(PETSC_COMM_WORLD,"# Did not create new directory: %s \n",name);
  }
  ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Compute crustal thickness and fluxes out in sill
// ---------------------------------------
PetscErrorCode ComputeSillOutflux(void *ctx) 
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, isill, sx, sz, nx, nz, iprev,inext;
  PetscScalar    **coordx,**coordz;
  PetscScalar    sill_F, sill_C, gsill_F, gsill_C;
  DM             dmHC, dmVel, dmEnth;
  Vec            xphiTlocal, xVellocal, xEnthlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dmHC = usr->dmHC;
  dmVel= usr->dmVel;
  dmEnth=usr->dmEnth;

  // get coordinates of dmVel for edges
  ierr = DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmVel,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmVel,RIGHT,&inext);CHKERRQ(ierr); 

  ierr = DMGetLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmHC, usr->xphiT, INSERT_VALUES, xphiTlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmVel, &xVellocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmVel, usr->xVel, INSERT_VALUES, xVellocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, usr->xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);

  // check indices for sill
  for (i = sx; i <sx+nx; i++) {
    PetscScalar xc;
    xc = (coordx[i][inext]+coordx[i][iprev])*0.5;
    if (xc <= usr->nd->xsill) isill = i;
  }

  // maybe check depth corresponding to max fluid velocity 
  // (how far deep can a melt package still travel to the surface?)

  sill_F = 0.0;
  sill_C = 0.0;

  // Loop over local domain for sill - make it parallel (isill may not be on this processor)
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i < isill+1; i++) {
      DMStagStencil point[2], pointT;
      PetscScalar v[2], vf, dz, phi, Cf, zf, zc, flux_ij;

      // get fluid velocity (z)
      point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;
      ierr = DMStagVecGetValuesStencil(dmVel,xVellocal,2,point,v); CHKERRQ(ierr);
      vf = (v[0]+v[1])*0.5;

      // get porosity and CF
      pointT.i = i; pointT.j = j; pointT.loc = ELEMENT; 
      pointT.c = 0; ierr = DMStagVecGetValuesStencil(dmHC,xphiTlocal,1,&pointT,&phi); CHKERRQ(ierr);
      pointT.c = 9; ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,1,&pointT,&Cf); CHKERRQ(ierr);

      dz = coordz[j][inext]-coordz[j][iprev];
      zc = -(coordz[j][inext]+coordz[j][iprev])*0.5;
      zf = vf*usr->nd->dt;

      // compute fluxes - check if flux reaches the surface in this time step
      if (zf > zc) { // outflux
        flux_ij = phi*vf*dz;
        sill_F += flux_ij;
        sill_C += flux_ij*Cf;
      }
    }
  }

  // Parallel
  ierr = MPI_Allreduce(&sill_F,&gsill_F,1,MPI_REAL,MPI_SUM,usr->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sill_C,&gsill_C,1,MPI_REAL,MPI_SUM,usr->comm);CHKERRQ(ierr);

  PetscScalar C, F, h_crust, t;
  t = (usr->nd->t+usr->nd->dt)*usr->scal->t/SEC_YEAR; 
  if (gsill_F==0.0) C = usr->par->C0;
  else C = gsill_C/gsill_F*usr->par->DC+usr->par->C0;
  F = gsill_F*usr->par->rho0*usr->scal->v*usr->scal->x*SEC_YEAR; // kg/m/year
  h_crust = F/(usr->par->rho0*usr->par->U0)*1.0e2; // m

  if (F < 1e-20) F = 1e-20;
  if (h_crust<1e-20) h_crust = 1e-20;

  // Output
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# SILL FLUXES: t = %1.12e [yr] C = %1.12e [wt. frac.] F = %1.12e [kg/m/yr] h_crust = %1.12e [m]\n",t,C,F,h_crust);

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmVel, &xVellocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth,&xEnthlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Output Parameters
// ---------------------------------------
PetscErrorCode OutputParameters(void *ctx) 
{
  UsrData        *usr = (UsrData*)ctx;
  char           prefix[FNAME_LENGTH],fout[FNAME_LENGTH],string[FNAME_LENGTH];
  FILE           *fp = NULL;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Output bag and parameters
  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%s/%s",usr->par->fdir_out,"parameters");
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s.pbin",prefix);
  ierr = PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);

  // create an additional file for loading in python
  ierr = PetscSNPrintf(string,sizeof(string),"%s.py",prefix);
  fp = fopen(string,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
  fprintf(fp,"import PetscBinaryIO as pio\n");
  fprintf(fp,"import numpy as np\n\n");
  fprintf(fp,"def _PETScBinaryFilePrefix():\n");
  fprintf(fp,"  return \"%s\"\n",prefix);
  fprintf(fp,"\n");
  fprintf(fp,"def _PETScBinaryLoad():\n");
  fprintf(fp,"  io = pio.PetscBinaryIO()\n");
  fprintf(fp,"  filename = \"%s\"\n",fout);
  fprintf(fp,"  data = dict()\n");
  fprintf(fp,"  with open(filename) as fp:\n");

  // parameters - scal
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->x,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->v,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->t,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->K,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->P,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->eta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->rho,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->H,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->Gamma,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalx'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalv'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalK'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalP'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scaleta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalrho'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalH'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalGamma'] = v\n");

  // parameters - nd
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->L,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->H,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->zmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xsill,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->U0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->visc_ratio,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_min,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_max,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['L'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['H'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['zmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xsill'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['U0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['visc_ratio'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_min'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_max'] = v\n");

  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->istep,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->t,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dt,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->tmax,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dtmax,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readInteger(fp)\n"); fprintf(fp,"    data['istep'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['t'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['tmax'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dtmax'] = v\n");

  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->delta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->alpha_s,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->beta_s,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->A,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->S,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeT,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeC,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->thetaS,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->G,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->RM,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['delta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['alpha_s'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['beta_s'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['A'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['S'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['PeT'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['PeC'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['thetaS'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['G'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['RM'] = v\n");

  // Note: readBag() in PetscBinaryIO.py is not yet implemented, so will close the python file without reading bag
  // Also some bag parameters needed for scaling
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->C0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->DC,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->T0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->DT,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['C0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DC'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['T0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DT'] = v\n");

  // output bag
  ierr = PetscBagView(usr->bag,viewer);CHKERRQ(ierr);

  fprintf(fp,"    return data\n\n");
  fclose(fp);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Load Parameters
// ---------------------------------------
PetscErrorCode LoadParametersFromFile(void *ctx) 
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/parameters.pbin",usr->par->fdir_out);
  ierr = PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  // parameters - scal
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->x,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->v,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->K,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->P,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->eta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->rho,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->H,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->Gamma,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // parameters - nd
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->L,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->H,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->zmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->xsill,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->U0,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->visc_ratio,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_min,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_max,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->istep,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dt,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->tmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dtmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->delta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->alpha_s,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->beta_s,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->A,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->S,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeC,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->thetaS,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->G,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->RM,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // these bag parameters are needed for scaling in python
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->C0,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->DC,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->T0,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->DT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // read bag
  ierr = PetscBagLoad(viewer,usr->bag);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}