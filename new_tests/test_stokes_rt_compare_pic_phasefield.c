// ---------------------------------------
// run: ./test_stokes_rt_compare_pic_phasefield -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 21 -nz 21 -snes_type ksponly -snes_fd_color -nt 101 -log_view
// python test: ./python/test_stokes_rt_compare_pic_phasefield.py
// ---------------------------------------
static char help[] = "Application to solve an Rayleigh-Taylor instability and compare Particle-in-Cell and Phase Field method for material interfaces\n\n";

// define convenient names for DMStagStencilLocation
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#include "../new_src/fdpde_stokes.h"
#include "../new_src/fdpde_dmswarm.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, nt, ppcell, vfopt, method, tout;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    rho0, rho1, eta0, eta1, g, ya, y0, dt, eps, gamma;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH]; 
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmPV, dmf, swarm;
  Vec            f,fprev,volf,dfx,dfz,xVel;
} UsrData;

// static functions
static PetscScalar volf_2d(PetscScalar f1, PetscScalar f2, PetscScalar cc, PetscScalar ar)

{ PetscScalar tol, r10, r20, r1, r2, d1, d2, fchk, aa, result;

  tol = 1e-2;
  r10 = 2*f1 - 1.0;
  r20 = 2*f2 - 1.0;
  r1 = r10;
  r2 = r20;


  d1 = 2.0*cc*r1;
  d2 = 2.0*cc*r2;


  //1st check: is the interface too far away
  fchk = PetscAbs(d1+d2);
  aa   = sqrt(ar*ar + 1.0);
  if (fchk < aa ) {
    PetscScalar tx, tz;
    tx = (d1-d2);

    if (PetscAbs(tx)>1.0) {
      if (PetscAbs(tx)-1.0 > tol) {
        PetscPrintf(PETSC_COMM_WORLD, "f1 = %1.4f, f2 = %1.4f, d1= %1.4f, d2 = %1.4f, r10 = %1.4f, r20=%1.4f, cc = %1.4f, tx = %1.4f\n", f1, f2,d1, d2,r10, r20, cc, tx);
      }
      tx = tx/PetscAbs(tx);
    }

    tz = sqrt(1.0 - tx*tx);

    if (PetscAbs(tz)<tol) {
      //2.1 check: a horizontal line?
      if (d1+d2>=0) {result = PetscMin(0.5+(d1+d2), 1.0);}
      else          {result = 1.0 - PetscMin(0.5-(d1+d2), 1.0);}
    }
    else if (PetscAbs(tx)<tol) {
      //2.2 check: a vertical line?
      if (d1>=0) {result = PetscMin(0.5+d1/ar, 1.0);}
      else       {result = 1.0 - PetscMin(0.5-d1/ar, 1.0);}
    }
    else {
      //3 check: intersection with z = 0 and z = 1.0
      PetscScalar xb, xu, k0, k1, x0, x1;

      k0 = tz/tx;

      xb = 0.5*ar + d2/tz;
      xu = 0.5*ar + d1/tz;
      k1 = -k0*xb;

      if      (xu<=0.0) {x1 = 0.0;}
      else if (xu>=ar ) {x1 = ar; }
      else              {x1 = xu; }

      if      (xb<=0.0) {x0 = 0.0;}
      else if (xb>=ar ) {x0 = ar ;}
      else              {x0 = xb ;}

      result = (x1 - (0.5*k0*(x1*x1 - x0*x0) + k1*(x1-x0)))/ar;

      if (result <0 || result > 1.0) {
        PetscPrintf(PETSC_COMM_WORLD, "WRONG vvf, greater than 1 or smaller than zero, volf = %1.4f", result);}

    }
  }
  else if (f1>=0.5) {result = 1.0;} // line too far, only fluid 1
  else {result =0.0;}                 // line is too far away, only fluid 2

  return(result);
}

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);

PetscErrorCode Stokes_RT_PIC(void*);
PetscErrorCode SetSwarmInitialCondition(DM,void*);
PetscErrorCode FormCoefficient_PIC(FDPDE, DM, Vec, DM, Vec, void*);

PetscErrorCode Stokes_RT_PhaseField(void*);
PetscErrorCode SetInitialPhaseField(DM,Vec,void*);
PetscErrorCode FormCoefficient_PhaseField(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode InterpCornerFacePhaseF(DM,Vec);
PetscErrorCode UpdateVolFrac(DM,Vec,void*);
PetscErrorCode UpdateDF(DM,Vec,void*);
PetscErrorCode ExplicitStep(DM,Vec,Vec,PetscScalar,void*);

PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  A = eta(x,y)\n"
"  Bx = 0 \n" 
"  By = rho(x,y)*g \n"
"  C = 0\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0\n"
"  RIGHT: Vx = 0, dVz/dx = 0\n" 
"  DOWN: Vx = Vz = 0,\n"
"  UP: Vx = Vz = 0, \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Stokes_RT_PIC"
PetscErrorCode Stokes_RT_PIC(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dmPV,dmswarm;
  Vec            x;
  PetscInt       nx, nz, istep;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscLogDouble start_time, end_time;
  const char     *fieldname[] = {"rho"};
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL));
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_PIC,coeff_description,usr));
  PetscCall(FDPDEGetDM(fd,&dmPV));
  
  // Create a swarm object and assign several fields
  PetscCall(DMStagPICCreateDMSwarm(dmPV,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"rho",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));
  usr->swarm = dmswarm;

  // Create swarm coordinates and set initial conditions
  PetscInt ppcell[] = {usr->par->ppcell,usr->par->ppcell};
  PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.5,ppcell,COOR_INITIALIZE));
  PetscCall(SetSwarmInitialCondition(dmswarm,usr));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_pic-0.xmf",usr->par->fdir_out,usr->par->fname_out));
  // PetscCall(DMSwarmViewXDMF(dmswarm,fout));
  PetscCall(DMSwarmViewFieldsXDMF(dmswarm,fout,1,fieldname)); 

  for (istep=1; istep<usr->par->nt; istep++) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);
    PetscCall(PetscTime(&start_time)); 

    // FD SNES Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x)); 
    
    // Advect particles
    PetscCall(MPoint_AdvectRK1(dmswarm,dmPV,x,usr->par->dt));

    // Output
    if (istep%usr->par->tout == 0) { 
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_pic-%d.xmf",usr->par->fdir_out,usr->par->fname_out,istep));
      // PetscCall(DMSwarmViewXDMF(dmswarm,fout));
      PetscCall(DMSwarmViewFieldsXDMF(dmswarm,fout,1,fieldname)); 

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_pic_xPV_ts%d",usr->par->fdir_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,x,fout));
    }

    PetscCall(VecDestroy(&x));
    PetscCall(PetscTime(&end_time)); 
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n\n", end_time - start_time);
  }

  // Destroy objects
  PetscCall(DMDestroy(&dmPV));
  PetscCall(FDPDEDestroy(&fd));
  PetscCall(DMDestroy(&dmswarm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetSwarmInitialCondition
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetSwarmInitialCondition"
PetscErrorCode SetSwarmInitialCondition(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield;
  PetscInt  npoints,p;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
    PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    
    PetscCall(DMSwarmGetField(dmswarm,"eta",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      PetscScalar yinterface,xcoor,ycoor;
      
      xcoor = pcoor[2*p+0];
      ycoor = pcoor[2*p+1];
      yinterface = usr->par->ya * PetscCosReal(PETSC_PI * xcoor) + usr->par->y0;
      pfield[p] = usr->par->eta0;
      if (ycoor < yinterface) {
        pfield[p] = usr->par->eta1;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarm,"eta",NULL,NULL,(void**)&pfield));
    
    PetscCall(DMSwarmGetField(dmswarm,"rho",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      PetscScalar yinterface,xcoor,ycoor;
      
      xcoor = pcoor[2*p+0];
      ycoor = pcoor[2*p+1];
      yinterface = usr->par->ya * PetscCosReal(PETSC_PI * xcoor) + usr->par->y0;
      pfield[p] = usr->par->rho0;
      if (ycoor < yinterface) {
        pfield[p] = usr->par->rho1;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarm,"rho",NULL,NULL,(void**)&pfield));
    PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_PIC
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PIC"
PetscErrorCode FormCoefficient_PIC(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscScalar    g;
  DM             dmswarm;
  PetscFunctionBeginUser;

  g = -usr->par->g;
  dmswarm = usr->swarm;
  // PetscCall(VecZeroEntries(coeff));

  // Project swarm into coefficient
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"eta",dm,dmcoeff,2,1,coeff));//cell
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"eta",dm,dmcoeff,0,0,coeff));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"rho",dm,dmcoeff,1,0,coeff));//face

  // Get dm coordinates array
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMGlobalToLocalBegin(dmcoeff,coeff,INSERT_VALUES,coefflocal)); 
  PetscCall(DMGlobalToLocalEnd  (dmcoeff,coeff,INSERT_VALUES,coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      // A (eta) has already been projected on the cell center and vertex
      { // Bx = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }

      { // Bz = rho*g
        DMStagStencil point[2];
        PetscScalar   fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          fval = c[j][i][idx];
          c[j][i][idx] = g * fval;
        }
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // dVz/dx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVz/dx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVx/dz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVx/dz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // pin pressure in 1 point
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  if (n_bc){
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Stokes_RT_PhaseField
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Stokes_RT_PhaseField"
PetscErrorCode Stokes_RT_PhaseField(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dmPV, dmf;
  Vec            x,f,fprev,dfx,dfz,volf;
  PetscInt       nx, nz, istep;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscLogDouble start_time, end_time;
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL));
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_PhaseField,coeff_description,usr));
  PetscCall(FDPDEGetDM(fd,&dmPV));

  usr->dmPV = dmPV;
  PetscCall(FDPDEGetSolution(fd,&x));
  PetscCall(VecDuplicate(x, &usr->xVel));
  PetscCall(VecDestroy(&x));

  // Create DM/vec for the phase field
  PetscCall(DMStagCreateCompatibleDMStag(dmPV,1,1,1,0,&dmf)); 
  PetscCall(DMSetUp(dmf)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(dmf,xmin,xmax,zmin,zmax,0.0,0.0));

  PetscCall(DMCreateGlobalVector(dmf,&f)); 
  PetscCall(VecDuplicate(f,&fprev)); 
  PetscCall(VecDuplicate(f,&dfx));
  PetscCall(VecDuplicate(f,&dfz));
  PetscCall(VecDuplicate(f,&volf));
  usr->dmf = dmf;
  usr->f = f;
  usr->fprev = fprev;
  usr->dfx = dfx;
  usr->dfz = dfz;
  usr->volf = volf;

  // Initialise the phase field
  PetscCall(SetInitialPhaseField(dmf,f,usr));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase-init",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmf,f,fout));

  // Interpolate phase values on the face and edges before FDPDE solver
  PetscCall(InterpCornerFacePhaseF(dmf,f)); 
  PetscCall(UpdateVolFrac(dmf,f,usr)); 
  PetscCall(VecCopy(f, fprev));

  // Output - initial state of the phase field
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase-0",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmf,f,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase-volf0",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmf,volf,fout));

  // Time loop 
  for (istep=1; istep<usr->par->nt; istep++) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);
    PetscCall(PetscTime(&start_time)); 

    // FD SNES Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x)); 
    PetscCall(VecCopy(x, usr->xVel));

    // Solve phase field
    PetscCall(UpdateDF(dmf,fprev,usr)); 

    // PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase-fprev1",usr->par->fdir_out,usr->par->fname_out));
    // PetscCall(DMStagViewBinaryPython(dmf,fprev,fout));

    PetscCall(ExplicitStep(dmf,fprev,f,usr->par->dt,usr)); 

    { // Runge-Kutta 2
      Vec hk1, hk2, f_bk, fprev_bk;
      // allocate storage for hk1, hk2, f_bk, fprev_bk
      PetscCall(VecDuplicate(f, &hk1)); 
      PetscCall(VecDuplicate(f, &hk2)); 
      PetscCall(VecDuplicate(f, &f_bk)); 
      PetscCall(VecDuplicate(f, &fprev_bk)); 

      // backup x and xprev
      PetscCall(VecCopy(f, f_bk)); 
      PetscCall(VecCopy(fprev, fprev_bk)); 
      
      // 1st stage - get h*k1 = f- fprev
      PetscCall(VecCopy(f, hk1)); 
      PetscCall(VecAXPY(hk1, -1.0, fprev));
      
      // 2nd stage - (t = t+0.5*dt, fprev = fprev + 0.5*hk1)
      PetscCall(VecCopy(fprev_bk, fprev)); 
      PetscCall(VecAXPY(fprev, 0.5, hk1)); 

      // correct time by half step
      // usr->par->t -= 0.5*usr->par->dt;
      
      // update dfx and dfz and solve for the second stage
      PetscCall(UpdateDF(dmf, fprev, usr)); 
      PetscCall(ExplicitStep(dmf, fprev, f, usr->par->dt, usr));

      // get hk2 and update the full step
      PetscCall(VecCopy(f, hk2)); 
      PetscCall(VecAXPY(hk2, -1.0, fprev));
      PetscCall(VecCopy(fprev_bk, fprev)); 
      PetscCall(VecCopy(fprev, f)); 
      PetscCall(VecAXPY(f, 1.0, hk2));

      // reset time
      // usr->par->t += 0.5*usr->par->dt;
      
      // check if hk1 and hk2 are zeros or NANs
      PetscScalar hk1norm, hk2norm;
      PetscCall(VecNorm(hk1, NORM_1, &hk1norm));
      PetscCall(VecNorm(hk2, NORM_1, &hk2norm));
      PetscPrintf(PETSC_COMM_WORLD, "hk1norm=%g, hk2norm=%g \n", hk1norm, hk2norm);
      
      // destroy vectors after use
      PetscCall(VecDestroy(&f_bk));
      PetscCall(VecDestroy(&fprev_bk));
      PetscCall(VecDestroy(&hk1));
      PetscCall(VecDestroy(&hk2));
    }

    PetscCall(InterpCornerFacePhaseF(dmf,f)); 
    PetscCall(UpdateVolFrac(dmf,f,usr)); 
    PetscCall(VecCopy(f,fprev)); 
    
    // Output
    if (istep%usr->par->tout == 0) { 
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase-%d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmf,f,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_phase_xPV_ts%d",usr->par->fdir_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,x,fout));
    }

    PetscCall(VecDestroy(&x));
    PetscCall(PetscTime(&end_time)); 
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n\n", end_time - start_time);
  }

  // Destroy objects
  PetscCall(VecDestroy(&dfx));
  PetscCall(VecDestroy(&dfz));
  PetscCall(VecDestroy(&volf));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&fprev));
  PetscCall(DMDestroy(&dmf)); 
  PetscCall(VecDestroy(&usr->xVel));
  PetscCall(DMDestroy(&dmPV));
  PetscCall(FDPDEDestroy(&fd));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialPhaseField
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPhaseField"
PetscErrorCode SetInitialPhaseField(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xlocal;
  PetscInt       i, j, sx, sz, nx, nz, icenter;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp, fval = 0.0, zinterface, xn;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      zinterface = usr->par->ya * PetscCosReal(PETSC_PI * xp) + usr->par->y0;
      xn = zinterface-zp;
      fval = 0.5*(1 + PetscTanhScalar(xn/2.0/usr->par->eps));
      
      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xx[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_PhaseField
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PhaseField"
PetscErrorCode FormCoefficient_PhaseField(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  Vec            flocal, volflocal;
  PetscScalar    **coordx,**coordz, ***c, F_u, F_d;
  PetscFunctionBeginUser;

  // body force (density) up and down
  F_u = -usr->par->rho0*usr->par->g;
  F_d = -usr->par->rho1*usr->par->g;

  // phase field
  PetscCall(DMGetLocalVector(usr->dmf, &flocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->f, INSERT_VALUES, flocal)); 

  // volume fraction
  PetscCall(DMGetLocalVector(usr->dmf, &volflocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->volf, INSERT_VALUES, volflocal)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   eta, ff, volf;

        // get the phase values in the element
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf)); 
        eta  = usr->par->eta0  * volf + usr->par->eta1  * (1.0 - volf);
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;
        //PetscPrintf(PETSC_COMM_WORLD, "A (center) = %g \n", c[j][i][idx]); 
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   ff[4], volf[4], eta;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // collect phase values for the four corners
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 

        for (ii = 0; ii < 4; ii++) {
          eta  = usr->par->eta0  * volf[ii] + usr->par->eta1  * (1.0 - volf[ii]);
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;
        }
      }

      { // Bx = rho*g (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   rhs[4], ff[4], volf[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // collect phase values for the four edges
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 
          
        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = F_u*volf[2] + F_d*(1.0-volf[2]);
        rhs[3] = F_u*volf[3] + F_d*(1.0-volf[3]);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMRestoreLocalVector(usr->dmf,  &flocal));    
  PetscCall(DMRestoreLocalVector(usr->dmf,  &volflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Interpolate corner and face values of f, for uniform grids only
// ---------------------------------------
PetscErrorCode InterpCornerFacePhaseF(DM dm, Vec x)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       ic, il, ir, iu, id, idl, idr, iul, iur;
  PetscScalar    ***xx;
  Vec            xlocal;
  PetscFunctionBeginUser;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 
  
  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL));  
  
  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT,    0, &ic )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl)); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr)); 
  PetscCall(DMStagGetLocationSlot(dm, LEFT,       0, &il )); 
  PetscCall(DMStagGetLocationSlot(dm, RIGHT,      0, &ir )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN,       0, &id )); 
  PetscCall(DMStagGetLocationSlot(dm, UP,         0, &iu )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul)); 
  PetscCall(DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    fval[4];

      // collect the elements points around the down left corner
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i  ; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i-1; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j;   point[3].loc = ELEMENT; point[3].c = 0;

      // fix the boundary cell
      if (i == 0)    {point[0].i = i; point[2].i = i;}
      if (j == 0)    {point[2].j = j; point[1].j = j;}

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval)); 

      xx[j][i][il]  = 0.5*(fval[3] + fval[0]); // left
      xx[j][i][id]  = 0.5*(fval[3] + fval[1]); // down
      xx[j][i][idl] = 0.25*(fval[0]+fval[1]+fval[2]+fval[3]); // downleft

      if (j==Nz-1) {
        xx[j][i][iu]  = xx[j][i][ic];
        xx[j][i][iul] = xx[j][i][il];
      }
      if (i==Nx-1) {
        xx[j][i][ir]  = xx[j][i][ic];
        xx[j][i][idr]  = xx[j][i][id];
      }
      if (i==Nx-1 && j==Nz-1) {
        xx[j][i][iur] = xx[j][i][ic];
      }
    }
  }

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update volumefraction for fluid 1 within each cube between two vertically adjacent cell center
// ---------------------------------------
PetscErrorCode UpdateVolFrac(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, vfopt;
  PetscInt       ic, il, ir, iu, id, idl, iul, idr, iur, icenter, iprev, inext;
  PetscScalar    ***vvf, **coordx, **coordz;
  PetscScalar    eps;
  Vec            xlocal, vflocal;
  PetscFunctionBeginUser;

  eps = usr->par->eps;
  vfopt = usr->par->vfopt;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMCreateLocalVector(dm, &vflocal)); 
  PetscCall(DMStagVecGetArray(dm, vflocal, &vvf)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL));  
  
  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT,    0, &ic )); 
  PetscCall(DMStagGetLocationSlot(dm, LEFT,       0, &il )); 
  PetscCall(DMStagGetLocationSlot(dm, RIGHT,      0, &ir )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN,       0, &id )); 
  PetscCall(DMStagGetLocationSlot(dm, UP,         0, &iu )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur )); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT   ,&iprev  ));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT  ,&inext  ));
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[8];
      PetscScalar    ff[8], dx, dz, cc, ar;

      // collect the elements points around the down left corner
      point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i; point[2].j = j  ; point[2].loc = LEFT   ; point[2].c = 0;
      point[3].i = i; point[3].j = j-1; point[3].loc = LEFT   ; point[3].c = 0;

      point[4] = point[0]; point[4].loc = DOWN;
      point[5] = point[0]; point[5].loc = DOWN_LEFT;
      point[6] = point[0]; point[6].loc = UP;
      point[7] = point[0]; point[7].loc = UP_LEFT;

      if (j==0) {point[1].j = point[0].j; point[3].j = point[0].j;}

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 8, point, ff)); 

      dz = coordz[j][inext] -  coordz[j  ][iprev];
      dx = coordx[i][inext  ] -  coordx[i  ][iprev  ];

      cc = eps/dz;
      ar = dx/dz;

      //diffuse
      if (vfopt == 0) {
        vvf[j][i][id] = ff[4];
        vvf[j][i][ic] = ff[0];
        vvf[j][i][il] = ff[2];
        vvf[j][i][idl]= ff[5];
      }
      
      //sharp: staggered
      if (vfopt ==1) {
        if (ff[4]>= 0.5) {vvf[j][i][id] = 1.0;}
        else             {vvf[j][i][id] = 0.0;}
        if (ff[0]>= 0.5) {vvf[j][i][ic] = 1.0;}
        else             {vvf[j][i][ic] = 0.0;}
        if (ff[2]>= 0.5) {vvf[j][i][il] = 1.0;}
        else             {vvf[j][i][il] = 0.0;}
        if (ff[5]>= 0.5) {vvf[j][i][idl]= 1.0;}
        else             {vvf[j][i][idl]= 0.0;}
      }

      //1d simplification
      if (vfopt ==2 ) {
        PetscScalar fftmp;
        fftmp = ff[4];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][id] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][id] = 0.0;}
        else    {vvf[j][i][id] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[0];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][ic] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][ic] = 0.0;}
        else    {vvf[j][i][ic] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[2];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][il] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][il] = 0.0;}
        else    {vvf[j][i][il] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[5];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][idl] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][idl] = 0.0;}
        else    {vvf[j][i][idl] = 0.5 - 4.0*cc*(fftmp-0.5);}
      }

      //2d
      if (vfopt ==3) {
        vvf[j][i][ic] = volf_2d(ff[6], ff[4], cc, ar);
        vvf[j][i][il] = volf_2d(ff[7], ff[5], cc, ar);
        if (j>0) {
          vvf[j][i][id] = volf_2d(ff[0], ff[1], cc, ar);
          vvf[j][i][idl]= volf_2d(ff[2], ff[3], cc, ar);
        } else {
          vvf[j][i][id] = vvf[j][i][ic];
          vvf[j][i][idl] = vvf[j][i][il];
        }
      }
    }
  }

  // for nodes on the up and right boundaries
  if (sz+nz == Nz) {
    j = Nz-1;
    for (i = sx; i<sx+nx; i++) {
      vvf[j][i][iul] = vvf[j][i][il];
      vvf[j][i][iu]  = vvf[j][i][ic];
    }
  }
  if (sx+nx == Nx) {
    i = Nx-1;
    for (j = sz; j<sz+nz; j++) {
      vvf[j][i][idr] = vvf[j][i][id];
      vvf[j][i][ir]  = vvf[j][i][ic];
    }
  }
  if (sx+nx==Nx && sz+nz ==Nz) {
    i = Nx-1;
    j = Nz-1;
    vvf[j][i][iur] = vvf[j][i][ir];
  }

  // for nodes on the bottom boundary
  if (sz == 0) {
    j = 0;
    for (i = sx; i<sx+nx; i++) {vvf[j][i][idl] = vvf[j][i][il];}
    if (sx+nx==Nx) {i = Nx-1; vvf[j][i][idr] = vvf[j][i][ir];}
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,vflocal,&vvf)); 
  PetscCall(DMLocalToGlobalBegin(dm,vflocal,INSERT_VALUES,usr->volf)); 
  PetscCall(DMLocalToGlobalEnd  (dm,vflocal,INSERT_VALUES,usr->volf)); 
  PetscCall(VecDestroy(&vflocal)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update dfdx and dfdz
// ---------------------------------------
PetscErrorCode UpdateDF(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       icenter, idx;
  PetscScalar    ***df1, ***df2;
  PetscScalar    **coordx,**coordz;
  Vec            dfxlocal, dfzlocal, xlocal;
  PetscFunctionBeginUser;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMCreateLocalVector(dm, &dfxlocal)); 
  PetscCall(DMStagVecGetArray(dm, dfxlocal, &df1)); 

  PetscCall(DMCreateLocalVector(dm, &dfzlocal)); 
  PetscCall(DMStagVecGetArray(dm, dfzlocal, &df2)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    dx, dz, fval[4];

      // df/dx, df/dz: center
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i+1; point[1].j = j  ; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i  ; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j+1; point[3].loc = ELEMENT; point[3].c = 0;

      if      ((i!=0) && (i!=Nx-1)) { dx = coordx[i+1][icenter] -  coordx[i-1][icenter]; } 
      else if (i == 0) { point[0].i = i; dx = coordx[i+1][icenter] - coordx[i][icenter]; } 
      else if (i == Nx-1) { point[1].i = i; dx = coordx[i][icenter] - coordx[i-1][icenter]; }

      if      ((j!=0) && (j!=Nz-1)) { dz = coordz[j+1][icenter] -  coordz[j-1][icenter]; } 
      else if (j == 0) { point[2].j = j; dz = coordz[j+1][icenter] - coordz[j][icenter]; } 
      else if (j == Nz-1) { point[3].j = j; dz = coordz[j][icenter] - coordz[j-1][icenter]; }

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval)); 

      df1[j][i][idx] = (fval[1] - fval[0])/dx;
      df2[j][i][idx] = (fval[3] - fval[2])/dz;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,dfxlocal,&df1)); 
  PetscCall(DMLocalToGlobalBegin(dm,dfxlocal,INSERT_VALUES,usr->dfx)); 
  PetscCall(DMLocalToGlobalEnd  (dm,dfxlocal,INSERT_VALUES,usr->dfx)); 
  PetscCall(VecDestroy(&dfxlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,dfzlocal,&df2)); 
  PetscCall(DMLocalToGlobalBegin(dm,dfzlocal,INSERT_VALUES,usr->dfz)); 
  PetscCall(DMLocalToGlobalEnd  (dm,dfzlocal,INSERT_VALUES,usr->dfz)); 
  PetscCall(VecDestroy(&dfzlocal)); 

  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
PetscErrorCode ExplicitStep(DM dm, Vec xprev, Vec x, PetscScalar dt, void *ctx)
/* ------------------------------------------------------------------- */
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    gamma, eps;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***xx,***xxp;
  Vec            dfx, dfz, dfxlocal, dfzlocal, xlocal, xplocal;
  Vec            xVellocal;
  PetscFunctionBeginUser;

  // User Parameter
  gamma = usr->par->gamma;
  eps = usr->par->eps;

  dfx = usr->dfx;
  dfz = usr->dfz;

  // create a dmPV and xPV in usrdata, copy data in and extract them here
  PetscCall(DMGetLocalVector(usr->dmPV, &xVellocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPV, usr->xVel, INSERT_VALUES, xVellocal)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  // Get global size
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));

  // Create local vector
  PetscCall(DMCreateLocalVector(dm,&xplocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,xprev,INSERT_VALUES,xplocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,xprev,INSERT_VALUES,xplocal)); 
  PetscCall(DMCreateLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMGetLocalVector(dm,&dfxlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,dfx,INSERT_VALUES,dfxlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,dfx,INSERT_VALUES,dfxlocal)); 
  PetscCall(DMGetLocalVector(dm,&dfzlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,dfz,INSERT_VALUES,dfzlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,dfz,INSERT_VALUES,dfzlocal)); 
  
  // get array from xlocal
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  PetscCall(DMStagVecGetArray(dm, xplocal, &xxp)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 

  // Get the cell sizes
  PetscScalar *dx, *dz;
  PetscCall(DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz)); 

  // loop over local domain and get the RHS value
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i < sx+nx; i++) {

      DMStagStencil point[5];
      PetscInt      ii,ix,iz;
      PetscScalar   fe[5], dfxe[5], dfze[5], gfe[5], c[5], fval = 0.0;

      ix = i - sx;
      iz = j - sz;

      point[0].i = i;   point[0].j = j;   point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i-1; point[1].j = j;   point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i+1; point[2].j = j;   point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i;   point[3].j = j-1; point[3].loc = ELEMENT; point[3].c = 0;
      point[4].i = i;   point[4].j = j+1; point[4].loc = ELEMENT; point[4].c = 0;

      // default zero flux on boundary
      if (i==0)    {point[1] = point[0];}
      if (i==Nx-1) {point[2] = point[0];}
      if (j==0)    {point[3] = point[0];}
      if (j==Nz-1) {point[4] = point[0];}
      
      PetscCall(DMStagVecGetValuesStencil(dm,dfxlocal,5,point,dfxe)); 
      PetscCall(DMStagVecGetValuesStencil(dm,dfzlocal,5,point,dfze)); 
      PetscCall(DMStagVecGetValuesStencil(dm,xplocal ,5,point,fe)); 

      for (ii=1; ii<5; ii++) {
        PetscScalar epsAlt;  //coefficients of anti-diffusion, center

        gfe[ii] = sqrt(dfxe[ii]*dfxe[ii]+dfze[ii]*dfze[ii]);
        if (gfe[ii] > 1e-10) {epsAlt = fe[ii]*(1-fe[ii])/gfe[ii];}
        else {epsAlt = eps;}
        c[ii] = epsAlt; //coefficients at the center
      }

      //diffusion terms
      fval = gamma*(eps * ((fe[2]+fe[1]-2*fe[0])/dx[ix]/dx[ix] + (fe[4]+fe[3]-2*fe[0])/dz[iz]/dz[iz]));

      //sharpen terms
      fval -= gamma* ( (c[2]*dfxe[2] - c[1]*dfxe[1])/(2.0*dx[ix]) + (c[4]*dfze[4]-c[3]*dfze[3])/(2.0*dz[iz]));

      { // velocity on the face and advection terms
        DMStagStencil pf[4];
        PetscScalar vf[4];

        pf[0].i = i; pf[0].j = j; pf[0].loc = LEFT;  pf[0].c = 0; 
        pf[1].i = i; pf[1].j = j; pf[1].loc = RIGHT; pf[1].c = 0;
        pf[2].i = i; pf[2].j = j; pf[2].loc = DOWN;  pf[2].c = 0;
        pf[3].i = i; pf[3].j = j; pf[3].loc = UP;    pf[3].c = 0;

        PetscCall(DMStagVecGetValuesStencil(usr->dmPV,xVellocal,4,pf,vf)); 

        // central difference method
        fval -= 0.5*(vf[1]*(fe[2]+fe[0]) - vf[0]*(fe[1]+fe[0]))/dx[ix] + 0.5*(vf[3]*(fe[4]+fe[0]) - vf[2]*(fe[3]+fe[0]))/dz[iz];
      }

      xx[j][i][idx] = xxp[j][i][idx] + dt*fval;
    }
  }

  // release dx dz
  PetscCall(PetscFree(dx));
  PetscCall(PetscFree(dz));

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,xplocal,&xxp));
  PetscCall(DMLocalToGlobalBegin(dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(VecDestroy(&xplocal)); 

  PetscCall(DMRestoreLocalVector(dm, &dfxlocal)); 
  PetscCall(DMRestoreLocalVector(dm, &dfzlocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmPV, &xVellocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// InputParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputParameters"
PetscErrorCode InputParameters(UsrData **_usr)
{
  UsrData       *usr;
  Params        *par;
  PetscBag       bag;
  PetscFunctionBeginUser;

  // Allocate memory to application context
  PetscCall(PetscMalloc1(1, &usr)); 

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank)); 

  // Create bag
  PetscCall(PetscBagCreate (usr->comm,sizeof(Params),&usr->bag)); 
  PetscCall(PetscBagGetData(usr->bag,(void **)&usr->par)); 
  PetscCall(PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -")); 

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nt, 5, "nt", "Number of time steps")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout, 10, "tout", "Output # time steps")); 
  PetscCall(PetscBagRegisterInt(bag, &par->ppcell, 4, "ppcell", "Number of particles/cell one-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho0, 1.2, "rho0", "Density rho0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho1, 1.0, "rho1", "Density rho1")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dt, 20.0, "dt", "Time step size")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->ya, 0.1, "ya", "Amplitude initial perturbation")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->y0, 0.3, "y0", "Deviation initial perturbation")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "Epsilon in the kernel function for phase-field")); 
  PetscCall(PetscBagRegisterInt(bag, &par->vfopt, 0, "vfopt", "Sharp or diffuse boundary vfopt = 0,1,2,3")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "Parameter gamma in the phase field method")); 

  PetscCall(PetscBagRegisterInt(bag, &par->method, 0, "method", "0-PIC, 1-PhaseField")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // Other variables
  par->fname_in[0] = '\0';

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscFunctionBeginUser;

  // Get date
  PetscCall(PetscGetDate(date,30)); 

  // Get petsc command options
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_stokes_rt_compare_pic_phasefield: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscPrintf(usr->comm,"# Input options file: NONE \n");
  }
  else {
    PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in);
  }
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  PetscLogDouble  start_time, end_time;
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  // Start time
  PetscCall(PetscTime(&start_time)); 
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution - PIC
  if (usr->par->method==0) { PetscCall(Stokes_RT_PIC(usr));  }

  // Numerical solution - PhaseField
  if (usr->par->method==1) { PetscCall(Stokes_RT_PhaseField(usr)); }

  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

  // End time
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}