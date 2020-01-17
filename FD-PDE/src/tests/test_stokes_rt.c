// ---------------------------------------
// run: ./tests/test_stokes_rt.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
// ---------------------------------------
static char help[] = "Application to solve an RT instability\n\n";

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

#include "petsc.h"
#include "../fdpde_stokes.h"
#include "../dmstagoutput.h"
#include "../material_point.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, nt;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    eta0, eta1, g;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             swarm;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode SNESStokes_RT(DM*,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = f(x,y)\n"
"  fux = 0 \n" 
"  fuz = rho*g \n" 
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0\n"
"  RIGHT: Vx = 0, dVz/dx = 0\n" 
"  DOWN: Vx = Vz = 0,\n"
"  UP: Vx = Vz = 0, \n";

// ---------------------------------------
// Application functions
// ---------------------------------------


static PetscErrorCode DumpSolution(DM dmStokes,Vec x)
{
  PetscErrorCode ierr;
  DM             dmVelAvg;
  Vec            velAvg;
  DM             daVelAvg;
  Vec            vecVelAvg;
  
  PetscFunctionBeginUser;
  
  /* For convenience, create a new DM and Vec which will hold averaged velocities
   Note that this could also be accomplished with direct array access, using
   DMStagVecGetArray() and related functions */
  ierr = DMStagCreateCompatibleDMStag(dmStokes,0,0,2,0,&dmVelAvg);CHKERRQ(ierr); /* 2 dof per element */
  ierr = DMSetUp(dmVelAvg);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesExplicit(dmVelAvg,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmVelAvg,&velAvg);CHKERRQ(ierr);
  {
    PetscInt ex,ey,startx,starty,nx,ny;
    Vec      stokesLocal;
    ierr = DMGetLocalVector(dmStokes,&stokesLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dmStokes,x,INSERT_VALUES,stokesLocal);CHKERRQ(ierr);
    ierr = DMStagGetCorners(dmVelAvg,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];
        from[0].i = ex; from[0].j = ey; from[0].loc = UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = RIGHT; from[3].c = 0;
        ierr = DMStagVecGetValuesStencil(dmStokes,stokesLocal,4,from,valFrom);CHKERRQ(ierr);
        to[0].i = ex; to[0].j = ey; to[0].loc = ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        ierr = DMStagVecSetValuesStencil(dmVelAvg,velAvg,2,to,valTo,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(velAvg);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(velAvg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmStokes,&stokesLocal);CHKERRQ(ierr);
  }
  
  /* Create individual DMDAs for sub-grids of our DMStag objects. This is
   somewhat inefficient, but allows use of the DMDA API without re-implementing
   all utilities for DMStag */
  
  
  ierr = DMStagVecSplitToDMDA(dmVelAvg,    velAvg,    DMSTAG_ELEMENT,  -3,&daVelAvg,    &vecVelAvg);CHKERRQ(ierr); /* note -3 : pad with zero */
  ierr = PetscObjectSetName((PetscObject)vecVelAvg,"Velocity (Averaged)");CHKERRQ(ierr);
  
  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVelAvg),"rt_element.vtr",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vecVelAvg,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  /* Dump vertex-based fields to a second .vtr file */
  
  /* Edge-based fields could similarly be dumped */
  
  /* Destroy DMDAs and Vecs */
  ierr = VecDestroy(&velAvg);CHKERRQ(ierr);
  ierr = VecDestroy(&vecVelAvg);CHKERRQ(ierr);
  ierr = DMDestroy(&daVelAvg);CHKERRQ(ierr);
  ierr = DMDestroy(&dmVelAvg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESStokes_RT"
PetscErrorCode SNESStokes_RT(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dmPV,dmswarm;
  Vec            x;
  PetscInt       nx, nz, k;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  ierr = FDPDEGetDM(fd, &dmPV); CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dmPV,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"rho",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);
  usr->swarm = dmswarm;

  ierr = DMDestroy(&dmPV);CHKERRQ(ierr);

  /* swarm coordinate layout */
  {
    PetscInt ppcell[] = {4,4};
    
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.0,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
  }
  
  /* swarm initial condition */
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    
    ierr = DMSwarmGetField(dmswarm,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      pfield[p] = 1.0;
    }
    ierr = DMSwarmRestoreField(dmswarm,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    
    ierr = DMSwarmGetField(dmswarm,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      PetscReal yinterface,xcoor,ycoor;
      
      xcoor = pcoor[2*p+0];
      ycoor = pcoor[2*p+1];
      yinterface = 0.1 * cos(M_PI * xcoor) + 0.3;
      pfield[p] = 1.2;
      if (ycoor < yinterface) {
        pfield[p] = 1.0;
      }
    }
    ierr = DMSwarmRestoreField(dmswarm,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  }
  
  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);

  // Some SNES options
  PetscInt  maxit, maxf, its;
  PetscReal atol, rtol, stol;
  SNES      snes;

  ierr = FDPDEGetSNES(fd,&snes);CHKERRQ(ierr);

  // SNES Options - default info on convergence
  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",         ""); CHKERRQ(ierr);
  //ierr = SNESSetOptionsPrefix(snes,"stk_");CHKERRQ(ierr);

  ierr = DMSwarmViewXDMF(dmswarm,"dms-0.xmf");CHKERRQ(ierr);

  for (k=1; k<usr->par->nt; k++) {
    char filename[100];
    
    printf("====== step %d =======\n",k);
    // FD SNES Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);

    ierr = SNESGetIterationNumber(fd->snes,&its);    CHKERRQ(ierr);
    ierr = SNESGetTolerances(fd->snes, &atol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
    
    // Get solution vector
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
    ierr = FDPDEGetDM(fd, &dmPV); CHKERRQ(ierr);
    
    ierr = MPoint_AdvectRK1(dmswarm,dmPV,x,20.0);CHKERRQ(ierr);

    ierr = DumpSolution(dmPV,x);CHKERRQ(ierr);
    PetscSNPrintf(filename,99,"dms-%d.xmf",k);
    if (k%10 == 0) { ierr = DMSwarmViewXDMF(dmswarm,filename);CHKERRQ(ierr); }

    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = DMDestroy(&dmPV);CHKERRQ(ierr);
  }
  
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd, &dmPV); CHKERRQ(ierr);

  
  // Output solution to file
  ierr = DMStagViewBinaryPython(dmPV,x,usr->par->fname_out);CHKERRQ(ierr);
  {
    DM dmcoeff;
    Vec coeff;
    ierr = FDPDEGetCoefficient(fd,&dmcoeff,&coeff);CHKERRQ(ierr);
    ierr = DMStagViewBinaryPython(dmcoeff,coeff,"out_coefficients");CHKERRQ(ierr);
  }

  // Destroy FD-PDE object
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarm);CHKERRQ(ierr);

  *_x  = x;
  *_dm = dmPV;

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory to application context
  ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank); CHKERRQ(ierr);

  // Create bag
  ierr = PetscBagCreate (usr->comm,sizeof(Params),&usr->bag); CHKERRQ(ierr);
  ierr = PetscBagGetData(usr->bag,(void **)&usr->par); CHKERRQ(ierr);
  ierr = PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -"); CHKERRQ(ierr);

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  ierr = PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nt, 5, "nt", "Number of time steps"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_stokes_rt: %s \n",&(date[0]));
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
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscScalar    g, L2 = 0.5;
  DM             dmswarm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  dmswarm = usr->swarm;
  
  //ierr = MPoint_ProjectP0_arith(dmswarm,"eta",dm,dmcoeff,0,coeff);CHKERRQ(ierr);
  //ierr = MPoint_ProjectP0_arith(dmswarm,"rho",dm,dmcoeff,1,coeff);CHKERRQ(ierr);

  VecZeroEntries(coeff);
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"eta",dm,dmcoeff,2,1,coeff);CHKERRQ(ierr);//cell
  //VecView(coeff,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"eta",dm,dmcoeff,0,0,coeff);CHKERRQ(ierr);//vertex

  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"rho",dm,dmcoeff,1,0,coeff);CHKERRQ(ierr);//face
  //VecView(coeff,PETSC_VIEWER_STDOUT_WORLD);
  //exit(0);
  
  // Density is defined (edges): sin(pi*z)*cos(pi*x); 
  // Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
  // DM dm, Vec x - used for non-linear coefficients

  // User parameters
  g = -usr->par->g;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dmcoeff,coeff,INSERT_VALUES,coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dmcoeff,coeff,INSERT_VALUES,coefflocal); CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

#if 1
      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }

      { // fuz = rho*g
        DMStagStencil point[2];
        PetscScalar   fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          fval = c[j][i][idx];
          c[j][i][idx] = g * fval;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // eta_c = eta0:eta1
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        //c[j][i][idx] = fval;
      }

      { // eta_n = eta0:eta1
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          //c[j][i][idx] = fval;
        }
      }
#endif
      
#if 0
      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;
        
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        
        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }
      
      { // fuz = rho*g
        DMStagStencil point[2];
        PetscScalar   xp[2], zp[2], fval = 0.0;
        PetscInt      ii, idx;
        
        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;
        
        xp[0] = coordx[i][icenter]; zp[0] = coordz[j][iprev  ];
        xp[1] = coordx[i][icenter]; zp[1] = coordz[j][inext  ];
        
        for (ii = 0; ii < 2; ii++) {
          fval = g*PetscSinScalar(PETSC_PI*zp[ii]) * PetscCosScalar(PETSC_PI*xp[ii]);
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
        }
      }
      
      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }
      
      { // eta_c = eta0:eta1
        DMStagStencil point;
        PetscScalar   xp, fval = 0.0;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter];
        
        if (xp <= L2) fval = usr->par->eta0;
        else          fval = usr->par->eta1;
        
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = fval;
      }
      
      { // eta_n = eta0:eta1
        DMStagStencil point[4];
        PetscScalar   xp[4], fval = 0.0;
        PetscInt      ii, idx;
        
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        xp[0] = coordx[i][iprev];
        xp[1] = coordx[i][inext];
        xp[2] = coordx[i][iprev];
        xp[3] = coordx[i][inext];
        
        for (ii = 0; ii < 4; ii++) {
          if (xp[ii] <= L2) fval = usr->par->eta0;
          else              fval = usr->par->eta1;
          
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
        }
      }
#endif
      
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  // dVz/dx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVz/dx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  DM              dmStokes;
  Vec             xStokes;
  PetscLogDouble  start_time, end_time;
  PetscErrorCode  ierr;
    
  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  // Start time
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = SNESStokes_RT(&dmStokes, &xStokes, usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = DMDestroy(&dmStokes); CHKERRQ(ierr);
  ierr = VecDestroy(&xStokes); CHKERRQ(ierr);

  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
