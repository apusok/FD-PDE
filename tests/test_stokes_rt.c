// ---------------------------------------
// run: ./test_stokes_rt_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -snes_type ksponly -snes_fd_color -nt 800 -log_view
// python test (need viz): ./python/test_stokes_rt.py
// Visualize: with ParaView to open xmf files
// ---------------------------------------
static char help[] = "Application to solve an Rayleigh-Taylor instability\n\n";

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

#include "../src/fdpde_stokes.h"
#include "../src/fdpde_dmswarm.h"

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
  char           fdir_out[FNAME_LENGTH]; 
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
"  fuz = rho(x,y)*g \n"
"  fp = 0\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0\n"
"  RIGHT: Vx = 0, dVz/dx = 0\n" 
"  DOWN: Vx = Vz = 0,\n"
"  UP: Vx = Vz = 0, \n";

// ---------------------------------------
// Application functions
// ---------------------------------------


static PetscErrorCode DumpSolution(DM dmStokes,Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmVelAvg;
  Vec            velAvg;
  DM             daVelAvg;
  Vec            vecVelAvg;
  char           fout[FNAME_LENGTH];
  
  PetscFunctionBeginUser;
  
  /* For convenience, create a new DM and Vec which will hold averaged velocities
   Note that this could also be accomplished with direct array access, using
   DMStagVecGetArray() and related functions */
  PetscCall(DMStagCreateCompatibleDMStag(dmStokes,0,0,2,0,&dmVelAvg)); /* 2 dof per element */
  PetscCall(DMSetUp(dmVelAvg));
  PetscCall(DMStagSetUniformCoordinatesProduct(dmVelAvg,0.0,1.0,0.0,1.0,0.0,0.0));
  PetscCall(DMCreateGlobalVector(dmVelAvg,&velAvg));
  {
    PetscInt ex,ey,startx,starty,nx,ny;
    Vec      stokesLocal;
    PetscCall(DMGetLocalVector(dmStokes,&stokesLocal));
    PetscCall(DMGlobalToLocal(dmStokes,x,INSERT_VALUES,stokesLocal));
    PetscCall(DMStagGetCorners(dmVelAvg,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL));
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];
        from[0].i = ex; from[0].j = ey; from[0].loc = UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = RIGHT; from[3].c = 0;
        PetscCall(DMStagVecGetValuesStencil(dmStokes,stokesLocal,4,from,valFrom));
        to[0].i = ex; to[0].j = ey; to[0].loc = ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        PetscCall(DMStagVecSetValuesStencil(dmVelAvg,velAvg,2,to,valTo,INSERT_VALUES));
      }
    }
    PetscCall(VecAssemblyBegin(velAvg));
    PetscCall(VecAssemblyEnd(velAvg));
    PetscCall(DMRestoreLocalVector(dmStokes,&stokesLocal));
  }
  
  /* Create individual DMDAs for sub-grids of our DMStag objects. This is
   somewhat inefficient, but allows use of the DMDA API without re-implementing
   all utilities for DMStag */
  
  
  PetscCall(DMStagVecSplitToDMDA(dmVelAvg,    velAvg,    DMSTAG_ELEMENT,  -3,&daVelAvg,    &vecVelAvg)); /* note -3 : pad with zero */
  PetscCall(PetscObjectSetName((PetscObject)vecVelAvg,"Velocity (Averaged)"));
  
  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_stokes_rt_element.vtr"));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVelAvg),fout,FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vecVelAvg,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  
  /* Dump vertex-based fields to a second .vtr file */
  
  /* Edge-based fields could similarly be dumped */
  
  /* Destroy DMDAs and Vecs */
  PetscCall(VecDestroy(&velAvg));
  PetscCall(VecDestroy(&vecVelAvg));
  PetscCall(DMDestroy(&daVelAvg));
  PetscCall(DMDestroy(&dmVelAvg));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  SNES           snes;
  char           fout[FNAME_LENGTH];
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
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  PetscCall(FDPDEGetDM(fd,&dmPV));
  
  /* Create a swarm object, assign several fields */
  PetscCall(DMStagPICCreateDMSwarm(dmPV,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"rho",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));
  usr->swarm = dmswarm;

  PetscCall(DMDestroy(&dmPV));

  /* swarm coordinate layout */
  {
    PetscInt ppcell[] = {4,4};
    
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.5,ppcell,COOR_INITIALIZE));
  }
  
  /* swarm initial condition */
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
    PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    
    PetscCall(DMSwarmGetField(dmswarm,"eta",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      PetscReal yinterface,xcoor,ycoor;
      
      xcoor = pcoor[2*p+0];
      ycoor = pcoor[2*p+1];
      yinterface = 0.1 * PetscCosReal(PETSC_PI * xcoor) + 0.3;
      pfield[p] = usr->par->eta0;
      if (ycoor < yinterface) {
        pfield[p] = usr->par->eta1;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarm,"eta",NULL,NULL,(void**)&pfield));
    
    PetscCall(DMSwarmGetField(dmswarm,"rho",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      PetscReal yinterface,xcoor,ycoor;
      
      xcoor = pcoor[2*p+0];
      ycoor = pcoor[2*p+1];
      yinterface = 0.1 * PetscCosReal(PETSC_PI * xcoor) + 0.3;
      pfield[p] = 1.2;
      if (ycoor < yinterface) {
        pfield[p] = 1.0;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarm,"rho",NULL,NULL,(void**)&pfield));
    PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  }
  
  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL));

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr));

  PetscCall(FDPDEGetSNES(fd,&snes));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s","out_stokes_rt-0.xmf"));
  PetscCall(DMSwarmViewXDMF(dmswarm,fout));

  for (k=1; k<usr->par->nt; k++) {
    char filename[100];
    
    PetscPrintf(PETSC_COMM_SELF,"====== step %d =======\n",k);
    // FD SNES Solver
    PetscCall(FDPDESolve(fd,NULL));

    // Get solution vector
    PetscCall(FDPDEGetSolution(fd,&x)); 
    PetscCall(FDPDEGetDM(fd,&dmPV));
    
    PetscCall(MPoint_AdvectRK1(dmswarm,dmPV,x,20.0));

    PetscCall(DumpSolution(dmPV,x,usr));
    PetscCall(PetscSNPrintf(filename,99,"out_stokes_rt-%d.xmf",k));
    if (k%10 == 0) { PetscCall(DMSwarmViewXDMF(dmswarm,filename)); }

    PetscCall(VecDestroy(&x));
    PetscCall(DMDestroy(&dmPV));
  }
  
  PetscCall(FDPDEGetSolution(fd,&x));
  PetscCall(FDPDEGetDM(fd,&dmPV));

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmPV,x,fout));
  {
    DM dmcoeff;
    Vec coeff;
    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&coeff));
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_coefficients"));
    PetscCall(DMStagViewBinaryPython(dmcoeff,coeff,fout));
  }

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));
  PetscCall(DMDestroy(&dmswarm));

  *_x  = x;
  *_dm = dmPV;

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

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution","output_file","Name for output file, set with: -output_file <filename>")); 
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
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    g;
  DM             dmswarm;
  PetscFunctionBeginUser;

  dmswarm = usr->swarm;
  
  //PetscCall(MPoint_ProjectP0_arith(dmswarm,"eta",dm,dmcoeff,0,coeff));
  //PetscCall(MPoint_ProjectP0_arith(dmswarm,"rho",dm,dmcoeff,1,coeff));

  // PetscCall(VecZeroEntries(coeff));
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"eta",dm,dmcoeff,2,1,coeff));//cell
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"eta",dm,dmcoeff,0,0,coeff));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"rho",dm,dmcoeff,1,0,coeff));//face
  
  // Density is defined on the edges
  // Viscosity is defined (vertices and cell center
  // DM dm, Vec x - used for non-linear coefficients

  // User parameters
  g = -usr->par->g;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
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

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }

      // Above we project rho into the slot associated with fuz. Hence we just need to scale it by g to define fuz //
      { // fuz = rho*g
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

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      /* // Don't require this as a projected value for eta on the cell center has already been performed //
      { // eta_c = eta0:eta1
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = fval;
      }
      */
       
      /* // Don't require this as a projected value for eta on the cell nodes has already been performed //
      { // eta_n = eta0:eta1
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
        }
      }
      */
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

  // pin P = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  if (n_bc){
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

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
  DM              dmStokes;
  Vec             xStokes;
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

  // Numerical solution using the FD pde object
  PetscCall(SNESStokes_RT(&dmStokes, &xStokes, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dmStokes)); 
  PetscCall(VecDestroy(&xStokes)); 

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
