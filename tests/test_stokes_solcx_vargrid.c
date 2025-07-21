// ---------------------------------------
// SOLCX benchmark - Irregular grid spacing
// run: ./test_stokes_solcx_vargrid.sh -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// viz: with Paraview (example VTK output)
// ---------------------------------------
static char help[] = "Application to solve the SolCx benchmark with FD-PDE - Irregular grid spacing \n\n";

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

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
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
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode SNESStokes_Solcx(DM*,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode DoOutput(DM,Vec,const char[]);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta0 if x<0.5, eta1 if x>=0.5\n"
"  fux = 0 \n" 
"  fuz = rho*g \n" 
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0\n"
"  RIGHT: Vx = 0, dVz/dx = 0\n" 
"  DOWN: Vz = 0, dVx/dz = 0\n" 
"  UP: Vz = 0, dVx/dz = 0 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SNESStokes_SolCx"
PetscErrorCode SNESStokes_Solcx(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dmPV;
  Vec            x;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscScalar    dx, dz, xmin, zmin, xmax, zmax;
  PetscInt       iprev, inext, icenter;
  PetscScalar    **coordx,**coordz;
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

  // ---------------------------------------
  // Modify coordinates for irregular grid spacing
  PetscCall(DMStagGetCorners(fd->dmstag, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(FDPDEGetCoordinatesArrayDMStag(fd,&coordx,&coordz));

  PetscCall(DMStagGetProductCoordinateLocationSlot(fd->dmstag,DMSTAG_ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(fd->dmstag,DMSTAG_LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(fd->dmstag,DMSTAG_RIGHT,&inext)); 

  dx = usr->par->L/nx/1.2;
  dz = usr->par->H/nz/1.2;

  for (i = sx; i<sx+nx-1; i++) {
    coordx[i  ][inext  ] -= dx/(i+1);
    coordx[i  ][icenter]  = (coordx[i][iprev]+coordx[i][inext])*0.5;
  }

  for (j = sz; j<sz+nz-1; j++) {
    coordz[j  ][inext  ] -= dz/(j+1);
    coordz[j  ][icenter]  = (coordz[j][iprev]+coordz[j][inext])*0.5;
  }

  PetscCall(FDPDERestoreCoordinatesArrayDMStag(fd,coordx,coordz));
  // ---------------------------------------

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd,&dmPV)); 

  // Output solution to file
  PetscCall(DoOutput(dmPV,x,"out_stokes_solcx_vargrid.vtr"));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

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

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"output","output_file","Name for output file, set with: -output_file <filename>")); 

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
  PetscPrintf(usr->comm,"# Test_stokes_solcx_vargrid: %s \n",&(date[0]));
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
  PetscScalar    g, L2 = 0.5;
  PetscFunctionBeginUser;

  // Density is defined (edges): sin(pi*z)*cos(pi*x); 
  // Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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

          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
        }
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
  //DMStagBC       *list;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  //list = bclist->bc_f;
  
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
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVx/dz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DoOutput
// ---------------------------------------
PetscErrorCode DoOutput(DM dm,Vec x,const char fname[])
{
  DM             dmVel,  daVel, daP;
  Vec            vecVel, vaVel, vecP;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xmin, xmax, zmin, zmax;
  PetscScalar    **coordx,**coordz,**cx, **cz;
  PetscInt       iprev=-1,inext=-1,icenter=-1,icenterc=-1;
  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  PetscCall(DMStagCreateCompatibleDMStag(dm,0,0,2,0,&dmVel)); 
  PetscCall(DMSetUp(dmVel)); 

  // Get corners
  PetscCall(DMStagGetCorners(dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL)); 

  // Set coordinates
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cx,&cz,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));

  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter));

  xmin = cx[0   ][iprev];
  xmax = cx[Nx-1][inext];
  zmin = cz[0   ][iprev];
  zmax = cz[Nz-1][inext];

  PetscCall(DMStagSetUniformCoordinatesProduct(dmVel,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmVel,DMSTAG_ELEMENT,&icenterc));
  PetscCall(DMStagGetProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL));

  for (i = sx; i<sx+nx; i++) {
    coordx[i][icenterc] = cx[i][icenter];
  }

  for (j = sz; j<sz+nz; j++) {
    coordz[j][icenterc] = cz[j][icenter];
  }

  // Restore coordinates
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cx,&cz,NULL));

  // Create global vectors
  PetscCall(DMCreateGlobalVector(dmVel,&vecVel)); 
  
  // Loop over elements
  {
    Vec          xlocal;
    
    // Access local vector
    PetscCall(DMGetLocalVector(dm,&xlocal)); 
    PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

    // Loop
    for (j = sz; j < sz+nz; j++) {
      for (i = sx; i < sx+nx; i++) {
        DMStagStencil from[4], to[2];
        PetscScalar   valFrom[4], valTo[2];
        
        from[0].i = i; from[0].j = j; from[0].loc = UP;    from[0].c = 0;
        from[1].i = i; from[1].j = j; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = i; from[2].j = j; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = i; from[3].j = j; from[3].loc = RIGHT; from[3].c = 0;
        
        // Get values from stencil locations
        PetscCall(DMStagVecGetValuesStencil(dm,xlocal,4,from,valFrom)); 
        
        // Average edge values to obtain ELEMENT values
        to[0].i = i; to[0].j = j; to[0].loc = ELEMENT; to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = i; to[1].j = j; to[1].loc = ELEMENT; to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        
        // Return values in new dm - averaged velocities
        PetscCall(DMStagVecSetValuesStencil(dmVel,vecVel,2,to,valTo,INSERT_VALUES)); 
      }
    }
    
    // Vector assembly
    PetscCall(VecAssemblyBegin(vecVel)); 
    PetscCall(VecAssemblyEnd  (vecVel)); 
    
    // Restore vector
    PetscCall(DMRestoreLocalVector(dm, &xlocal)); 
  }

  // Create individual DMDAs for sub-grids of our DMStag objects
  PetscCall(DMStagVecSplitToDMDA(dm,x,ELEMENT,0,&daP,&vecP)); 
  PetscCall(PetscObjectSetName  ((PetscObject)vecP,"Pressure"));         
  
  PetscCall(DMStagVecSplitToDMDA(dmVel,vecVel,ELEMENT,-3,&daVel,&vaVel));  // note -3 : output 2 DOFs
  PetscCall(PetscObjectSetName  ((PetscObject)vaVel,"Velocity"));          

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;

    // Warning: is being output as Point Data instead of Cell Data - the grid is shifted to be in the center points.
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),fname,FILE_MODE_WRITE,&viewer)); 
    PetscCall(VecView(vaVel,    viewer)); 
    PetscCall(VecView(vecP,     viewer)); 
    
    // Free memory
    PetscCall(PetscViewerDestroy(&viewer)); 
  }

  // Destroy DMDAs and Vecs
  PetscCall(VecDestroy(&vecVel)); 
  PetscCall(VecDestroy(&vaVel )); 
  PetscCall(VecDestroy(&vecP  )); 
  
  PetscCall(DMDestroy(&dmVel  )); 
  PetscCall(DMDestroy(&daVel  )); 
  PetscCall(DMDestroy(&daP    )); 

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
  Vec             xStokes; //,xAnalytic;
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
  PetscCall(SNESStokes_Solcx(&dmStokes, &xStokes, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dmStokes)); 
  PetscCall(VecDestroy(&xStokes)); 
  // PetscCall(VecDestroy(&xAnalytic)); 

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
