#include "petsc.h"
#include "prealloc_helper.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
// general
#define FNAME_LENGTH  200

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

// define grid location names 
enum LocationType {
  CENTER,  // center (for effective viscosity)
  CORNER,  // corner (for effective viscosity)
  BCLEFT,  // left   (for BC)
  BCRIGHT, // right  (for BC)
  BCUP,    // up     (for BC)
  BCDOWN,  // down   (for BC)
  NONE
};

// define boundary conditions type
enum BCType {
  FREE_SLIP, 
  NO_SLIP
};

// define names for standard models
enum ModelType {
  SOLCX, 
  MOR
};

// ---------------------------------------
// Application Context
// ---------------------------------------
// user defined and model-dependent variables
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    rho1, rho2, eta0, ndisl;
  PetscScalar    solcx_eta0, solcx_eta1;
  PetscScalar    g;
  PetscInt       bcleft, bcright, bcup, bcdown;
  PetscInt       mtype, tests;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} UsrData;

// grid variables
typedef struct {
  // stencils: dof0 per vertex, dof1 per edge, dof1 per face/element
  PetscInt dofPV0, dofPV1, dofPV2;
  PetscInt dofCf0, dofCf1, dofCf2;
  PetscInt stencilWidth;

  // domain parameters
  PetscInt       nx, nz;
  PetscScalar    dx, dz;
  PetscScalar    xmin, zmin, xmax, zmax;
  enum BCType    bcleft, bcright, bcup, bcdown;
  enum ModelType mtype;
  PetscScalar    Vleft, Vright, Vup, Vdown;
  PetscInt       dofV, dofP;
} GridData;

// solver variables
typedef struct {
  MPI_Comm     comm;
  PetscMPIInt  rank;
  PetscBag     bag;
  UsrData      *usr;
  GridData     *grd;
  DM           dmPV, dmCoeff;
  Vec          coeff;
  Vec          r, x, xguess;
  Mat          J;
  PetscScalar  Pdiff, Pdisl;
} SolverCtx;

// ---------------------------------------
// Function Definitions
// ---------------------------------------
// input
PetscErrorCode InputParameters(SolverCtx**);
PetscErrorCode InputPrintData (SolverCtx*);

// initialize model
PetscErrorCode InitializeModel(SolverCtx*);
PetscErrorCode InitializeModel_SolCx(SolverCtx*);

// solver
PetscErrorCode CreateSystem(SolverCtx*);
PetscErrorCode JacobianMatrixPreallocation(SolverCtx*);
PetscErrorCode FormInitialGuess(SolverCtx*);
PetscErrorCode SolveSystem(SNES, SolverCtx*);

// residual calculations
PetscErrorCode FormFunctionPV(SNES, Vec, Vec, void*); // global to local

// physics - governing equations
PetscErrorCode XMomentumResidual(SolverCtx*, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode ZMomentumResidual(SolverCtx*, Vec, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode XMomentumStencil(PetscInt, PetscInt, PetscInt, PetscInt, DMStagStencil*);
PetscErrorCode ZMomentumStencil(PetscInt, PetscInt, PetscInt, PetscInt, DMStagStencil*);

// constitutive equations
PetscErrorCode CalcEffViscosity(SolverCtx*, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode CalcEffViscosity_SolCx(SolverCtx*, PetscInt, PetscInt, enum LocationType, PetscScalar*);

// output
PetscErrorCode DoOutput(SolverCtx*);

// benchmarks
PetscErrorCode DoBenchmarks(SolverCtx*);
PetscErrorCode CreateSolCx(SolverCtx*,DM*,Vec*);
PetscErrorCode CalculateErrorNorms(SolverCtx*,DM,Vec);
PetscErrorCode DoOutput_SolCx(DM,Vec);

// utils
PetscErrorCode StrCreateConcatenate(const char[], const char[], char**);
PetscErrorCode GetCoordinateStencilPoint(DM, Vec, DMStagStencil, PetscScalar[]);