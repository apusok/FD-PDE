static char help[] = "Solves for 2D variable viscosity Stokes equations - based on dmstag/ex4.c.\n";

/* Goal: create a single-phase MOR model */

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>
#include <petscdmda.h>

/* Define convenient names for DMStagStencilLocation entries */
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

/* Define application context */
typedef struct {
  MPI_Comm     comm;
  DM           dmStokes, dmCoeff;
  Vec          coeff;
  PetscInt     nx, ny;
  PetscScalar  xmax, ymax, xmin, ymin;
  PetscScalar  hxCharacteristic,hyCharacteristic,etaCharacteristic;
  PetscScalar  eta1, eta2, rho1, rho2, gy;
  PetscScalar  Kbound, Kcont;
} CtxData;
typedef CtxData* Ctx;

/* Definition of helper functions */
static PetscErrorCode PopulateCoefficientData(Ctx);
static PetscErrorCode CreateSystem           (const Ctx, Mat*,Vec*);
static PetscErrorCode DumpSolution           (Ctx, Vec);

/* Extract coefficients functions */
static PetscScalar getRho(Ctx ctx, PetscScalar x) { return PetscRealPart(x) < (ctx->xmax-ctx->xmin)/2.0 ? ctx->rho1 : ctx->rho2; }
static PetscScalar getEta(Ctx ctx, PetscScalar x) { return PetscRealPart(x) < (ctx->xmax-ctx->xmin)/2.0 ? ctx->eta1 : ctx->eta2; }

/* --------------------------------------- */
/* Main function                           */
/* --------------------------------------- */
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  Ctx             ctx;
  Mat             A;
  Vec             x, b;
  KSP             ksp;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Allocate memory to application context */
  ierr = PetscMalloc1(1, &ctx); CHKERRQ(ierr);
  
  /* Initialize default variables */
  ctx->comm = PETSC_COMM_WORLD;

  ctx->nx   = 21;   // Element count X-dir
  ctx->ny   = 21;   // Element count Y-dir
  ctx->xmin = 0.0;  // x-min coordinate
  ctx->xmax = 1.0;  // x-max coordinate
  ctx->ymin = 0.0;  // y-min coordinate
  ctx->ymax = 1.0;  // y-max coordinate
  ctx->rho1 = 2;    // density 1
  ctx->rho2 = 1;    // density 2
  ctx->eta1 = 1;    // viscosity 1
  ctx->eta2 = 1;    // viscosity 2
  ctx->gy   = 1.0;  // gravitational acceleration (non-dimensional)
  
  /* Read variables from command line (or from file) */
  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &ctx->nx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-ny", &ctx->nx, NULL); CHKERRQ(ierr);
  
  // For this problem, allow only change in density and viscosity
  ierr = PetscOptionsGetScalar(NULL, NULL, "-eta1", &ctx->eta1, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-eta2", &ctx->eta2, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-rho1", &ctx->rho1, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-rho2", &ctx->rho2, NULL); CHKERRQ(ierr);

  /* Create two DMStag objects: dmStokes(P-element,v-vertex) and dmCoeff(rho-corner,eta-corner/element) */
  /* Define stencils: dof0 per vertex, dof1 per edge, dof1 per face/element */
  PetscInt dofS0 = 0, dofS1 = 1,dofS2 = 1;
  PetscInt dofC0 = 2, dofC1 = 0,dofC2 = 1;
  PetscInt stencilWidth = 1;
    
  ierr = DMStagCreate2d(ctx->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, ctx->nx, ctx->ny, 
            PETSC_DECIDE, PETSC_DECIDE, dofS0, dofS1, dofS2, DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &ctx->dmStokes); CHKERRQ(ierr);

  /* Set options */
  ierr = DMSetFromOptions(ctx->dmStokes); CHKERRQ(ierr);
  ierr = DMSetUp         (ctx->dmStokes); CHKERRQ(ierr);
  
  /* Set coordinates dmStokes */
  ierr = DMStagSetUniformCoordinatesExplicit(ctx->dmStokes, ctx->xmin, ctx->xmax, ctx->ymin, ctx->ymax, 0.0, 0.0); CHKERRQ(ierr);
  
  /* Create dmCoeff(rho-corner,eta-corner/element) */
  ierr = DMStagCreateCompatibleDMStag(ctx->dmStokes, dofC0, dofC1, dofC2, 0, &ctx->dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(ctx->dmCoeff); CHKERRQ(ierr);
  
  /* Set coordinates dmCoeff */
  /* Note: use DMStagSetUniformCoordinatesProduct() for a more-efficient way to work with coordinates */
  ierr = DMStagSetUniformCoordinatesExplicit(ctx->dmCoeff, ctx->xmin, ctx->xmax, ctx->ymin, ctx->ymax, 0.0, 0.0); CHKERRQ(ierr);

  /* Get scaling constants as in Gerya 2009 (p.102) */
  {
    PetscScalar hxAvgInv;
    ctx->hxCharacteristic  = (ctx->xmax - ctx->xmin)/ctx->nx;
    ctx->hyCharacteristic  = (ctx->ymax - ctx->ymin)/ctx->ny;
    ctx->etaCharacteristic = PetscMin(PetscRealPart(ctx->eta1),PetscRealPart(ctx->eta2));
    hxAvgInv               = 2.0/(ctx->hxCharacteristic + ctx->hyCharacteristic);
    ctx->Kcont             = ctx->etaCharacteristic * hxAvgInv;
    ctx->Kbound            = ctx->etaCharacteristic * hxAvgInv * hxAvgInv;
  }
  /* Populate coefficient data */
  ierr = PopulateCoefficientData(ctx); CHKERRQ(ierr);

  /* Construct stiffness matrix and right-hand-side (system) */
  ierr = CreateSystem(ctx, &A, &b); CHKERRQ(ierr);

  /* Create solution vector to match rhs */
  ierr = VecDuplicate(b, &x); CHKERRQ(ierr);
  
  /* Create KSP context */
  ierr = KSPCreate(ctx->comm, &ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPFGMRES); CHKERRQ(ierr);
  
  /* Set Amat and Pmat */
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  
  /* Get default info on convergence */
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason", ""); CHKERRQ(ierr);
  
  /* Set KSP options */
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  
  /* Solve system */
  ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
  
  /* Analyze convergence */
  {
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Linear solve failed");CHKERRQ(ierr);
  }
  
  ierr = DumpSolution(ctx, x); CHKERRQ(ierr);

  /* Destroy PETSc objects (clean-up) */
  ierr = MatDestroy(&A           ); CHKERRQ(ierr);
  ierr = VecDestroy(&x           ); CHKERRQ(ierr);
  ierr = VecDestroy(&b           ); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->coeff  ); CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp         ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&ctx->dmStokes); CHKERRQ(ierr);
  ierr = DMDestroy(&ctx->dmCoeff ); CHKERRQ(ierr);
  
  ierr = PetscFree(ctx           ); CHKERRQ(ierr);
  
  /* Finalize main */
  ierr = PetscFinalize();
  return ierr;
}

/* --------------------------------------- */
/* Create System                           */
/* --------------------------------------- */
static PetscErrorCode CreateSystem(const Ctx ctx, Mat *pA, Vec *pRhs)
{
  PetscErrorCode ierr;
  PetscInt       Nx, Ny;                         // global variables
  PetscInt       ex, ey, startx, starty, nx, ny; // local variables
  Mat            A;
  Vec            rhs, coeffLocal;
  PetscScalar    hx, hy;
  PetscBool      pinPressure = PETSC_TRUE;

  PetscFunctionBeginUser;
  
  /* Create stiffness matrix A and rhs vector */
  ierr = DMCreateMatrix      (ctx->dmStokes, pA  ); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->dmStokes, pRhs); CHKERRQ(ierr);
  
  /* Assign pointers and other variables */
  A   = *pA;
  rhs = *pRhs;
  
  Nx = ctx->nx;
  Ny = ctx->ny;
  hx = ctx->hxCharacteristic;
  hy = ctx->hyCharacteristic;
  
  /* Get local domain */
  ierr = DMStagGetCorners(ctx->dmStokes, &startx, &starty, NULL, &nx, &ny, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dmCoeff, &coeffLocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (ctx->dmCoeff, ctx->coeff, INSERT_VALUES, coeffLocal); CHKERRQ(ierr);

  /* Loop over all local elements. Note: it may be more efficient in real applications to loop over each boundary separately */
  for (ey = starty; ey<starty+ny; ++ey) {
    for (ex = startx; ex<startx+nx; ++ex) {
    
      /* Top boundary velocity Dirichlet */
      if (ey == Ny-1) {
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar   valA = ctx->Kbound;
        
        row.i = ex; row.j = ey; row.loc = UP; row.c = 0;
        valRhs = 0.0;
        
        ierr   = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,1,&row,&valA  ,INSERT_VALUES);CHKERRQ(ierr);
        ierr   = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,       &valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }
      
      /* Bottom boundary velocity Dirichlet */
      if (ey == 0) {
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
        
        row.i = ex; row.j = ey; row.loc = DOWN; row.c = 0;
        valRhs = 0.0;
        
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,1,&row,&valA  ,INSERT_VALUES);CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,       &valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } 
      
      /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y : includes non-zero forcing */
      else {
        PetscInt      nEntries;
        DMStagStencil row,col[11];
        PetscScalar   valA[11];
        DMStagStencil rhoPoint[2];
        PetscScalar   rho[2],valRhs;
        DMStagStencil etaPoint[4];
        PetscScalar   eta[4],etaLeft,etaRight,etaUp,etaDown;
        
        /* Get rho values  and compute rhs value*/
        rhoPoint[0].i = ex; rhoPoint[0].j = ey; rhoPoint[0].loc = DOWN_LEFT;  rhoPoint[0].c = 1;
        rhoPoint[1].i = ex; rhoPoint[1].j = ey; rhoPoint[1].loc = DOWN_RIGHT; rhoPoint[1].c = 1;
        ierr = DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,2,rhoPoint,rho);CHKERRQ(ierr);
        valRhs = -ctx->gy * 0.5 * (rho[0] + rho[1]);

        /* Get eta values */
        etaPoint[0].i = ex; etaPoint[0].j = ey;   etaPoint[0].loc = DOWN_LEFT;  etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex; etaPoint[1].j = ey;   etaPoint[1].loc = DOWN_RIGHT; etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex; etaPoint[2].j = ey;   etaPoint[2].loc = ELEMENT;    etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex; etaPoint[3].j = ey-1; etaPoint[3].loc = ELEMENT;    etaPoint[3].c = 0; /* Down  */
        ierr = DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,4,etaPoint,eta);CHKERRQ(ierr);
        etaLeft = eta[0]; etaRight = eta[1]; etaUp = eta[2]; etaDown = eta[3];
        
        /* Left boundary y velocity stencil */
        if (ex == 0) {
          nEntries = 10;
          row.i    = ex  ; row.j     = ey  ; row.loc     = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j  = ey  ; col[0].loc  = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaRight) /(hx*hx);
          col[1].i = ex  ; col[1].j  = ey-1; col[1].loc  = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j  = ey+1; col[2].loc  = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          /* No left entry */
          col[3].i = ex+1; col[3].j  = ey  ; col[3].loc  = DOWN;     col[3].c  = 0; valA[3]  =        etaRight / (hx*hx);
          col[4].i = ex  ; col[4].j  = ey-1; col[4].loc  = LEFT;     col[4].c  = 0; valA[4]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[5].i = ex  ; col[5].j  = ey-1; col[5].loc  = RIGHT;    col[5].c  = 0; valA[5]  = -      etaRight / (hx*hy); /* down right x edge */
          col[6].i = ex  ; col[6].j  = ey  ; col[6].loc  = LEFT;     col[6].c  = 0; valA[6]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = RIGHT;    col[7].c  = 0; valA[7]  =        etaRight / (hx*hy); /* up right x edge */
          col[8].i = ex  ; col[8].j  = ey-1; col[8].loc  = ELEMENT;  col[8].c  = 0; valA[8]  =  ctx->Kcont / hy;
          col[9].i = ex  ; col[9].j = ey   ; col[9].loc = ELEMENT;   col[9].c  = 0; valA[9]  = -ctx->Kcont / hy;
        } 
        
        /* Right boundary y velocity stencil */
        else if (ex == Nx-1) {
          nEntries = 10;
          row.i    = ex  ; row.j     = ey  ; row.loc     = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j  = ey  ; col[0].loc  = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaLeft) /(hx*hx );
          col[1].i = ex  ; col[1].j  = ey-1; col[1].loc  = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j  = ey+1; col[2].loc  = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          col[3].i = ex-1; col[3].j  = ey  ; col[3].loc  = DOWN;     col[3].c  = 0; valA[3]  =        etaLeft  / (hx*hx);
          /* No right element */
          col[4].i = ex  ; col[4].j  = ey-1; col[4].loc  = LEFT;     col[4].c  = 0; valA[4]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[5].i = ex  ; col[5].j  = ey-1; col[5].loc  = RIGHT;    col[5].c  = 0; valA[5]  = -      etaRight / (hx*hy); /* down right x edge */
          col[6].i = ex  ; col[6].j  = ey  ; col[6].loc  = LEFT;     col[6].c  = 0; valA[7]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = RIGHT;    col[7].c  = 0; valA[7]  =        etaRight / (hx*hy); /* up right x edge */
          col[8].i = ex  ; col[8].j  = ey-1; col[8].loc  = ELEMENT;  col[8].c  = 0; valA[8]  =  ctx->Kcont / hy;
          col[9].i = ex  ; col[9].j = ey   ; col[9].loc = ELEMENT;   col[9].c  = 0; valA[9]  = -ctx->Kcont / hy;
        } 
        
        /* U_y interior equation */
        else {
          nEntries = 11;
          row.i    = ex  ; row.j     = ey  ; row.loc     = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j  = ey  ; col[0].loc  = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaLeft + etaRight) /(hx*hx);
          col[1].i = ex  ; col[1].j  = ey-1; col[1].loc  = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j  = ey+1; col[2].loc  = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          col[3].i = ex-1; col[3].j  = ey  ; col[3].loc  = DOWN;     col[3].c  = 0; valA[3]  =        etaLeft  / (hx*hx);
          col[4].i = ex+1; col[4].j  = ey  ; col[4].loc  = DOWN;     col[4].c  = 0; valA[4]  =        etaRight / (hx*hx);
          col[5].i = ex  ; col[5].j  = ey-1; col[5].loc  = LEFT;     col[5].c  = 0; valA[5]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[6].i = ex  ; col[6].j  = ey-1; col[6].loc  = RIGHT;    col[6].c  = 0; valA[6]  = -      etaRight / (hx*hy); /* down right x edge */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = LEFT;     col[7].c  = 0; valA[7]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[8].i = ex  ; col[8].j  = ey  ; col[8].loc  = RIGHT;    col[8].c  = 0; valA[8]  =        etaRight / (hx*hy); /* up right x edge */
          col[9].i = ex  ; col[9].j  = ey-1; col[9].loc  = ELEMENT;  col[9].c  = 0; valA[9]  =  ctx->Kcont / hy;
          col[10].i = ex ; col[10].j = ey  ; col[10].loc = ELEMENT; col[10].c  = 0; valA[10] = -ctx->Kcont / hy;
        }

        /* Insert Y-momentum entries */
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,nEntries,col, valA  , INSERT_VALUES); CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,             &valRhs, INSERT_VALUES); CHKERRQ(ierr);
      }
      
      /* Right Boundary velocity Dirichlet */
      if (ex == Nx-1) {
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
      
        row.i = ex; row.j = ey; row.loc = RIGHT; row.c = 0;
        valRhs = 0.0;
        
        ierr   = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,1,&row,&valA  ,INSERT_VALUES); CHKERRQ(ierr);
        ierr   = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,       &valRhs,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      /* Left velocity Dirichlet */
      if (ex == 0) {
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
        
        row.i = ex; row.j = ey; row.loc = LEFT; row.c = 0;
        valRhs = 0.0;
        
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,1,&row,&valA  ,INSERT_VALUES); CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,       &valRhs,INSERT_VALUES); CHKERRQ(ierr);
      } 
      
      /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
      else {
        PetscInt nEntries;
        DMStagStencil row,col[11];
        PetscScalar   valRhs,valA[11];
        DMStagStencil etaPoint[4];
        PetscScalar eta[4],etaLeft,etaRight,etaUp,etaDown;
        
        /* Get eta values */
        etaPoint[0].i = ex-1; etaPoint[0].j = ey; etaPoint[0].loc = ELEMENT;   etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex;   etaPoint[1].j = ey; etaPoint[1].loc = ELEMENT;   etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex;   etaPoint[2].j = ey; etaPoint[2].loc = UP_LEFT;   etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex;   etaPoint[3].j = ey; etaPoint[3].loc = DOWN_LEFT; etaPoint[3].c = 0; /* Down  */
        ierr = DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,4,etaPoint,eta); CHKERRQ(ierr);
        etaLeft = eta[0]; etaRight = eta[1]; etaUp = eta[2]; etaDown = eta[3];
        
        /* Bottom boundary x velocity stencil (with zero vel deriv) */
        if (ey == 0) {
          nEntries = 10;
          row.i     = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c      = 0;
          col[0].i  = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c   = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaUp) / (hy*hy);
          /* Missing element below */
          col[1].i  = ex  ; col[1].j  = ey+1; col[1].loc  = LEFT;    col[1].c   = 0; valA[1]  =        etaUp    / (hy*hy);
          col[2].i  = ex-1; col[2].j  = ey  ; col[2].loc  = LEFT;    col[2].c   = 0; valA[2]  =  2.0 * etaLeft  / (hx*hx);
          col[3].i  = ex+1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c   = 0; valA[3]  =  2.0 * etaRight / (hx*hx);
          col[4].i  = ex-1; col[4].j  = ey  ; col[4].loc  = DOWN;    col[4].c   = 0; valA[4]  =        etaDown  / (hx*hy); /* down left */
          col[5].i  = ex  ; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c   = 0; valA[5]  = -      etaDown  / (hx*hy); /* down right */
          col[6].i  = ex-1; col[6].j  = ey  ; col[6].loc  = UP;      col[6].c   = 0; valA[6]  = -      etaUp    / (hx*hy); /* up left */
          col[7].i  = ex  ; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c   = 0; valA[7]  =        etaUp    / (hx*hy); /* up right */
          col[8].i  = ex-1; col[8].j  = ey  ; col[8].loc  = ELEMENT; col[8].c   = 0; valA[8]  =  ctx->Kcont / hx;
          col[9].i = ex   ; col[9].j  = ey  ; col[9].loc  = ELEMENT; col[9].c   = 0; valA[9]  = -ctx->Kcont / hx;
          valRhs = 0.0;
        } 
        
        /* Top boundary x velocity stencil */
        else if (ey == Ny-1) {
          nEntries = 10;
          row.i     = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c      = 0;
          col[0].i  = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c   = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaDown) / (hy*hy);
          col[1].i  = ex  ; col[1].j  = ey-1; col[1].loc  = LEFT;    col[1].c   = 0; valA[1]  =        etaDown  / (hy*hy);
          /* Missing element above */
          col[2].i  = ex-1; col[2].j  = ey  ; col[2].loc  = LEFT;    col[2].c   = 0; valA[2]  =  2.0 * etaLeft  / (hx*hx);
          col[3].i  = ex+1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c   = 0; valA[3]  =  2.0 * etaRight / (hx*hx);
          col[4].i  = ex-1; col[4].j  = ey  ; col[4].loc  = DOWN;    col[4].c   = 0; valA[4]  =        etaDown  / (hx*hy); /* down left */
          col[5].i  = ex  ; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c   = 0; valA[5]  = -      etaDown  / (hx*hy); /* down right */
          col[6].i  = ex-1; col[6].j  = ey  ; col[6].loc  = UP;      col[6].c   = 0; valA[6]  = -      etaUp    / (hx*hy); /* up left */
          col[7].i  = ex  ; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c   = 0; valA[7]  =        etaUp    / (hx*hy); /* up right */
          col[8].i  = ex-1; col[8].j  = ey  ; col[8].loc  = ELEMENT; col[8].c   = 0; valA[8]  =  ctx->Kcont / hx;
          col[9].i = ex   ; col[9].j  = ey   ; col[9].loc = ELEMENT;  col[9].c  = 0; valA[9]  = -ctx->Kcont / hx;
          valRhs = 0.0;
        } 
        
        /* U_x interior equation */
        else {
          nEntries = 11;
          row.i     = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c      = 0;
          col[0].i  = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c   = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaUp + etaDown) / (hy*hy);
          col[1].i  = ex  ; col[1].j  = ey-1; col[1].loc  = LEFT;    col[1].c   = 0; valA[1]  =        etaDown  / (hy*hy);
          col[2].i  = ex  ; col[2].j  = ey+1; col[2].loc  = LEFT;    col[2].c   = 0; valA[2]  =        etaUp    / (hy*hy);
          col[3].i  = ex-1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c   = 0; valA[3]  =  2.0 * etaLeft  / (hx*hx);
          col[4].i  = ex+1; col[4].j  = ey  ; col[4].loc  = LEFT;    col[4].c   = 0; valA[4]  =  2.0 * etaRight / (hx*hx);
          col[5].i  = ex-1; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c   = 0; valA[5]  =        etaDown  / (hx*hy); /* down left */
          col[6].i  = ex  ; col[6].j  = ey  ; col[6].loc  = DOWN;    col[6].c   = 0; valA[6]  = -      etaDown  / (hx*hy); /* down right */
          col[7].i  = ex-1; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c   = 0; valA[7]  = -      etaUp    / (hx*hy); /* up left */
          col[8].i  = ex  ; col[8].j  = ey  ; col[8].loc  = UP;      col[8].c   = 0; valA[8]  =        etaUp    / (hx*hy); /* up right */
          col[9].i  = ex-1; col[9].j  = ey  ; col[9].loc  = ELEMENT; col[9].c   = 0; valA[9]  =  ctx->Kcont / hx;
          col[10].i = ex  ; col[10].j = ey  ; col[10].loc = ELEMENT; col[10].c  = 0; valA[10] = -ctx->Kcont / hx;
          valRhs = 0.0;
        }
        
        /* Insert X-momentum entries */
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,nEntries,col, valA  ,INSERT_VALUES); CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,             &valRhs,INSERT_VALUES); CHKERRQ(ierr);
      }

      /* P equation : u_x + v_y = 0 */
      /* Pin the first pressure node to zero, if requested */
      if (pinPressure && ex == 0 && ey == 0) { 
        DMStagStencil row;
        PetscScalar valA,valRhs;
      
        row.i = ex; row.j = ey; row.loc = ELEMENT; row.c = 0;
        valA   = ctx->Kbound;
        valRhs = 0.0;
        
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,1,&row,&valA  ,INSERT_VALUES); CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,       &valRhs,INSERT_VALUES); CHKERRQ(ierr);
      } 
      else {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;
        
        row.i    = ex; row.j    = ey; row.loc    = ELEMENT; row.c    = 0;
        col[0].i = ex; col[0].j = ey; col[0].loc = LEFT;    col[0].c = 0; valA[0] = -ctx->Kcont / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = RIGHT;   col[1].c = 0; valA[1] =  ctx->Kcont / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DOWN;    col[2].c = 0; valA[2] = -ctx->Kcont / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = UP;      col[3].c = 0; valA[3] =  ctx->Kcont / hy;
        col[4] = row;                                                     valA[4] = 0.0;
        valRhs = 0.0;
        
        /* Insert P-equation entries */
        ierr = DMStagMatSetValuesStencil(ctx->dmStokes,A  ,1,&row,5,col, valA  ,INSERT_VALUES); CHKERRQ(ierr);
        ierr = DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,      &valRhs,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  
  /* Restore local vector */
  ierr = DMRestoreLocalVector(ctx->dmCoeff,&coeffLocal); CHKERRQ(ierr);
  
  /* Matrix and Vector Assembly */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(rhs); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (rhs); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* --------------------------------------- */
/* Populate Coefficients                   */
/* --------------------------------------- */
static PetscErrorCode PopulateCoefficientData(Ctx ctx)
{
  PetscErrorCode ierr;
  PetscInt       startx, starty, nExtrax, nExtray;
  PetscInt       ex, ey, nx, ny, Nx, Ny;
  //Vec            coeffLocal; 
  Vec            coordLocal;
  DM             dmCoord;

  PetscFunctionBeginUser;
  
  /* Access vector with coefficient (rho, eta) */
  ierr = DMCreateGlobalVector(ctx->dmCoeff, &ctx->coeff); CHKERRQ(ierr);
  //ierr = DMGetLocalVector    (ctx->dmCoeff, &coeffLocal); CHKERRQ(ierr);
  
  /* Get domain corners */
  ierr = DMStagGetCorners(ctx->dmCoeff, &startx, &starty, NULL, &nx, &ny, NULL, &nExtrax, &nExtray, NULL); CHKERRQ(ierr);
  
  /* Get coordinates */
  ierr = DMGetCoordinatesLocal(ctx->dmCoeff, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (ctx->dmCoeff, &dmCoord);    CHKERRQ(ierr);
  
  Nx = ctx->nx;
  Ny = ctx->ny;
  
  /* Loop over local domain */
  for (ey = starty; ey<starty+ny+nExtray; ++ey) {
    for (ex = startx; ex<startx+nx+nExtrax; ++ex) {

      /* Eta (element) */
      if (ey < starty + ny && ex < startx + nx) {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of eta point */
        point.i = ex; point.j = ey; point.loc = ELEMENT; point.c = 0;
        pointCoordx = point;
        ierr = DMStagVecGetValuesStencil(dmCoord, coordLocal, 1, &pointCoordx, &x); CHKERRQ(ierr);
        
        /* Get eta value - this should be replaced by interpolation */
        val  = getEta(ctx, x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff, ctx->coeff, 1, &point, &val, INSERT_VALUES); CHKERRQ(ierr);
      }

      /* Rho */
      {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of rho point */
        point.i = ex; point.j = ey; point.loc = DOWN_LEFT; point.c = 1;
        pointCoordx = point; pointCoordx.c = 0;
        ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
        
        /* Get rho value - this should be replaced by interpolation */
        val  = getRho(ctx,x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES); CHKERRQ(ierr);
      }

      /* Eta (corner) - populate extra corners on right and top of domain */
      {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of eta point */
        point.i = ex; point.j = ey; point.loc = DOWN_LEFT; point.c = 0;
        pointCoordx = point;
        ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
        
        /* Get eta value */
        val = getEta(ctx,x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      /* Eta (extra corners) - DOWN-RIGHT*/
      if (ex == Nx-1) {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of eta point */
        point.i = ex; point.j = ey; point.loc = DOWN_RIGHT; point.c = 0;
        pointCoordx = point;
        ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
        
        /* Get eta value */
        val = getEta(ctx,x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      /* Eta (extra corners) - UP-LEFT */
      if (ey == Ny-1) {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of eta point */
        point.i = ex; point.j = ey; point.loc = UP_LEFT; point.c = 0;
        pointCoordx = point;
        ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
        
        /* Get eta value */
        val = getEta(ctx,x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      /* Eta (extra corners) - UP-RIGHT */
      if (ex == Nx-1 && ey == Ny-1) {
        DMStagStencil point, pointCoordx;
        PetscScalar   val, x;
        
        /* Get coordinate of eta point */
        point.i = ex; point.j = ey; point.loc = UP_RIGHT; point.c = 0;
        pointCoordx = point;
        ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
        
        /* Get eta value */
        val = getEta(ctx,x);
        ierr = DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  
  /* Vector Assembly and Restore local vector */
  ierr = VecAssemblyBegin(ctx->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (ctx->coeff); CHKERRQ(ierr);
  
  /* Restore local vector */
  //ierr = DMRestoreLocalVector(ctx->dmCoeff, &coeffLocal); CHKERRQ(ierr);
  
  /* Return function */
  PetscFunctionReturn(0);
}

/* --------------------------------------- */
/* Dump Solution                           */
/* --------------------------------------- */
static PetscErrorCode DumpSolution(Ctx ctx, Vec x)
{
  PetscErrorCode ierr;
  DM             daVelAvg,  daP,  daEtaElement,  daEtaCorner,  daRho,  dmVelAvg;
  Vec            vecVelAvg, vecP, vecEtaElement, vecEtaCorner, vecRho, velAvg;

  PetscFunctionBeginUser;

  /* For convenience, create a new DM and Vec which will hold averaged velocities */
  ierr = DMStagCreateCompatibleDMStag(ctx->dmStokes,0,0,2,0,&dmVelAvg); CHKERRQ(ierr); /* 2 dof per element */
  ierr = DMSetUp(dmVelAvg); CHKERRQ(ierr);
  
  /* Set Coordinates */
  ierr = DMStagSetUniformCoordinatesExplicit(dmVelAvg,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,0.0,0.0); CHKERRQ(ierr);
  
  /* Create global vector */
  ierr = DMCreateGlobalVector(dmVelAvg,&velAvg); CHKERRQ(ierr);
  
  /* Loop over elements */
  {
    PetscInt ex, ey, startx, starty, nx, ny;
    Vec      stokesLocal;
    
    /* Access local vector */
    ierr = DMGetLocalVector(ctx->dmStokes,&stokesLocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (ctx->dmStokes,x,INSERT_VALUES,stokesLocal); CHKERRQ(ierr);
    
    /* Get corners */
    ierr = DMStagGetCorners(dmVelAvg,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
    
    /* Loop */
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4], to[2];
        PetscScalar   valFrom[4], valTo[2];
        
        from[0].i = ex; from[0].j = ey; from[0].loc = UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = RIGHT; from[3].c = 0;
        
        /* Get values from stencil locations */
        ierr = DMStagVecGetValuesStencil(ctx->dmStokes,stokesLocal,4,from,valFrom); CHKERRQ(ierr);
        
        /* Average edge values to obtain ELEMENT values */
        to[0].i = ex; to[0].j = ey; to[0].loc = ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        
        /* Return values in new dm */
        ierr = DMStagVecSetValuesStencil(dmVelAvg,velAvg,2,to,valTo,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    
    /* Vector assembly */
    ierr = VecAssemblyBegin(velAvg); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (velAvg); CHKERRQ(ierr);
    
    /* Restore vector */
    ierr = DMRestoreLocalVector(ctx->dmStokes,&stokesLocal); CHKERRQ(ierr);
  }

  /* Create individual DMDAs for sub-grids of our DMStag objects */
  ierr = DMStagVecSplitToDMDA(ctx->dmStokes,x,DMSTAG_ELEMENT,0,&daP,&vecP); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecP,"p (scaled)");              CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_DOWN_LEFT,0,&daEtaCorner,&vecEtaCorner); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecEtaCorner,"eta");                                       CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_ELEMENT,  0,&daEtaElement,&vecEtaElement); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecEtaElement,"eta");                                        CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_DOWN_LEFT,  1, &daRho,       &vecRho); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecRho,"density");                                       CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(dmVelAvg,    velAvg,    DMSTAG_ELEMENT,  -3,&daVelAvg,    &vecVelAvg); CHKERRQ(ierr); /* note -3 : pad with zero */
  ierr = PetscObjectSetName  ((PetscObject)vecVelAvg,"Velocity (Averaged)");                         CHKERRQ(ierr);

  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVelAvg),"stagridge_element.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    
    ierr = VecView(vecVelAvg,    viewer); CHKERRQ(ierr);
    ierr = VecView(vecP,         viewer); CHKERRQ(ierr);
    ierr = VecView(vecEtaElement,viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerDestroy  (&viewer); CHKERRQ(ierr);
  }

  /* Dump vertex-based fields to a second .vtr file */
  {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daEtaCorner),"stagridge_corner.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    
    ierr = VecView(vecEtaCorner, viewer); CHKERRQ(ierr);
    ierr = VecView(vecRho,       viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerDestroy  (&viewer); CHKERRQ(ierr);
  }

  /* Destroy DMDAs and Vecs */
  ierr = VecDestroy(&vecVelAvg    ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecP         ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecEtaCorner ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecEtaElement); CHKERRQ(ierr);
  ierr = VecDestroy(&vecRho       ); CHKERRQ(ierr);
  ierr = VecDestroy(&velAvg       ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&daVelAvg    ); CHKERRQ(ierr);
  ierr = DMDestroy(&daP         ); CHKERRQ(ierr);
  ierr = DMDestroy(&daEtaCorner ); CHKERRQ(ierr);
  ierr = DMDestroy(&daEtaElement); CHKERRQ(ierr);
  ierr = DMDestroy(&daRho       ); CHKERRQ(ierr);
  ierr = DMDestroy(&dmVelAvg    ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// Run it:
// mpiexec -n 2 solCx -nx 51 -ny 51
