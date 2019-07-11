#include "stagridge.h"

// ---------------------------------------
// DoOutput
// ---------------------------------------
PetscErrorCode DoOutput(SolverCtx *sol)
{
  DM             dmVel,  daVel, dmEta,  daEta, daP,  daRho;
  Vec            vecVel, vaVel, vecEta, vaEta, vecP, vecRho;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV,0,0,2,0,&dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(dmVel); CHKERRQ(ierr);

  // Create a new DM and Vec for viscosity
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV,0,0,1,0,&dmEta); CHKERRQ(ierr);
  ierr = DMSetUp(dmEta); CHKERRQ(ierr);
  
  // Set Coordinates
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,sol->grd->xmin,sol->grd->xmax,sol->grd->zmin,sol->grd->zmax,0.0,0.0); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesExplicit(dmEta,sol->grd->xmin,sol->grd->xmax,sol->grd->zmin,sol->grd->zmax,0.0,0.0); CHKERRQ(ierr);
  
  // Create global vectors
  ierr = DMCreateGlobalVector(dmVel,&vecVel); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmEta,&vecEta); CHKERRQ(ierr);
  
  // Loop over elements
  {
    PetscInt     i, j, sx, sz, nx, nz;
    PetscScalar  eta;
    Vec          xlocal;
    
    // Access local vector
    ierr = DMGetLocalVector(sol->dmPV,&xlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (sol->dmPV,sol->x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
    
    // Get corners
    ierr = DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
    
    // Loop
    for (j = sz; j < sz+nz; ++j) {
      for (i = sx; i < sx+nx; ++i) {
        DMStagStencil from[4], to[2], row;
        PetscScalar   valFrom[4], valTo[2];
        
        from[0].i = i; from[0].j = j; from[0].loc = UP;    from[0].c = 0;
        from[1].i = i; from[1].j = j; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = i; from[2].j = j; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = i; from[3].j = j; from[3].loc = RIGHT; from[3].c = 0;
        
        // Get values from stencil locations
        ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,4,from,valFrom); CHKERRQ(ierr);
        
        // Average edge values to obtain ELEMENT values
        to[0].i = i; to[0].j = j; to[0].loc = ELEMENT; to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = i; to[1].j = j; to[1].loc = ELEMENT; to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        
        // Return values in new dm - averaged velocities
        ierr = DMStagVecSetValuesStencil(dmVel,vecVel,2,to,valTo,INSERT_VALUES); CHKERRQ(ierr);

        // Calculate element viscosity
        ierr = CalcEffViscosity(sol, xlocal, i, j, CENTER, &eta); CHKERRQ(ierr);

        // Return values in new dm - viscosity
        row.i = i; row.j = j; row.loc = ELEMENT; row.c = 0;
        ierr = DMStagVecSetValuesStencil(dmEta,vecEta,1,&row,&eta,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    
    // Vector assembly
    ierr = VecAssemblyBegin(vecVel); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (vecVel); CHKERRQ(ierr);

    ierr = VecAssemblyBegin(vecEta); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (vecEta); CHKERRQ(ierr);
    
    // Restore vector
      // Restore arrays, local vectors
    ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);
  }

  // Create individual DMDAs for sub-grids of our DMStag objects
  ierr = DMStagVecSplitToDMDA(sol->dmPV,sol->x,ELEMENT, 0,&daP,&vecP); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecP,"Pressure");         CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(dmVel, vecVel,ELEMENT,-3,&daVel,&vaVel); CHKERRQ(ierr); // note -3 : output 2 DOFs
  ierr = PetscObjectSetName  ((PetscObject)vaVel,"Velocity");          CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(sol->dmCoeff,sol->coeff,ELEMENT,0, &daRho, &vecRho); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecRho,"Density");                      CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dmEta, vecEta,ELEMENT,0,&daEta,&vaEta); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vaEta,"Eta");              CHKERRQ(ierr);

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),"stagridge_element.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    
    ierr = VecView(vecRho,   viewer); CHKERRQ(ierr);
    ierr = VecView(vaEta,    viewer); CHKERRQ(ierr);
    ierr = VecView(vaVel,    viewer); CHKERRQ(ierr);
    ierr = VecView(vecP,     viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerDestroy  (&viewer); CHKERRQ(ierr);
  }

  // Destroy DMDAs and Vecs
  ierr = VecDestroy(&vecVel); CHKERRQ(ierr);
  ierr = VecDestroy(&vaVel ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecP  ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecRho); CHKERRQ(ierr);
  ierr = VecDestroy(&vecEta); CHKERRQ(ierr);
  ierr = VecDestroy(&vaEta ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&dmVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daP    ); CHKERRQ(ierr);
  ierr = DMDestroy(&daRho  ); CHKERRQ(ierr);
  ierr = DMDestroy(&dmEta  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daEta  ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}