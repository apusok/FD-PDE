#include "fdstokes.h"

// ---------------------------------------
// ContinuityResidual
// ---------------------------------------
PetscErrorCode ContinuityResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt n[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[4], rhs, dx, dz;
  PetscInt       sx, sz, nz;
  PetscInt       iprev, inext, nEntries = 4;
  DMStagStencil  point[4];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  sx = n[0]; 
  sz = n[1]; 
  nz = n[3];
  iprev = n[7]; 
  inext = n[8];

  // Get stencil values
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 0;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 0;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 0;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 0;
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);
  
  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT;  point[0].c = 0;
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, point, &rhs); CHKERRQ(ierr);

  // Calculate residual
  dx = coordx[i][inext]-coordx[i][iprev];
  dz = coordz[j][inext]-coordz[j][iprev];
  ffi = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz - rhs;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// XMomentumResidual
// ---------------------------------------
PetscErrorCode XMomentumResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt i, PetscInt j,PetscInt n[],PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdx, dVxdx, rhs, ffi;
  PetscInt       nEntries = 11, iprev, inext, icenter;
  PetscInt       sx, sz, nz, Nz;
  PetscScalar    xx[11], dx, dx1, dx2, dz, dz1, dz2;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  point[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  sx = n[0]; sz = n[1]; 
  nz = n[3]; Nz = n[5];
  icenter = n[6]; iprev = n[7]; inext = n[8];

  // Get stencil values
  point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_LEFT;    point[0].c   = 0; // Vx(i  ,j  )
  point[1].i  = i  ; point[1].j  = j-1; point[1].loc  = DMSTAG_LEFT;    point[1].c   = 0; // Vx(i  ,j-1)
  point[2].i  = i  ; point[2].j  = j+1; point[2].loc  = DMSTAG_LEFT;    point[2].c   = 0; // Vx(i  ,j+1)
  point[3].i  = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_LEFT;    point[3].c   = 0; // Vx(i-1,j  )
  point[4].i  = i  ; point[4].j  = j  ; point[4].loc  = DMSTAG_RIGHT;   point[4].c   = 0; // Vx(i+1,j  )
  point[5].i  = i-1; point[5].j  = j  ; point[5].loc  = DMSTAG_DOWN;    point[5].c   = 0; // Vz(i-1,j-1)
  point[6].i  = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_DOWN;    point[6].c   = 0; // Vz(i  ,j-1)
  point[7].i  = i-1; point[7].j  = j  ; point[7].loc  = DMSTAG_UP;      point[7].c   = 0; // Vz(i-1,j  )
  point[8].i  = i  ; point[8].j  = j  ; point[8].loc  = DMSTAG_UP;      point[8].c   = 0; // Vz(i  ,j  )
  point[9].i  = i-1; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0; // P (i-1,j  )
  point[10].i = i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c  = 0; // P (i  ,j  )

  // For boundaries remove the flux term
  if (j == Nz-1) point[2] = point[0];
  if (j == 0   ) point[1] = point[0];

  // Residual values
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);

  // Coefficients
  PetscScalar  cx[5];
  point[0].i = i  ; point[0].j = j; point[0].loc = DMSTAG_LEFT;      point[0].c = 0; // rhs
  point[1].i = i-1; point[1].j = j; point[1].loc = DMSTAG_ELEMENT;   point[1].c = 1; // etc_c - left
  point[2].i = i  ; point[2].j = j; point[2].loc = DMSTAG_ELEMENT;   point[2].c = 1; // etc_c - right
  point[3].i = i  ; point[3].j = j; point[3].loc = DMSTAG_UP_LEFT;   point[3].c = 0; // etc_n - up
  point[4].i = i  ; point[4].j = j; point[4].loc = DMSTAG_DOWN_LEFT; point[4].c = 0; // etc_n - down

  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 5, point, cx); CHKERRQ(ierr);

  rhs      = cx[0];
  etaLeft  = cx[1];
  etaRight = cx[2];
  etaUp    = cx[3];
  etaDown  = cx[4];

  // Grid spacings - need to correct for missing values
  dx  = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx1 = coordx[i-1][inext  ]-coordx[i-1][iprev  ];
  dx2 = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Correct for boundaries
  if (j == 0) {
    dz1 = 2.0*(coordz[j][icenter]-coordz[j][iprev]);
  } else {
    dz1 = coordz[j  ][icenter]-coordz[j-1][icenter];
  }
  if (j == Nz-1) {
    dz2 = 2.0*(coordz[j][inext]-coordz[j][icenter]);
  } else {
    dz2 = coordz[j+1][icenter]-coordz[j  ][icenter];
  }

  // Calculate new residual
  dPdx  = (xx[10]-xx[9])/dx;
  dVxdx = etaRight*(xx[4]-xx[0])/dx2 - etaLeft*(xx[0]-xx[3])/dx1;
  dVxdz = etaUp   *(xx[2]-xx[0])/dz2 - etaDown*(xx[0]-xx[1])/dz1;
  dVzdx = etaUp   *(xx[8]-xx[7])/dx  - etaDown*(xx[6]-xx[5])/dx;

  ffi   = -dPdx + 2.0*dVxdx/dx + dVxdz/dz + dVzdx/dz - rhs;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ZMomentumResidual
// ---------------------------------------
PetscErrorCode ZMomentumResidual(DM dm, Vec xlocal, DM dmcoeff, Vec coefflocal,PetscScalar **coordx,PetscScalar **coordz,PetscInt i, PetscInt j,PetscInt n[],PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdz, dVzdz, rhs, ffi;
  PetscInt       nEntries = 11, iprev, inext, icenter;
  PetscInt       sx, sz, nz, Nx;
  PetscScalar    xx[11], dx, dz, dx1, dx2, dz1, dz2;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  point[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  sx = n[0]; sz = n[1]; 
  nz = n[3]; Nx = n[4];
  icenter = n[6]; iprev = n[7]; inext = n[8];

  // Get stencil values
  point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_DOWN;    point[0].c   = 0; // Vz(i  ,j  )
  point[1].i  = i  ; point[1].j  = j  ; point[1].loc  = DMSTAG_UP;      point[1].c   = 0; // Vz(i  ,j+1)
  point[2].i  = i  ; point[2].j  = j-1; point[2].loc  = DMSTAG_DOWN;    point[2].c   = 0; // Vz(i  ,j-1)
  point[3].i  = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_DOWN;    point[3].c   = 0; // Vz(i-1,j  )
  point[4].i  = i+1; point[4].j  = j  ; point[4].loc  = DMSTAG_DOWN;    point[4].c   = 0; // Vz(i+1,j  )
  point[5].i  = i  ; point[5].j  = j  ; point[5].loc  = DMSTAG_LEFT;    point[5].c   = 0; // Vx(i-1,j  )
  point[6].i  = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_RIGHT;   point[6].c   = 0; // Vx(i  ,j  )
  point[7].i  = i  ; point[7].j  = j-1; point[7].loc  = DMSTAG_LEFT;    point[7].c   = 0; // Vx(i-1,j-1)
  point[8].i  = i  ; point[8].j  = j-1; point[8].loc  = DMSTAG_RIGHT;   point[8].c   = 0; // Vx(i  ,j-1)
  point[9].i  = i  ; point[9].j  = j-1; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0; // P (i  ,j-1)
  point[10].i = i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c  = 0; // P (i  ,j  )
  
  // For boundaries remove the flux term
  if (i == 0   ) point[3] = point[0];
  if (i == Nx-1) point[4] = point[0];

  // Get values
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);

  // Coefficients
  PetscScalar  cx[5];
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_DOWN;       point[0].c = 0; // rhs
  point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_DOWN_LEFT;  point[1].c = 0; // etc_n - left
  point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN_RIGHT; point[2].c = 0; // etc_n - right
  point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_ELEMENT;    point[3].c = 1; // etc_c - up
  point[4].i = i  ; point[4].j = j-1; point[4].loc = DMSTAG_ELEMENT;    point[4].c = 1; // etc_c - down

  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 5, point, cx); CHKERRQ(ierr);

  rhs      = cx[0];
  etaLeft  = cx[1];
  etaRight = cx[2];
  etaUp    = cx[3];
  etaDown  = cx[4];

  // Grid spacings
  dx  = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz1 = coordz[j-1][inext  ]-coordz[j-1][iprev  ];
  dz2 = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Correct for boundaries
  if (i == 0) {
    dx1 = 2.0*(coordx[i][icenter]-coordx[i][iprev]);
  } else {
    dx1 = coordx[i  ][icenter]-coordx[i-1][icenter];
  }
  if (i == Nx-1) {
    dx2 = 2.0*(coordx[i][inext]-coordx[i][icenter]);
  } else {
    dx2 = coordx[i+1][icenter]-coordx[i  ][icenter];
  }

  // Calculate residual
  dPdz  = (xx[10]-xx[9])/dz;
  dVzdz = etaUp   *(xx[1]-xx[0])/dz2 - etaDown *(xx[0]-xx[2])/dz1;
  dVzdx = etaRight*(xx[4]-xx[0])/dx2 - etaLeft *(xx[0]-xx[3])/dx1;
  dVxdz = etaRight*(xx[6]-xx[8])/dz - etaLeft *(xx[5]-xx[7])/dz;

  ffi   = -dPdz + 2.0*dVzdz/dz + dVzdx/dx + dVxdz/dx - rhs;

  *ff = ffi;

  PetscFunctionReturn(0);
}