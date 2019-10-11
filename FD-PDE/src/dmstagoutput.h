/* Output routines for DMStag */

#ifndef DMSTAGOUTPUT_H
#define DMSTAGOUTPUT_H

#include "petsc.h"

#define OUTPUT_NAME_LENGTH 200

typedef enum { OUT_VERTEX = 0, OUT_FACE, OUT_ELEMENT } OutputType;
typedef enum { VTK_CENTER = 0, VTK_CORNER} OutputVTKType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
typedef struct {
  char                  name[OUTPUT_NAME_LENGTH]; 
  PetscInt              c;
  // DMStagStencilLocation loc;
  OutputType            type;
  PetscBool             filled;
} DMStagOutputLabel;

typedef struct {
  char                 *name; 
  PetscScalar          *data;
  PetscInt              size;
} DMStagOutputBuffer;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode DMStagOutputGetLabels(DM,DMStagOutputLabel**);
PetscErrorCode DMStagOutputAddLabel(DM,DMStagOutputLabel*, const char[],PetscInt, DMStagStencilLocation);
PetscErrorCode DMStagOutputVTKBinary(DM,Vec,DMStagOutputLabel*,OutputVTKType,const char[]);

#endif