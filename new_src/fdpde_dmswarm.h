#ifndef __fdpde_dmswarm_h__
#define __fdpde_dmswarm_h__

#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmswarm.h>
#include <petscdmstag.h>
#include <petsc/private/dmswarmimpl.h>
#include <petsc/private/dmstagimpl.h>
#include <petscdmproduct.h>

#define DMSWARM_DATAFIELD_POINT_ACCESS_GUARD

typedef enum { COOR_INITIALIZE = 0, COOR_APPEND } MPointCoordinateInsertMode;
typedef enum { SWARM_FIELDS_SAME = 0, SWARM_FIELDS_SUBSET, SWARM_FIELDS_SUPERSET, SWARM_FIELDS_DISJOINT } DMSwarmFieldsCompareType;

struct _p_DMSwarmDataField {
	char          *registration_function;
	PetscInt      L,bs;
	PetscBool     active;
	size_t        atomic_size;
	char          *name; /* what are they called */
	void          *data; /* the data - an array of structs */
  PetscDataType petsc_type;
};

struct _p_DMSwarmDataBucket {
	PetscInt  L;             /* number in use */
	PetscInt  buffer;        /* memory buffer used for re-allocation */
	PetscInt  allocated;     /* number allocated, this will equal datafield->L */
	PetscBool finalised;     /* DEPRECATED */
	PetscInt  nfields;       /* how many fields of this type */
	DMSwarmDataField *field; /* the data */
};

#define DMSWARM_DATAFIELD_point_access(data,index,atomic_size) (void*)((char*)(data) + (index)*(atomic_size))
#define DMSWARM_DATAFIELD_point_access_offset(data,index,atomic_size,offset) (void*)((char*)(data) + (index)*(atomic_size) + (offset))

PetscErrorCode DMStagPICCreateDMSwarm(DM,DM*);
PetscErrorCode DMStagPICFinalize(DM);

PetscErrorCode MPoint_ProjectP0_arith(DM,const char[],DM,DM,PetscInt,Vec);
PetscErrorCode MPoint_ProjectQ1_arith_general(DM,const char[],DM,DM,PetscInt,PetscInt,Vec);
PetscErrorCode MPoint_ProjectQ1_arith_general_AP(DM,const char[],DM,DM,PetscInt,PetscInt,Vec);

// PetscErrorCode MPointCoordLayout_DomainVolume(DM dmswarm,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode);

PetscErrorCode MPointCoordLayout_DomainVolumeWithCellList(DM,PetscInt,PetscInt[],PetscReal,PetscInt[],MPointCoordinateInsertMode);
PetscErrorCode MPointCoordLayout_DomainFace(DM,char,PetscReal,PetscInt,MPointCoordinateInsertMode);
PetscErrorCode MPoint_AdvectRK1(DM,DM,Vec,PetscReal);

PetscErrorCode DMSwarmQueryField(DM dm,const char fieldname[],PetscBool *found);
PetscErrorCode DMSwarmDuplicateRegisteredFields(DM,DM);
PetscErrorCode DMSwarmCopyFieldValues(DM,DM,PetscBool*);
PetscErrorCode DMSwarmCopySubsetFieldValues(DM,PetscInt,PetscInt[],DM,PetscBool*);
PetscErrorCode DMSwarmRemovePoints(DM,PetscInt,PetscInt[]);

PetscErrorCode DMSwarmFieldSet(DM,const char[],PetscReal);
PetscErrorCode DMSwarmFieldSetWithRange(DM,const char[],PetscInt,PetscInt,PetscReal);
PetscErrorCode DMSwarmFieldSetWithList(DM,const char[],PetscInt,PetscInt[],PetscReal);

// ---------------------------------------
// Other functions
// ---------------------------------------
PetscErrorCode DMStagGetLocalElementIndex(DM,PetscInt*,PetscInt*);
PetscErrorCode DMStagGetLocalElementGlobalIndices(DM,PetscInt,PetscInt*);
PetscErrorCode DMLocatePoints_Stag(DM,Vec,DMPointLocationType,PetscSF);

PetscErrorCode DMSetPointLocation(DM,PetscErrorCode (*)(DM,Vec,DMPointLocationType,PetscSF));
PetscErrorCode DMStagGetBoundingBox(DM,PetscReal[],PetscReal[]);
PetscErrorCode DMStagLocalElementIndexInGlobalSpace_2d(DM,PetscInt,PetscBool*);
PetscErrorCode DMStagFieldISCreate_2d(DM,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscInt,PetscInt[],IS*);
PetscErrorCode DMStagISCreateL2L_2d(DM,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscInt,PetscInt[],IS*,DM,PetscInt[],PetscInt[],PetscInt[],IS*);

#endif
