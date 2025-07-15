
#ifndef __material_point_h__
#define __material_point_h__


#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>

#define DMSWARM_DATAFIELD_POINT_ACCESS_GUARD

typedef enum { COOR_INITIALIZE = 0, COOR_APPEND } MPointCoordinateInsertMode;
typedef enum { SWARM_FIELDS_SAME = 0, SWARM_FIELDS_SUBSET, SWARM_FIELDS_SUPERSET, SWARM_FIELDS_DISJOINT } DMSwarmFieldsCompareType;

const char DMSwarmPICField_cellid[] = "DMSwarm_cellid";

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

PetscErrorCode MPoint_ProjectP0_arith(DM dmswarm,const char propname[],
                                      DM dmstag,DM dmcell,PetscInt element_dof,Vec cellcoeff);

PetscErrorCode MPoint_ProjectQ1_arith_general(DM dmswarm,const char propname[],
                                              DM dmstag,DM dmcell,
                                              PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
                                              PetscInt dof,Vec cellcoeff);

PetscErrorCode MPoint_ProjectQ1_arith_general_AP(DM dmswarm,const char propname[],
                                      DM dmstag,
                                      DM dmcell,
                                      PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
                                      PetscInt dof,Vec cellcoeff);

PetscErrorCode MPointCoordLayout_DomainVolume(DM dmswarm,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode);

PetscErrorCode MPointCoordLayout_DomainVolumeWithCellList(DM dmswarm,
                                                          PetscInt _ncells,PetscInt celllist[],
                                                          PetscReal factor,PetscInt points_per_dim[],
                                                          MPointCoordinateInsertMode mode);

PetscErrorCode MPointCoordLayout_DomainFace(DM dmswarm,char face,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode);

PetscErrorCode MPoint_AdvectRK1(DM,DM,Vec,PetscReal);

PetscErrorCode DMSwarmQueryField(DM dm,const char fieldname[],PetscBool *found);
PetscErrorCode DMSwarmDuplicateRegisteredFields(DM dmA,DM dmB);
PetscErrorCode DMSwarmCopyFieldValues(DM dmA,DM dmB,PetscBool *copy_occurred);
PetscErrorCode DMSwarmCopySubsetFieldValues(DM dmA,PetscInt np,PetscInt list[],DM dmB,PetscBool *copy_occurred);
PetscErrorCode DMSwarmRemovePoints(DM dm,PetscInt np,PetscInt list[]);

PetscErrorCode DMSwarmFieldSet(DM dm,const char fieldname[],PetscReal alpha);
PetscErrorCode DMSwarmFieldSetWithRange(DM dm,const char fieldname[],PetscInt ps,PetscInt pe,PetscReal alpha);
PetscErrorCode DMSwarmFieldSetWithList(DM dm,const char fieldname[],PetscInt np,PetscInt list[],PetscReal alpha);

#endif
