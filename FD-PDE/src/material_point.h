
#ifndef __material_point_h__
#define __material_point_h__


#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>

typedef enum { COOR_INITIALIZE = 0, COOR_APPEND } MPointCoordinateInsertMode;
typedef enum { SWARM_FIELDS_SAME = 0, SWARM_FIELDS_SUBSET, SWARM_FIELDS_SUPERSET, SWARM_FIELDS_DISJOINT } DMSwarmFieldsCompareType;

PetscErrorCode DMStagPICCreateDMSwarm(DM,DM*);

PetscErrorCode DMStagPICFinalize(DM);

PetscErrorCode MPoint_ProjectP0_arith(DM dmswarm,const char propname[],
                                      DM dmstag,DM dmcell,PetscInt element_dof,Vec cellcoeff);

PetscErrorCode MPoint_ProjectQ1_arith_general(DM dmswarm,const char propname[],
                                              DM dmstag,DM dmcell,
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
