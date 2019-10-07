CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
PROGDIR  = tests
OBJECTS  = dmstagbclist.o prealloc_helper.o fdpde.o fdpde_stokes_physics.o fdpde_stokes_residual.o fdpde_stokes.o 

all: test_stokes_solcx test_stokes_solcx_vargrid test_dmstagbclist test_fdpde
	@echo "# -------------------------------- #"
	@echo "# Compiled Programs"
	@echo "# -------------------------------- #"
	${RM} ${OBJECTS}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

test_stokes_solcx: ${PROGDIR}/test_stokes_solcx.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_stokes_solcx ${PROGDIR}/test_stokes_solcx.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

test_stokes_solcx_vargrid: ${PROGDIR}/test_stokes_solcx_vargrid.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_stokes_solcx_vargrid ${PROGDIR}/test_stokes_solcx_vargrid.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

test_dmstagbclist: ${PROGDIR}/test_dmstagbclist.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_dmstagbclist ${PROGDIR}/test_dmstagbclist.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

test_fdpde: ${PROGDIR}/test_fdpde.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_fdpde ${PROGDIR}/test_fdpde.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf ${PROGDIR}/test_stokes_solcx ${PROGDIR}/test_stokes_solcx_vargrid ${PROGDIR}/test_dmstagbclist ${PROGDIR}/test_fdpde ${PROGDIR}/*.o ${PROGDIR}/*.dSYM