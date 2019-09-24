CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
PROGDIR  = tests
OBJECTS  = prealloc_helper.o utils.o bc.o fd.o coefficient.o fdstokes_physics.o fdstokes_residual.o fdstokes.o 

all: test_stokes_solcx test_bclist
	@echo "# -------------------------------- #"
	@echo "# -------------------------------- #"

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

test_stokes_solcx: ${PROGDIR}/test_stokes_solcx.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_stokes_solcx ${PROGDIR}/test_stokes_solcx.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

test_bclist: ${PROGDIR}/test_bclist.o ${OBJECTS} chkopts
	-${CLINKER} -o ${PROGDIR}/test_bclist ${PROGDIR}/test_bclist.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf ${PROG} *.dSYM
