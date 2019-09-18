CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
PROGDIR  = tests
PROG     = ${PROGDIR}/test_stokes_solcx
OBJECTS  = ${PROG}.o prealloc_helper.o utils.o bc.o fd.o coefficient.o fdstokes_physics.o fdstokes_residual.o fdstokes.o 

all: ${PROG}
	@echo "# -------------------------------- #"
	@echo "# Compiled Program: " ${PROG} "    #"
	@echo "# -------------------------------- #"
	${RM} ${OBJECTS}
	@rm -rf ${PROG}.dSYM

debug: ${PROG}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

${PROG}: ${OBJECTS} chkopts
	${RM} ${PROG}
	-${CLINKER} -o ${PROG} ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf ${PROG} *.dSYM