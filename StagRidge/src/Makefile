CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
PROG     = stagridge
OBJECTS  = ${PROG}.o prealloc_helper.o input.o consteq.o physics.o bc.o residual.o solver.o initialize.o utils.o output.o benchmark.o

all: ${PROG}
	@echo "# -------------------------------- #"
	@echo "# Compiled Program: " ${PROG} "    #"
	@echo "# -------------------------------- #"
	${RM} ${OBJECTS}

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
	@rm -rf ${PROG} *.vtr
