CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
OBJECTS  = prealloc_helper.o input.o consteq.o physics.o bc.o advect.o residual.o solver.o initialize.o utils.o output.o benchmark.o

all:
	@echo "# -------------------------------- #"
	@echo "# Compiled Programs                #"
	@echo "# -------------------------------- #"
	@make stagridge
	@make stagridge_advdiff
	${RM} ${OBJECTS} stagridge.o stagridge_advdiff.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

stagridge: stagridge.o ${OBJECTS} chkopts
	${RM} stagridge
	-${CLINKER} -o stagridge stagridge.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

stagridge_advdiff: stagridge_advdiff.o ${OBJECTS} chkopts
	${RM} stagridge_advdiff
	-${CLINKER} -o stagridge_advdiff stagridge_advdiff.o ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf stagridge stagridge_advdiff *.vtr *.dSYM
