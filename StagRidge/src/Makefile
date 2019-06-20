CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
PROG     = stagridge
OBJECTS  = ${PROG}.o prealloc_helper.o

all: ${PROG}
	${RM} ${OBJECTS}

debug: ${PROG}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

${PROG}: ${OBJECTS} chkopts
	${RM} ${PROG}
	-${CLINKER} -o ${PROG} ${OBJECTS} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}