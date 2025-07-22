include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

FDPDE_DIR = ../../src

PROG = morfault
OBJECTS  = ${PROG}.o

SRC := 	fdpde_snes.c \
	fdpde_dmstag.c \
	fdpde_dmswarm.c \
	fdpde_utils.c \
	fdpde.c \
	fdpde_stokes.c \
	fdpde_stokes_residual.c \
	fdpde_stokesdarcy2field.c \
	fdpde_stokesdarcy2field_residual.c \
	fdpde_stokesdarcy3field.c \
	fdpde_stokesdarcy3field_residual.c \
	fdpde_advdiff_advect.c \
	fdpde_advdiff_residual.c \
	fdpde_advdiff.c \
	fdpde_enthalpy.c \
	fdpde_enthalpy_residual.c \
	fdpde_composite.c \

HDR := 	benchmark_solcx.h \
	benchmark_cornerflow.h \
	fdpde_snes.h \
	fdpde_dmstag.h \
	fdpde_dmswarm.h \
	fdpde_utils.h \
	fdpde.h \
	fdpde_stokes.h \
	fdpde_stokesdarcy2field.h \
	fdpde_stokesdarcy3field.h \
	fdpde_advdiff.h \
	fdpde_enthalpy.h \
	fdpde_composite.h \

FDPDE_SRC = $(patsubst %,$(FDPDE_DIR)/%,$(SRC))
FDPDE_HDR = $(patsubst %,$(FDPDE_DIR)/%,$(HDR))

FDPDE_OBJ := $(patsubst %.c,%.o,$(FDPDE_SRC))

MODEL_OBJ = consteq.o utils.o input.o physics.o bc.o 

all: ${PROG}
	@echo "# -------------------------------- #"
	@echo "# Compiled Program: " ${PROG} "    #"
	@echo "# -------------------------------- #"
	${RM} ${OBJECTS} $(FDPDE_OBJ) ${MODEL_OBJ}

%.o: %.c $(FDPDE_HDR)
	-$(PCC) $(CFLAGS) -O2 -o $@ -c $< $(PETSC_CC_INCLUDES)

${PROG}: ${OBJECTS} $(FDPDE_OBJ) ${MODEL_OBJ}
	${RM} ${PROG}
	-${CLINKER} -o ${PROG} ${OBJECTS} ${FDPDE_OBJ} ${MODEL_OBJ} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf $(FDPDE_OBJ) $(MODEL_OBJ) ${PROG}
