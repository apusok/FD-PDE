include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

FDPDE_DIR = ../../src

SRC := 	dmstagbclist.c \
	dmstagoutput.c \
	prealloc_helper.c \
	consteq.c \
	fdpde.c \
	fdpde_stokes_residual.c fdpde_stokes.c \
	fdpde_stokesdarcy2field_residual.c fdpde_stokesdarcy2field.c \
	fdpde_stokesdarcy3field_residual.c fdpde_stokesdarcy3field.c \
	fdpde_advdiff_advect.c fdpde_advdiff_residual.c fdpde_advdiff.c \
	fdpde_enthalpy_residual.c fdpde_enthalpy.c \
	fdpde_composite.c composite_prealloc_utils.c \
	dmstag_utils.c material_point.c \
	snes_picard.c \

HDR := 	benchmark_solcx.h \
	benchmark_cornerflow.h \
	dmstagoutput.h \
	consteq.h \
	dmstagbclist.h \
	prealloc_helper.h \
	fdpde.h \
	fdpde_stokes.h \
	fdpde_advdiff.h \
	fdpde_enthalpy.h \
	fdpde_composite.h composite_prealloc_utils.h \
	dmstag_utils.h material_point.h \

FDPDE_SRC = $(patsubst %,$(FDPDE_DIR)/%,$(SRC))
FDPDE_HDR = $(patsubst %,$(FDPDE_DIR)/%,$(HDR))

FDPDE_OBJ := $(patsubst %.c,%.o,$(FDPDE_SRC))

all:
	@make mor
	${RM} mor_mechanics.o $(FDPDE_OBJ)

%.o: %.c $(FDPDE_HDR)
	-$(PCC) $(CFLAGS) -O2 -o $@ -c $< $(PETSC_CC_INCLUDES)

mor: mor_mechanics.o $(FDPDE_OBJ)
	${RM} mor_mechanics.app
	-${CLINKER} -O2 -o mor_mechanics.app mor_mechanics.o $(FDPDE_OBJ) ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf $(FDPDE_OBJ) $(MODEL_OBJ) mor_mechanics.app out_*
