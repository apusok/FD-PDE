include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

FDPDE_DIR = .

TEST_DIR = ../new_tests
PYTHON_DIR = ../new_tests/python

SRC := 	fdpde_snes.c \
	fdpde_dmstag.c \
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

HDR := 	benchmark_solcx.h \
	benchmark_cornerflow.h \
	fdpde_snes.h \
	fdpde_dmstag.h \
	fdpde_utils.h \
	fdpde.h \
	fdpde_stokes.h \
	fdpde_stokesdarcy2field.h \
	fdpde_stokesdarcy3field.h \
	fdpde_advdiff.h \

FDPDE_SRC = $(patsubst %,${FDPDE_DIR}/%,${SRC})
FDPDE_HDR = $(patsubst %,${FDPDE_DIR}/%,${HDR})

FDPDE_OBJ := $(patsubst %.c,%.o,${FDPDE_SRC})

TEST_SRC = 	${TEST_DIR}/test_stokesdarcy2field_plume.c \

# TEST_SRC = 	${TEST_DIR}/test_fdpde.c \
	${TEST_DIR}/test_dmstagbclist.c \
	${TEST_DIR}/test_dmstagoutput.c \
	${TEST_DIR}/test_dmstagoutput_read.c \
	${TEST_DIR}/test_stokes_solcx.c \
	${TEST_DIR}/test_stokes_mor.c \
	${TEST_DIR}/test_stokes_solcx_vargrid.c \
	${TEST_DIR}/test_stokes_lid_driven.c \
	${TEST_DIR}/test_trueboundary.c \
	${TEST_DIR}/test_stokesdarcy2field_mms_katz_ch13.c \
	${TEST_DIR}/test_stokesdarcy2field_rhebergen-siam-2014.c \
	${TEST_DIR}/test_stokesdarcy2field_mms_compare_nd.c \
	${TEST_DIR}/test_stokesdarcy3field_mms_bulkviscosity.c \
	${TEST_DIR}/test_advdiff_laplace.c \
	${TEST_DIR}/test_advdiff_elman.c \
	${TEST_DIR}/test_advdiff_advtime.c \
	${TEST_DIR}/test_advdiff_mms_2d_diffusion.c \
	${TEST_DIR}/test_advdiff_mms_convergence.c \
	${TEST_DIR}/test_advdiff_periodic.c \
	${TEST_DIR}/test_stokesdarcy2field_mms_porosity.c \
	${TEST_DIR}/test_stokesdarcy2field_beam.c \
	${TEST_DIR}/test_stokesdarcy2field_plume.c \

TEST_OBJ := $(patsubst %.c,%.o,${TEST_SRC})
TEST_APP := $(patsubst %.c,%,${TEST_SRC})

all: ${FDPDE_OBJ}
	@echo "# -------------------------------- #"
	@echo "# Compiled src: " ${FDPDE_OBJ} "    "
	@echo "# -------------------------------- #"

tests: ${TEST_APP}
	@echo "# -------------------------------- #"
	@echo "# Compiled tests: " ${TEST_APP} "   "
	@echo "# -------------------------------- #"
	${RM} ${TEST_OBJ} $(FDPDE_OBJ)

%.o: %.c $(FDPDE_HDR)
	-$(PCC) $(CFLAGS) -O2 -o $@ -c $< $(PETSC_CC_INCLUDES)

${TEST_APP}: ${TEST_OBJ} ${FDPDE_OBJ}
	${RM} ${TEST_APP}
	-${CLINKER} -o ${TEST_APP} ${TEST_OBJ} ${FDPDE_OBJ} ${PETSC_SNES_LIB} ${PETSC_CONTRIB}

clean_tests:
	@rm -rf $(TEST_OBJ)
	@rm -rf $(TEST_APP)

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf $(FDPDE_OBJ)
	@rm -rf $(TEST_OBJ)
	@rm -rf $(TEST_APP)
	@rm -rf *.dSYM $(TEST_DIR)/*.dSYM *.py *.vtr *.pbin *.info *.out $(TEST_DIR)/*.out $(PYTHON_DIR)/*.out
	@rm -rf out_* $(TEST_DIR)/out_* $(PYTHON_DIR)/out_*
	@rm -rf snes_failure* $(TEST_DIR)/snes_failure* $(PYTHON_DIR)/snes_failure*
	@rm -rf enthalpy_failure* $(TEST_DIR)/enthalpy_failure* $(PYTHON_DIR)/enthalpy_failure*
	@rm -rf __pycache__ $(TEST_DIR)/__pycache__ $(PYTHON_DIR)/__pycache__
