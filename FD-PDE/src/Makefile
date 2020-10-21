
include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

TEST_DIR = ./tests
PYTHON_DIR = ./tests/python

SRC := 	dmstagbclist.c \
	dmstagoutput.c \
	prealloc_helper.c \
	consteq.c \
	fdpde.c \
	fdpde_stokes_residual.c fdpde_stokes.c \
	fdpde_stokesdarcy2field_residual.c fdpde_stokesdarcy2field.c \
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

OBJ := $(patsubst %.c,%.o,$(SRC))

TEST_SRC = 	tests/test_enthalpy.c \
		tests/test_enthalpy_1d_solidification.c \

# TEST_SRC = 	tests/test_stokes_solcx.c \
# 		tests/test_stokes_solcx_vargrid.c \
# 		tests/test_stokes_mor.c \
# 		tests/test_stokesdarcy2field_rhebergen-siam-2014.c \
# 		tests/test_stokesdarcy2field_mms_katz_ch13.c \
# 		tests/test_stokesdarcy2field_mms_compare_nd.c \
# 		tests/test_stokesdarcy2field_mms_porosity.c \
# 		tests/test_advdiff_laplace.c \
# 		tests/test_advdiff_elman.c \
# 		tests/test_advdiff_advtime.c \
# 		tests/test_advdiff_mms_2d_diffusion.c \
# 		tests/test_advdiff_mms_convergence.c \
# 		tests/test_effvisc_mms.c \
# 		tests/test_dmstagbclist.c \
# 		tests/test_dmstagoutput.c \
# 		tests/test_fdpde.c \
# 		tests/test_composite_laplace.c \
# 		tests/test_decoupled_convection.c \
# 		tests/test_coupling.c \
# 		tests/test_material_point.c \
# 		tests/test_stokes_rt.c \
# 		tests/test_plastic_indenter.c \
# 		tests/test_vp_inclusion_gerya.c \
# 		tests/test_enthalpy.c \
# 		tests/test_enthalpy_1d_solidification.c \

TEST_OBJ := $(patsubst %.c,%.o,$(TEST_SRC))
TEST_APP := $(patsubst %.c,%.app,$(TEST_SRC))

MODEL_SRC = 	../models/mor/mor_mechanics.c \

MODEL_OBJ := $(patsubst %.c,%.o,$(MODEL_SRC))
MODEL_APP := $(patsubst %.c,%.app,$(MODEL_SRC))

all: $(OBJ)

tests: $(OBJ) $(TEST_APP)

models: $(OBJ) $(MODEL_APP)

%.o: %.c $(HDR)
	-$(PCC) $(CFLAGS) -o $@ -c $< $(PETSC_CC_INCLUDES)


%.app: %.c $(OBJ) $(HDR)
	-$(CLINKER) -o $@ $< $(OBJ) $(PETSC_SNES_LIB) $(PETSC_CONTRIB) $(PETSC_CC_INCLUDES)

clean_tests:
	@rm -rf $(TEST_OBJ)
	@rm -rf $(TEST_APP)

clean_all:
	@echo "# -------------------------------- #"
	@echo "#            Full clean            #"
	@echo "# -------------------------------- #"
	@rm -rf $(OBJ)
	@rm -rf $(TEST_OBJ)
	@rm -rf $(TEST_APP)
	@rm -rf $(MODEL_OBJ)
	@rm -rf $(MODEL_APP)
	@rm -rf *.dSYM $(TEST_DIR)/*.dSYM *.py *.vtr *.pbin *.info
	@rm -rf out_* $(TEST_DIR)/out_* $(PYTHON_DIR)/out_*
	@rm -rf snes_failure* $(TEST_DIR)/snes_failure* $(PYTHON_DIR)/snes_failure*
	@rm -rf __pycache__ $(TEST_DIR)/__pycache__ $(PYTHON_DIR)/__pycache__
