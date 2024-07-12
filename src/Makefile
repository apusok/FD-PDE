
include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

TEST_DIR = ../tests
PYTHON_DIR = ../tests/python

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

OBJ := $(patsubst %.c,%.o,$(SRC))

TEST_SRC = 	$(TEST_DIR)/test_stokes_solcx.c \
		$(TEST_DIR)/test_stokes_solcx_vargrid.c \
		$(TEST_DIR)/test_stokes_mor.c \
		$(TEST_DIR)/test_stokesdarcy2field_rhebergen-siam-2014.c \
		$(TEST_DIR)/test_stokesdarcy2field_mms_katz_ch13.c \
		$(TEST_DIR)/test_stokesdarcy2field_mms_compare_nd.c \
		$(TEST_DIR)/test_stokesdarcy2field_mms_porosity.c \
		$(TEST_DIR)/test_stokesdarcy3field_mms_bulkviscosity.c \
		$(TEST_DIR)/test_advdiff_laplace.c \
		$(TEST_DIR)/test_advdiff_elman.c \
		$(TEST_DIR)/test_advdiff_advtime.c \
		$(TEST_DIR)/test_advdiff_mms_2d_diffusion.c \
		$(TEST_DIR)/test_advdiff_mms_convergence.c \
		$(TEST_DIR)/test_effvisc_mms.c \
		$(TEST_DIR)/test_dmstagbclist.c \
		$(TEST_DIR)/test_dmstagoutput.c \
		$(TEST_DIR)/test_dmstagoutput_read.c \
		$(TEST_DIR)/test_fdpde.c \
		$(TEST_DIR)/test_composite_laplace.c \
		$(TEST_DIR)/test_decoupled_convection.c \
		$(TEST_DIR)/test_coupling.c \
		$(TEST_DIR)/test_material_point.c \
		$(TEST_DIR)/test_stokes_rt.c \
		$(TEST_DIR)/test_plastic_indenter.c \
		$(TEST_DIR)/test_vp_inclusion_gerya.c \
		$(TEST_DIR)/test_stokes_lid_driven.c \
		$(TEST_DIR)/test_trueboundary.c \
		$(TEST_DIR)/test_enthalpy_2d_diffusion.c \
		$(TEST_DIR)/test_enthalpy_1d_eutectic_solidification.c \
		$(TEST_DIR)/test_enthalpy_1d_solidification_TC.c \
		$(TEST_DIR)/test_advdiff_periodic.c \
		$(TEST_DIR)/test_enthalpy_periodic.c \
		$(TEST_DIR)/test_convection_stokes_periodic.c \
		$(TEST_DIR)/test_stokesdarcy2field_vp_inclusion.c \
		$(TEST_DIR)/test_stokesdarcy2field_vep_0d_shear.c \
		$(TEST_DIR)/test_stokesdarcy2field_vep_0d_expansion.c \
		$(TEST_DIR)/test_stokesdarcy2field_vep_inclusion.c \
		$(TEST_DIR)/test_dmstag_phasemethod.c \
		$(TEST_DIR)/test_stokesdarcy2field_rt.c \
		$(TEST_DIR)/test_stokesdarcy2field_rt2.c \
		$(TEST_DIR)/test_stokesdarcy2field_beam.c \
		$(TEST_DIR)/test_stokesdarcy2field_plume.c \
		$(TEST_DIR)/test_stokes_rt_compare_pic_phasefield.c \
		$(TEST_DIR)/test_stokesdarcy2field_vep_inclusion_pic.c \
		$(TEST_DIR)/test_dmswarm_project.c \

TEST_OBJ := $(patsubst %.c,%.o,$(TEST_SRC))
TEST_APP := $(patsubst %.c,%,$(TEST_SRC))

all: $(OBJ)

tests: $(OBJ) $(TEST_APP)

%.o: %.c $(HDR)
	-$(PCC) $(CFLAGS) -O2 -o $@ -c $< $(PETSC_CC_INCLUDES)


%: %.c $(OBJ) $(HDR)
	-$(CLINKER) -O2 -o $@ $< $(OBJ) $(PETSC_SNES_LIB) $(PETSC_CONTRIB) $(PETSC_CC_INCLUDES)

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
	@rm -rf *.dSYM $(TEST_DIR)/*.dSYM *.py *.vtr *.pbin *.info *.out $(TEST_DIR)/*.out $(PYTHON_DIR)/*.out
	@rm -rf out_* $(TEST_DIR)/out_* $(PYTHON_DIR)/out_*
	@rm -rf snes_failure* $(TEST_DIR)/snes_failure* $(PYTHON_DIR)/snes_failure*
	@rm -rf enthalpy_failure* $(TEST_DIR)/enthalpy_failure* $(PYTHON_DIR)/enthalpy_failure*
	@rm -rf __pycache__ $(TEST_DIR)/__pycache__ $(PYTHON_DIR)/__pycache__
