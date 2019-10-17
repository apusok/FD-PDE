
include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

SRC := 	dmstagbclist.c \
	dmstagoutput.c \
	prealloc_helper.c \
	fdpde.c \
	fdpde_stokes_residual.c fdpde_stokes.c \
	fdpde_advdiff_residual.c fdpde_advdiff.c

HDR := 	benchmark_solcx.h \
	benchmark_cornerflow.h \
	dmstagoutput.h \
	dmstagbclist.h \
	prealloc_helper.h \
	fdpde.h \
	fdpde_stokes.h \
	fdpde_advdiff.h


OBJ := $(patsubst %.c,%.o,$(SRC))

TEST_SRC = 	tests/test_stokes_solcx.c \
		tests/test_stokes_solcx_vargrid.c \
		tests/test_stokes_mor.c \
		tests/test_advdiff_laplace.c \
		tests/test_dmstagbclist.c \
		tests/test_dmstagoutput.c \
		tests/test_fdpde.c \

TEST_OBJ := $(patsubst %.c,%.o,$(TEST_SRC))

TEST_APP := $(patsubst %.c,%.app,$(TEST_SRC))

all: $(OBJ)

tests: $(OBJ) $(TEST_APP)


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
	@rm -rf *.dSYM ./tests/*.dSYM *.py *.vtr *.pbin *.info
	@rm -rf ./tests/python/out_* ./tests/python/__pycache__
