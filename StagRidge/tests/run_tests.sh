#!/bin/bash    

# Create output directory if necessary
mkdir -p ./logfiles
mkdir -p ./output

# Run python tests
python ./python/test_solcx.py
python ./python/test_solcx_etaeff.py
python ./python/test_cornerflow_mor.py

python ./python/test_laplace.py
python ./python/test_advdiff.py