#!/bin/bash    

# Creates pdf files
# A.E. Pusok

# Invoke latex and bibtex
$PDFLATEX equations.tex
$BIBTEX equations.aux
$PDFLATEX equations.tex

# For some reason the references show up correctly only if you compile 2nd time after bibtex
$PDFLATEX equations.tex

# Clean up
rm -f *.aux
rm -f *.log
rm -f *.out
rm -f *.gz
rm -f *.toc
rm -f *.bbl
rm -f *.blg
rm -f *.lof
rm -f *.lot
