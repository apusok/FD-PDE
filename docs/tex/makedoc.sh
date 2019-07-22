#!/bin/bash    

# Creates pdf files
# A.E. Pusok

# Compile images
if [ -z "$INKSCAPE" ]
then
  echo "Cannot compile images because no INKSCAPE package was detected!"
else
  $INKSCAPE --file=../img/dmstag.svg --without-gui --export-pdf=../img/dmstag.pdf
  $INKSCAPE --file=../img/dmstag_snes.svg --without-gui --export-pdf=../img/dmstag_snes.pdf
  $INKSCAPE --file=../img/corner_flow.svg --without-gui --export-pdf=../img/corner_flow.pdf
  echo "Compiled pdf images with inkscape"
fi

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
