#====================================================

# Environment required for documentation 
# Define paths to most current pdflatex and bibtex implementation

PDFLATEX = $(shell which pdflatex)
BIBTEX = $(shell which bibtex)

#====================================================
# create pdf manuscript
docs:
ifneq "$(PDFLATEX)" ""
	PDFLATEX=$(PDFLATEX) BIBTEX=$(BIBTEX) ./makedoc.sh
	@echo "............................................."
	@echo "  Created: ./equations.pdf   "
	@echo "............................................."
else
	@echo "Cannot compile file because no TEX package was detected!"
endif

#====================================================

