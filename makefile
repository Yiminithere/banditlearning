PDFLATEX=pdflatex
BIBTEX=biber
PROJECT=stochastic-bandits

all:
	$(PDFLATEX) $(PROJECT)
	$(BIBTEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)
