LATEX=lualatex
BIBER=biber
REPORT_NAME=cosc6342-hw2-michael-yantosca
all: $(REPORT_NAME).pdf

%.pdf: %.tex
	@$(LATEX) -shell-escape $*.tex
	@$(BIBER) --input-directory src $*
	@$(LATEX) -shell-escape $*.tex

superclean: clean superclean-doc-$(REPORT_NAME)

clean: clean-doc-$(REPORT_NAME)
	@rm -f *~

superclean-doc-%:
	rm -f $*.pdf

clean-doc-%:
	@rm -f $*.aux
	@rm -f $*.bbl
	@rm -f $*.bcf
	@rm -f $*.log
	@rm -f $*.run.xml
	@rm -f $*.dvi
	@rm -f $*.blg
	@rm -f $*.auxlock
	@rm -f $*.pyg
	@rm -f $*-figure*
	@rm -f $*.toc
	@rm -f $*.out
	@rm -f $*.snm
	@rm -f $*.nav
