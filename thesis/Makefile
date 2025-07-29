MAIN=main
CHAPTERS=chapters/*.tex

all:
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

chapters:
	latexmk -pdf -pvc -silent $(CHAPTERS)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls