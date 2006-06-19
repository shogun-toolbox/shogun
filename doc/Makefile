all:	doxygen manual/manual.pdf

manual: manual/manual.pdf manual/manual.html

manual/manual.pdf: manual/manual.tex
	cd manual && pdflatex manual.tex
manual/manual.tex: manual/manual.txt
	cd manual && rst2latex manual.txt manual.tex
manual/manual.html: manual/manual.txt
	cd manual && rst2html manual.txt manual.html
doxygen/index.html:
	cd doxygen && doxygen

clean: 
	rm -f doxygen/*.html doxygen/shogun.tex
	rm -f manual/manual.{out,aux,log,pdf,html} manual/texput.log

