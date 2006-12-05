.PHONY: all clean distclean doxygen manual 

DOCFILES := introduction.txt license.txt installation.txt interfaces.txt cmdline.txt matlab.txt octave.txt python.txt R.txt commands.txt  faq.txt

all:	manual.pdf manual.html
		
manual.pdf: manual.txt manual.tex $(DOCFILES) 
	pdflatex manual.tex
manual.html: manual.txt manual.tex $(DOCFILES)
	rst2html manual.txt manual.html
manual.tex: manual.txt $(DOCFILES)
	rst2latex manual.txt manual.tex
manual.txt: $(DOCFILES)
			cat $^ >$@

distclean: clean

clean:
	rm -f manual.{out,aux,log,pdf,html,txt,tex} texput.log
