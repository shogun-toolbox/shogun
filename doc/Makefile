.PHONY: all clean distclean doxygen manual 

all:	doxygen manual

manual: 
	make -C manual

doxygen:
	doxygen

distclean: clean

clean: 
	rm -f doxygen/*.html doxygen/shogun.tex
	make -C manual clean
