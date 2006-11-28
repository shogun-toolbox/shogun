.PHONY: all doc clean distclean doxygen manual 

all:	doc

doc:	doxygen manual

manual: 
	make -C manual

doxygen:
	cd doxygen && doxygen

distclean: clean

clean: 
	rm -rf doxygen/html doxygen/latex
	make -C manual clean
