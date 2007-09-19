#
# Makefile for automated Shogun-R package generation
#
#
SHOGUN:=sg_0.4.2-1.tar.gz

package: $(SHOGUN)

install: sg/src
	R CMD INSTALL sg

$(SHOGUN): sg/src
	@echo "Create package..."
	( export LC_ALL=C ; R CMD build sg )
	@echo "Check package..."
	R CMD check sg

DOCFILES := AUTHORS CONTRIBUTIONS ChangeLog INSTALL LICENSE LICENSE.SVMlight README README.developer TODO

sg/src:
	@echo "Copy all needed files from Shogun src"
	if [ -d .svn ]; then make -C ../src distclean lib/versionstring.h; fi
	rm -rf sg/src
	cp -a ../src sg/src
	find ./sg -iname *.i -o -iname *.o -exec rm {} \; 2>/dev/null || true
	find ./sg/src -type d -name .svn -exec rm -rf {} \; 2>/dev/null || true
	cd sg/src && $(RM) .config .depend
	sed -i -f Makefile.sed sg/src/Makefile
	-cd ../src && cp -a $(DOCFILES) ../R/sg/inst/doc/

debian:
	@echo "Copy all needed files from Shogun src"
	if [ -d .svn ]; then make -C ../src distclean lib/versionstring.h; fi
	rm -rf sg/src
	cp -a ../src sg/src
	find ./sg -iname *.i -o -iname *.o -exec rm {} \; 2>/dev/null || true
	find ./sg/src -type d -name .svn -exec rm -rf {} \; 2>/dev/null || true
	cd sg/src && ( $(RM) .config .depend ; ./configure --interface=R )
	-cd ../src && cp -a $(DOCFILES) ../R/sg/inst/doc/

clean: 
	rm -rf sg/src sg.Rcheck
