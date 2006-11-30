#
# Makefile for automated Shogun-R package generation
#
#
SHOGUN:=sg_0.2-1.tar.gz

package: $(SHOGUN)

install: sg/src
	@ R CMD INSTALL sg

$(SHOGUN): sg/src
	@echo "Create package..."
	( export LC_ALL=C ; R CMD build sg )
	@echo "Check package..."
	R CMD check sg

sg/src:
	@echo "Copy all needed files from Shogun src"
	rm -rf sg/src
	cd ../src && make distclean
	cd ../src && make lib/versionstring.h
	cp -a ../src sg/src
	find ./sg -iname *.i -o -iname *.o -exec rm {} \; 2>/dev/null || true
	find ./sg/src -type d -name .svn -exec rm -rf {} \; 2>/dev/null || true
	rm -f ./sg/src/.config ./sg/src/.depend
	cat ../src/Makefile | sed -e 's/#removeconfighunk/rm -f .config/' -e "/#runconfigurehunkstart/,/#runconfigurehunkend/c\\\t./configure --enable-R && make\n\texit 0"  >./sg/src/Makefile
	cp ../src/AUTHORS ../src/CONTRIBUTIONS ../src/ChangeLog ../src/INSTALL ../src/LICENSE ../src/LICENSE.SVMlight ../src/README ../src/README.developer ../src/TODO ./sg/inst/doc

clean: 
	rm -rf sg/src sg.Rcheck
