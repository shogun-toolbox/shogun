#
# Makefile for automated Shogun-R package generation
#
#
SHOGUN:=sg_0.3-1.tar.gz

install: $(SHOGUN)
	@ R CMD INSTALL $(SHOGUN)
	#@ ~/projects/R.debug/R-2.3.0/bin/R CMD INSTALL --library=/tmp/myRLib/ $(SHOGUN)

$(SHOGUN): configure ./sg/src/configure.sg
	cd sg/src && ./configure
	@echo "Compile package..."
	sh -c "R CMD build sg"
	#@echo "Check package..."
	#sh -c "R CMD check sg"

./sg/src/configure.sg: configure
	@echo "Copy all needed files from Shogun src"
	sh -c "cd ../src && make lib/versionstring.h"
	cp configure ./sg/src
	cp -R ../src/classifier ../src/features ../src/gui ../src/guilib ../src/kernel ../src/lib ../src/preproc ../src/regression ../src/distributions ../src/distance ./sg/src
	cp ../src/.generate_link_dependencies.py ../src/.version.sh ./sg/src
	cat ../src/Makefile | sed 's/#removeconfighunk/rm -f .config/' >./sg/src/Makefile
	cp ../src/configure ./sg/src/configure.sg
	cp ../src/AUTHORS ../src/CONTRIBUTIONS ../src/ChangeLog ../src/INSTALL ../src/LICENSE ../src/LICENSE.SVMlight ../src/README ../src/README.developer ../src/TODO ./sg/inst/doc
	chmod +x ./sg/src/configure.sg ./sg/src/configure ./sg/src/.version.sh
	rm `find ./sg -iname *.i -o -iname *.o`
	find ./sg/src -type d -name .svn -exec rm -rf {} \; 2>/dev/null || true


clean: 
	rm -rf ./sg/src/* sg.Rcheck/ $(SHOGUN)
	rm -f ./sg/inst/doc/AUTHORS ./sg/inst/doc/CONTRIBUTIONS ./sg/inst/doc/ChangeLog ./sg/inst/doc/INSTALL ./sg/inst/doc/LICENSE ./sg/inst/doc/LICENSE.SVMlight ./sg/inst/doc/README ./sg/inst/doc/README.developer ./sg/inst/doc/TODO
	rm -f sg/src/.generate_link_dependencies.py sg/src/.version.sh sg/src/.depend
