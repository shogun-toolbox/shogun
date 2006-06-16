#
# Makefile for automated Shogun-R package generation
#
#
SHOGUN:=sg_0.1-1.tar.gz

install: $(SHOGUN)
	@ R CMD INSTALL $(SHOGUN)
#@ ~/projects/R.debug/R-2.3.0/bin/R CMD INSTALL --library=/tmp/myRLib/ $(SHOGUN)
$(SHOGUN): configure
	@ echo "Copy all needed files from Shogun src"
	@ cp configure ./sg/src
	@ cp -R ../src/classifier ../src/features ../src/gui ../src/guilib ../src/kernel ../src/lib ../src/preproc ../src/regression ../src/distributions ./sg/src
	@ cp ../src/.generate_link_dependencies.py ../src/.version.sh ./sg/src
	@ cat ../src/Makefile | sed 's/#removeconfighunk/rm -f .config/' >./sg/src/Makefile
	@ cp ../src/configure ./sg/src/configure.sg
	@ cp ../src/AUTHORS ../src/CONTRIBUTIONS ../src/ChangeLog ../src/INSTALL ../src/LICENSE ../src/LICENSE.SVMlight ../src/README ../src/README.developer ../src/TODO ./sg/inst/doc
	@ chmod +x ./sg/src/configure.sg ./sg/src/configure ./sg/src/.version.sh
	@ rm `find ./sg -iname *.i -o -iname *.o`
	@ echo "Compile package..."
	@ R CMD build sg
	@ echo "Check package..."
	@ R CMD check sg

clean: 
	@ rm -rf sg/src/* sg.Rcheck/ $(SHOGUN)
