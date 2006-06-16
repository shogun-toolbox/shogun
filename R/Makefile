# Makefile for automated Shogun-R package generation
#
#
install: sg_0.1-1.tar.gz
	@ R CMD INSTALL sg_0.1-1.tar.gz 
#@ ~/projects/R.debug/R-2.3.0/bin/R CMD INSTALL --library=/tmp/myRLib/ sg_0.1.tar.gz 

sg_0.1-1.tar.gz: configure
	@ echo "Copy all needed files from Shogun src"
	@ cp configure ./sg/src
	@ cp -R ../src/classifier ../src/features ../src/gui ../src/guilib ../src/kernel ../src/lib ../src/preproc ../src/regression ../src/distributions ./sg/src
	@ cp ../src/Makefile ../src/.generate_link_dependencies.py ../src/.version.sh ./sg/src
	@ cp ../src/configure ./sg/src/configure.sg
	@ chmod +x ./sg/src/configure.sg ./sg/src/configure
	@ rm `find ./sg -iname *.i -o -iname *.o`
	@ echo "Compile package..."
	@ R CMD build sg
	@ echo "Check package..."
	@ R CMD check sg

clean: 
	@ rm -rf sg/src/* sg.Rcheck/ sg_0.1-1.tar.gz
