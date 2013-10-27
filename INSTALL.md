GENERAL
=======

Starting from the version 3.0 Shogun uses CMake to facilitate
the building process. When using command line on Linux- and Unix-based
systems with the `make` being available the building steps are:

1) go to the shogun repository root
2) do `mkdir build`
3) do `cmake [options] ..` (or `ccmake ..` if available). It is very
recommended to use any of CMake GUIs (such as ccmake) if you feel unsure
about possible parameters and configurations.
4) do `make` (and `sudo make install` if needed)

In case you want to generate some IDE project (e.g. Eclipse CDT4 project)
use the `-G generator-name` key. You may obtain possible generators with
the `cmake --help` command. For example to generate Eclipse CDT4 project
for Shogun use the `cmake -G "Eclipse CDT4 - Unix Makefiles"`.

Sometimes you would need to clean up your build (e.g. in case of some major
changes). The easiest way to do that is straightforward:
just remove the `build` directory you created before.

If you prefer to not run the `make install` command, you should
instead include the shogun library in your path:

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:path_to_shogun/src/shogun/

Often you are just interested in one language and we always recommend to
use the more powerful modular interfaces.

SPECIAL FEATURES

To enable Multiple Kernel Learning with CPLEX(tm) just make sure cplex can
be found in the PATH. If it is not found shogun will resort to GLPK (if found)
for 1-norm MKL, p-norm MKL with p>1 will work nonetheless.

REQUIREMENTS

The standard linux utils like bash, grep, test, sed, cut, awk, ldd, uname gcc
g++ and cat, python (debian package: python2.7) are required
for the cmake to work.

To compile the R interface you need to have the R developer files
(debian package r-base-dev) installed.

To compile the octave interface you need to have the octave developer files
(debian package liboctave-dev) installed.

To compile the java interface you need to have the java developer files
(debian package openjdk-6-jdk or openjdk-7-jdk) installed.

To compile the python interface you need to have the python developer files
installed (debian packages python2.7-dev or python3.X-dev) and numpy
version 1.x installed (debian package python-numpy) installed.

Optionally you will need atlas, lapack and eigen3 (debian packages lapack3-dev,
atlas3-headers atlas3-base-dev, libeigen3-dev) installed. Note that
atlas/lapack is only supported under linux and osx (high performance computing
should be done under linux only anyway). In case atlas/lapack is unavailable,
don't worry most of shogun will work without, though slightly slower versions
are used. For standard 1-norm multiple kernel learning (MKL) the GNU Linear
Programming Kit (GLPK) version at least 4.29 or CPLEX is required. If you want
to build the html documentation or python online help you will need doxygen
version 1.6.0 or higher.

For the interfaces to compile you will need swig.


SPECIFIC BUILD INSTRUCTIONS FOR MODULAR INTERFACES

object oriented python/swig interface:
======================================
  mkdir build && cd build
  cmake -DPythonModular=ON ..
  make
  sudo make install

to test if it is working try
```
  $ export LD_LIBRARY_PATH=SHOGUN_INSTALL_DIR/lib
  $ export PYTHONPATH=SHOGUN_INSTALL_DIR/lib/pythonX.Y/dist-packages/
  $ python examples/undocumented/python_modular/graphical/svm.py
```

object oriented octave/swig interface:
======================================

do all of the above you did for octave but now in addition install the swig
package and configure+compile shogun with:

  mkdir build && cd build
  cmake -DOctaveModular=ON ..
  make
  sudo make install

to test if it is working try octave examples/documented/octave_modular/libsvm.m

object oriented r/swig interface:
======================================

note that this interface is pre-alpha quality

  mkdir build && cd build
  cmake -DRModular=ON ..
  make
  sudo make install

to test if it is working try R  examples/documented/r_modular/all_classifier.R

object oriented java/swig interface:
======================================

  mkdir build && cd build
  cmake -DJavaModular=ON ..
  make
  make install

to test if it is working try
```
  $ export CLASSPATH=/usr/share/java/jblas.jar:SHOGUNDIR/src/java_modular/shogun.jar:.
  $ export LD_LIBRARY_PATH=SHOGUNDIR/src/shogun:SHOGUNDIR/src/java_modular
  $ javac ../examples/udocumented/java_modular/classifier_libsvm_minimal_modular.java
  $ java classifier_libsvm_minimal_modular
```

SPECIFIC BUILD INSTRUCTIONS FOR LEGACY STATIC INTERFACES

standalone:
===========

mkdir build && cd build
cmake -DCmdLineStatic=ON ..
make

a shogun executable can be found in interfaces/cmdline_static

In order to test the shogun standalone binary, you can run the following
commands from the project root directory:

cd examples/documented/cmdline_static
../../../src/interfaces/cmdline_static/shogun classifier_liblinear.sg

octave
======

To compile the octave interface you need to have the octave developer files
(debian package liboctave-dev).

then do a

mkdir build && cd build
cmake -DOctaveStatic=ON ..
make
sudo make install

a sg.oct file should be created. as a test start octave in the
examples/documented/octave_static/ directory and type

addpath('../../../src/interfaces/octave_static/graphical')
svr_regression

matlab
======

To compile the matlab interface you need to have matlab installed in the path
(i.e., typing matlab in the shell should start matlab).

then do a

mkdir build && cd build
cmake -DMatlabStatic=ON ..
make
sudo make install

a sg.mexglx (or sg.mexa64 or sg.mexmac etc file should be created in
src/interfaces/matlab_static/). As a test start matlab in the
examples/documented/matlab_static directory and type

addpath('../../../src/interfaces/matlab_static/graphical')
svr_regression

For permanent use you could add the following line to your matlab/startup.m

    addpath('path_to_shogun/src/interfaces/matlab_static');

R
=

To compile the R interface you need to have the R developer files
(debian package r-base-dev) installed.

then do the usual

mkdir build && cd build
cmake -DRStatic=ON ..
make
sudo make install

python
======

To compile the python interface you need to have numpy version 1.x installed
(debian package python-numpy) and optionally for plotting
python-matplotlib installed.

mkdir build && cd build
cmake -DPythonStatic=ON ..
make
sudo make install

A sg.so file should be created in the src/interfaces/python_static directory:
To test whether it is working change to examples/documented/python_static/graphical
and run:

PYTHONPATH=path_to_shogun/src/interfaces/python_static/ python svm_classification.py

eierlegendewollmichsau (elwms) interface
========================================

This is a .so file that works with R,python,matlab,octave all in one. To compile
you should have at least python and some other interface enabled:

mkdir build && cd build
cmake -DElwmsStatic=ON ..
make
sudo make install


cd src/interfaces/elwms_static
LD_LIBRARY_PATH=/path/to/octave/lib:/path/to/matlab/libs octave

All examples from
examples/documented/{r_static,python_static,matlab_static,octave_static}/*
should work plus the ones in examples/documented/elwms_static/
(that allows lang -> python subcommands).

PROBLEMS

In case header files or libraries are not at standard locations one needs
to manually adjust the libray/include paths. This can be done with
-DCMAKE_INCLUDE_PATH=/my/include/path (for includes) and -DCMAKE_LIBRARY_PATH=/my/library/path .
A good reference for that is http://cmake.org/Wiki/CMake_Useful_Variables .
