INSTALL   {#install}
=======
This file explains how to build Shogun from the source-code. We recommend installing a binary package of Shogun, unless your system is not supported or you want to modify the source code. See our website for details Also see [QUICKSTART](https://github.com/shogun-toolbox/shogun/wiki/QUICKSTART) for a more compact version.

##GENERAL
Starting from the version 3.0 Shogun uses CMake to facilitate
the building process. When using command line on Linux- and Unix-based
systems with the `make` being available the building steps are:

1. go to the shogun repository root
2. do `mkdir build`
3. do `cmake [options] ..` (or `ccmake ..` if available). It is very
recommended to use any of CMake GUIs (such as ccmake) if you feel unsure
about possible parameters and configurations.
4. do `make` (and `sudo make install` if needed)

In case you want to generate some IDE project (e.g. Eclipse CDT4 project)
use the `-G generator-name` key. You may obtain possible generators with
the `cmake --help` command. For example to generate Eclipse CDT4 project
for Shogun use the `cmake -G "Eclipse CDT4 - Unix Makefiles"`.

Sometimes you would need to clean up your build (e.g. in case of some major
changes). The easiest way to do that is straightforward:
just remove the `build` directory you created before.

If you prefer to not run the `make install` command, you should
instead include the shogun library in your path:

`export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:path_to_shogun/src/shogun/`

You might have to run `sudo ldconfig` to update your shared library cache. Note that the interfaces require additional variables to be set, see below and [INTERFACES](https://github.com/shogun-toolbox/shogun/wiki/INFERFACES).

##REQUIREMENTS

The standard linux utils like bash, grep, test, sed, cut, awk, ldd, uname gcc
g++ and cat, python (debian package: python2.7) are required
for the cmake to work. [CCache](https://ccache.samba.org/) will massively speed up the compilation process (enabled by default).
To compile the interfaces, in addition to [SWIG](http://www.swig.org/) itself, you will need language specific development packages installed, see below.

There is a larger number of optional requirements, detected through our build system. You can always check the cmake output to find out what was detected. If a particular Shogun class is unavailable, this is likely due to an unmet dependency.

##SPECIFIC BUILD INSTRUCTIONS FOR MODULAR INTERFACES
The cmake targets are `-DPythonModular -DOctaveModular -DRModular -DJavaModular -DRubyModular -DLuaModular -DCSharpModular` etc. To compile for example the Python interface you need to

    $ mkdir build && cd build
    $ cmake -DPythonModular=ON ..
    $ make
    $ sudo make install

Run Shogun examples, see the interface specific instructions at [INTERFACES](https://github.com/shogun-toolbox/shogun/wiki/INFERFACES)

##SPECIAL FEATURES

To enable Multiple Kernel Learning with CPLEX(tm) just make sure cplex can
be found in the PATH. If it is not found shogun will resort to GLPK (if found)
for 1-norm MKL, p-norm MKL with p>1 will work nonetheless.

##MINIMUM REQUIREMENTS TO BUILD SHOGUN FROM SOURCE CODES
You need at least 1 Gigabytes free disk space and 4 Gigabytes RAM to build Shogun from source codes.

The compiler will use a lot of RAM and your computer will be slow if you do not have enough RAM for the compiler.

Again, consider using [ccache](https://ccache.samba.org/).


##PROBLEMS
In case header files or libraries are not at standard locations one needs
to manually adjust the libray/include paths. This can be done with
`-DCMAKE_INCLUDE_PATH=/my/include/path` (for includes) and `-DCMAKE_LIBRARY_PATH=/my/library/path`.
A good reference for that is http://cmake.org/Wiki/CMake_Useful_Variables .

In case you have a problem building Shogun, please open an issue on github with your system details, *exact* commands used, and logs posted as a [gist](https://gist.github.com/).
