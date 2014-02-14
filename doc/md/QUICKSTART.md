# SHOGUN-TOOLBOX Quickstart

This quickstart assumes that you have access to standard unix/linux tools,
cmake and a C/C++ compiler.  It may be neccessary to install additional
libraries or header files to compile shogun or its interfaces.

## Build prerequisites

* Minimal requirements: standard utils like cmake, gcc/g++/clang, ldd,
  wget/curl, tar/bzip2, bash, grep, test, sed, cut, awk, ldd, uname, cat,
  python-2.7
* Optional libraries to improve performance: lapack3-dev, atlas3-headers,
  atlas3-base-dev, libeigen3-dev
* Depending on the enabled interfaces you may need: swig 2, r-base-dev, 
  liboctave-dev, openjdk-6-jdk/openjdk-7-jdk, jblas, jblas-dev,
  python2.7-dev, python-numpy

## Download sources

The following commands will get the prepared shogun source archives.  If
you choose to fetch "shogun-data" as well, approximately 250 MB of data
will be downloaded.

```

$ wget ftp://shogun-toolbox.org/shogun/releases/3.1/sources/shogun-3.1.1.tar.bz2
$ tar xjf shogun-3.1.1.tar.bz2

$ wget ftp://shogun-toolbox.org/shogun/data/shogun-data-0.7.tar.bz2
$ shogun-data-0.7.tar.bz2

$ rm data/
$ ln -s ../shogun-data-0.7 data
```

## Compile and install SHOGUN-TOOLBOX into home directory

```

$ cd shogun-3.1.1

$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX="/home/$USER/shogun-install" ..

$ make -j5 all
$ make install
```

## Run the examples

Many toy examples on can be found within `share/shogun/examples/libshogun`.
In order to run them, you need to point `LD_LIBRARY_PATH` to the location
of the compiled library.  If everything from above was successful, this
should work well:

```

$ export LD_LIBRARY_PATH="/home/$USER/shogun-install/lib:$LD_LIBRARY_PATH"
$ cd "/home/$USER/share/shogun/examples/libshogun"
$ chmod +x ./so_multiclass_BMRM && ./so_multiclass_BMRM
```

# You know what you're doing?

A small cheat sheet of available cmake options.  This list does not
claim to be comprehensive -- it's meant to be a quick reference for
those you know what they do.

## Enabling modular interfaces
* `-DPythonModular=ON`, `-DOctaveModular=ON`, `-DJavaModular=ON`,
* `-DPerlModular=ON`, `-DRubyModular=ON`, `-DCSharpModular=ON`,
* `-DRModular=ON`, `-DLuaModular=ON`

## Enabling static interfaces (legacy)
* `-DCmdLineStatic=ON`, `-DPythonStatic=ON`, `-DOctaveStatic=ON`
* `-DMatlabStatic=ON`, `-DRStatic=ON`

## Handy cmake options
* `-DCMAKE_BUILD_TYPE=Debug` or `-DCMAKE_BUILD_TYPE=Release`
* `-DENABLE_TESTING=ON` or `-DENABLE_TESTING=OFF`
* `-DCMAKE_INCLUDE_PATH=...`
* `-DCMAKE_LIBRARY_PATH=...`

# Got stuck? Found a bug? Need help?

* Bug tracker: https://github.com/shogun-toolbox/shogun/issues
* IRC chat: Channel #shogun at irc.freenode.net
* Mailing list: shogun-list-subscribe@shogun-toolbox.org
