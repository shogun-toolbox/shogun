# Installing Shogun

For certain systems, we offer pre-built packages of Shogun.
This is the easiest way to start using it.
For other cases, we describe how to build Shogun from source code.


# Quicklinks
 * [Ready-to-install packages](#binaries)
   - [Anaconda](#anaconda)
   - [Ubuntu](#ubuntu)
   - [Debian](#debian)
   - [Fedora](#fedora)
   - [MacOS](#mac)
   - [Windows](#windows)
 * [Docker images](#docker)
 * [Integration with interface languages](#language)
   - [Python](#pipy)
 * [Compiling manually](#manual)
   - [Requirements](#manual-requirements)
   - [Basics](#manual-basics)
   - [Interfaces](#manual-interfaces)
   - [Examples](#manual-examples)
   - [Problems](#manual-problems)
   - [CMake tips](#manual-cmake)
   - [Customized Python](#manual-python)
   - [Windows](#manual-windows)

## Ready-to-install packages <a name="binaries"></a>

### Anaconda packages <a name="anaconda"></a>
The base shogun library and its Python interface are available through the conda package manager, via <a href="https://conda-forge.org">conda-forge</a>.
To install both:

    conda install -c conda-forge shogun

or to get just the library:

    conda install -c conda-forge shogun-cpp

These packages include most of the optional dependencies and are currently available for Linux, MacOS and Windows.

### Ubuntu ppa <a name="ubuntu"></a>
We are working on integrating Shogun with Debian/Ubuntu.
In the meantime, we offer a [prepackaged ppa](https://launchpad.net/~shogun-toolbox/+archive/ubuntu/stable).
These currently do contain the C++ library and Python bindings.
Add this to your system as

    sudo add-apt-repository ppa:shogun-toolbox/stable
    sudo apt-get update

Then, install as

    sudo apt-get install libshogun18

The Python (2) bindings can be installed as

    sudo apt-get install python-shogun

In addition to the latest stable release, we offer [nightly builds](https://launchpad.net/~shogun-toolbox/+archive/ubuntu/nightly) of our development branch.

### Debian <a name="debian"></a>
Latest packages for Debian `jessie` and `stretch` are available in our own repository at [http://apt.shogun.ml](http://apt.shogun.ml).
We provide both the stable and nightly packages, currently only for amd64 architecture.
In order to add the stable packages to your system, simply run the following commands

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3DD2174ACAB30365
    echo "deb http://apt.shogun.ml/ stretch main" | sudo tee /etc/apt/sources.list.d/shogun-toolbox.list  > /dev/null
    sudo apt-get update

After this just simply install the shogun library

    sudo apt-get install libshogun18

The nightly packages are available in the `nightly` component, i.e. `deb http://apt.shogun.ml/ stretch nightly`

### Fedora <a name="fedora"></a>
Shogun is part of [Fedora 25](https://apps.fedoraproject.org/packages/shogun).
Install as

    sudo dnf install shogun


### MacOS <a name="mac"></a>
Shogun is part of [Homebrew](https://formulae.brew.sh/formula/shogun).
Install the latest stable version as

    brew install shogun

### Windows <a name="windows"></a>
Shogun natively compiles under Windows using MSVC, see the [AppVeyor CI build](https://ci.appveyor.com/project/vigsterkr/shogun) and the [Windows section](#manual-windows). We currently only support binary packages via conda.
If you are interested in packaging, documenting, or contributing otherwise, please contact us.

## Docker images <a name="docker"></a>
You can set up Shogun using our
[Docker images](https://hub.docker.com/r/shogun/shogun/) as:

    sudo docker pull shogun/shogun:master
    sudo docker run -it shogun/shogun:master bash

The docker image follows both the `master` and the `develop` branch of the repository, just specify the desired branch name as tag for the image. For example in order to use the develop version of shogun simply pull the `shogun/shogun:develop` docker image.

There's an [SDK docker image](https://hub.docker.com/r/shogun/shogun-dev/) for shogun development as well, which we use to run our [Travis CI](https://travis-ci.org/shogun-toolbox/shogun/) jobs.

Sometimes mounting a local folder into the docker image is useful.
You can do this via passing an additional option

```
-v /your/local/folder:/same/folder/in/docker
```

See the [Docker documentation](https://docs.docker.com/storage/volumes/) for further details.

## Building shogun - Using vcpkg

You can download and install shogun using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install shogun

The shogun port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

## Integration with interface language build systems <a name="language"></a>
Shogun is can be automatically built from source from the following langauges.

### Python pypi <a name="pypi"></a>
You can install from [pypi](https://pypi.python.org/pypi/shogun-ml/).
There is limited control over options and it might take a long time as everything is done from scratch.

    pip install shogun-ml

We do not recommend this option and suggest to rather compile by hand as described below.


# Compiling manually <a name="manual"></a>

In case none of the binary packages listed on our website work for your system, or you want to modify Shogun, you will need to build it from source.

## Requirements <a name="manual-requirements"></a>
The standard GNU/Linux tools and Python are minimal requirements to compile Shogun.

By default, building the meta examples is enabled, requiring `python-ply` and `ctags`. Disable using `-DBUILD_META_EXAMPLES=Off` if those requirements are a problem, also see [generating meta examples](#manual-examples).

To compile the interfaces, in addition to [swig](http://www.swig.org/) itself, you will need language specific development packages installed, see [interfaces](#manual-interfaces) below.

There is a larger number of optional requirements.
The output of cmake output lists optional dependencies that were found and not found.
If a particular Shogun class is unavailable, this is likely due to an unmet dependency.
See our [docker configuration file](https://github.com/shogun-toolbox/shogun/blob/develop/configs/shogun/Dockerfile) for an example configuration used in our test builds.

You need at least 1GB free disk space. If you compile any interface, roughly 4 GB RAM are need (we are working on reducing this).
[CCache](https://ccache.samba.org/) will massively speed up the compilation process and is enabled by default if installed.

## Basics <a name="manual-basics"></a>
Shogun uses [CMake](https://cmake.org/) for its build. The general workflow is now explained.
For further details on testing etc, see [DEVELOPING.md](DEVELOPING.md).

Download the latest [stable release source code](https://github.com/shogun-toolbox/shogun/releases/latest), or (as demonstrated here) clone the latest develop code.
Potentially update submodules

    git clone https://github.com/shogun-toolbox/shogun.git
    cd shogun
    git submodule update --init

Create the build directory in the source tree root

    mkdir build

Configure cmake, from the build directory, passing the Shogun source root as argument.
It is recommended to use any of CMake GUIs (e.g. replace `cmake ..` with `ccmake ..`), in particular if you feel unsure about possible parameters and configurations.
Note that all cmake options read as `-DOPTION=VALUE`.

    cd build
    cmake [options] ..

Compile

    make


Install (prepend `sudo` if installing system wide), and your are done.

    make install

Sometimes you might need to clean up your build (e.g. in case of some major changes).
First, try

    make clean

If that does not help, try removing the build directory and starting from scratch afterwards

    rm -rf build

If you prefer to not run the `sudo make install` command system wide, you can either install Shogun to a custom location (`-DCMAKE_INSTALL_PREFIX=/custom/path`, defaults to `/usr/local`), or even skip `make install` at all.
In both cases, it is necessary to set a number of system libraries for using Shogun, see [INTERFACES.md](INTERFACES.md).

## Interfaces <a name="manual-interfaces"></a>
The native C++ interface is always included.
The cmake options for building interfaces are `-DINTERFACE_PYTHON=ON -DINTERFACE_R ..` etc.
For example, replace the cmake step above by
```
cmake -DINTERFACE_PYTHON=ON [potentially more options] ..
```

The required packages (here debian/Ubuntu package names) for each interface are

 * Python
   - `python-dev python-numpy`
   - For dealing with customized Python environments, see [here](#manual-python)
 * Octave
   - `octave liboctave-dev`
 * R
   - `r-base-core`
 * Java
   - `oracle-java8-installer`, non-standard, e.g. `https://launchpad.net/~webupd8team/+archive/ubuntu/java`
   - `jblas`, a standard third party library, `https://mikiobraun.github.io/jblas/`
 * Ruby
   - `ruby ruby-dev`, and `narray` a non-standard third party library, `http://masa16.github.io/narray/`, install with `gem install narray`
 * Lua
   - `lua5.1 liblua5.1-0-dev`
 * C-Sharp
   - `mono-devel mono-gmcs cli-common-dev`

To *use* the interfaces, in particular if not installing to the default system-wide location, see [INTERFACES.md](INTERFACES.md).
See [examples](#manual-examples) below for how to create the examples from the website locally.

## Generating examples <a name="manual-examples"></a>
All Shogun examples at our website are automatically generated code. You can
generate them (plus additional ones) locally (needs cmake switch `-DBUILD_META_EXAMPLES=ON`)

    make meta_examples

This requires [PLY for Python](https://pypi.python.org/pypi/ply), package `python-ply`, and [ctags](http://ctags.sourceforge.net/), package `ctags`.
Both source code and potential executables (C++, Java, C-Sharp) are created in `build/examples/meta/` when running `make`.

See [INTERFACES.md](INTERFACES.md) to run the generated examples and see [EXAMPLES.md](EXAMPLES.md) for more details on their mechanics.
See [DEVELOPING.md](DEVELOPING.md) for how the examples are used as tests.

## Problems? Got stuck? Found a bug? Help?  <a name="manual-problems"></a>

In case you have a problem building Shogun, please open an [issue on github](https://github.com/shogun-toolbox/shogun/issues) with your system details, *exact* commands used, and logs posted as a [gist](https://gist.github.com/).

## CMake tips <a name="manual-cmake"></a>
CMake is a beast.
Make sure to read the [docs](https://cmake.org/documentation/) and [CMake_Useful_Variables](http://cmake.org/Wiki/CMake_Useful_Variables).
Make sure to understand the concept of [out of source builds](https://cmake.org/Wiki/CMake_FAQ#Out-of-source_build_trees).
Here are some tips on common options that are useful

Specify a different swig executable:

    cmake -DSWIG_EXECUTABLE=/usr/bin/swig_custom

To specify a different compiler, see [CMake FAQ, "How do I use a different compiler?"](http://www.cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F).
You might have to delete the build directory or clear the cmake cache otherwise for this to work.

    CC=/path/to/gcc CXX=/path/to/g++ cmake ..

In case header files or libraries are not at standard locations one needs
to manually adjust the libray and include paths, `-DCMAKE_INCLUDE_PATH=/my/include/path` and `-DCMAKE_LIBRARY_PATH=/my/library/path`.


## Customized Python environments <a name="manual-python"></a>
Often, there are multiple Python versions installed on the system.
There are various reasons for this, i.e. Linux without root access, MacOS + homebrew, using [Anaconda](https://www.continuum.io/downloads) or [virtualenv](https://virtualenv.pypa.io).
If Shogun is executed using a different Python version that the one it was built against, one will observe crashes when importing Shogun.
If this is your setup, you need to make sure that Shogun is both **built** and **executed** against the Python environment of **your** choice.
For that, you need to do something similar to

    cmake -DPYTHON_INCLUDE_DIR=/path/to/python/include/dir -DPYTHON_LIBRARY=path/to/python/libpythonVERSION.{so|dynlib} -DPYTHON_EXECUTABLE=/path/to/python/executable -DPYTHON_PACKAGES_PATH=/path/to/python/dist-packages ..

For example, for `brew` installed Python under MacOS, use something like:

    cmake -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Headers -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib  -DINTERFACE_PYTHON=ON ..

Under Linux, where you want to use Python 3, which is not the system's default:

    cmake -DPYTHON_INCLUDE_DIR=/usr/include/python3.3 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DPYTHON_PACKAGES_PATH=/usr/local/lib/python3.3/dist-packages -DINTERFACE_PYTHON=ON ..

On a Linux cluster without root access, using [Anaconda](https://www.continuum.io/downloads) (note you will need to activate your environment everytime you want to run Shogun):

    source path/to/anaconda/bin/activate
    cmake -DCMAKE_INSTALL_PREFIX=path/to/shogun/install/dir -DPYTHON_INCLUDE_DIR=path/to/anaconda/include/python2.7/ -DPYTHON_LIBRARY=path/to/anaconda/lib/libpython2.7.so  -DPYTHON_EXECUTABLE=path/to/anaconda/bin/python -DINTERFACE_PYTHON=ON ..

## Windows build <a name="manual-windows"></a>

Please see our [Azure Pipelines](https://dev.azure.com/shogunml/shogun/_build?definitionId=2) build.
It is recommended to use "Visual Studio 16 2019" or "MSBuild".

1. From the Start menu, open the *`[YOUR_ARCHITECTURE]`*`Native Tools Command Prompt for VS`*`[YOUR_VS_VERSION]`*, for example `x64 Native Tools Command Prompt for VS 2019`. It is in the `Visual Studio`*`[YOUR_VS_VERSION]`* folder
2. Execute the following command with the path you need: `set DESTDIR="X:\path\to\install"`, for example:
   ```
   set DESTDIR="D:\Projects\shogun"
   ```
3. Execute line by line:
   ```
   git clone https://github.com/shogun-toolbox/shogun.git %DESTDIR%
   chdir /d %DESTDIR%
   git submodule -q update --init
   ```
4. Specify the generator in cmake to match your IDE (take a look at the *Platform Selection* CMake wiki [section](https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2016%202019.html#platform-selection) or the *Generators* section of the `cmake /?` command output), for example:
   ```
   cmake -S %DESTDIR% -B %DESTDIR%\build -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_META_EXAMPLES=OFF -DENABLE_TESTING=ON
   ```
   The above `cmake` has the following arguments:
   1. `-S %DESTDIR%` specifies the source folder
   2. `-B %DESTDIR%\build` specifies the build folder
   3. `-G "Visual Studio 16 2019" -A x64` specifies the target platform to be x64.   
   4. `-DCMAKE_BUILD_TYPE=Release` specifies a build type and asks compiler to perform optimization and omit debug information.     
   5. `-DBUILD_META_EXAMPLES=OFF` specifies to not generate meta examples.   
   6. `-DENABLE_TESTING=ON` Enable testing while cmake.   
5. Compile:
   ```
   msbuild %DESTDIR%\build\shogun.sln /verbosity:minimal /t:Clean /p:Configuration=Release /p:Platform=x64
   ```
   > Note: If you use /m in msbuild command without specifying the number, it may occur out of memory errors.

<details>
   <summary>Full listing with an output</summary>
   <pre>
      
      **********************************************************************
      ** Visual Studio 2019 Developer Command Prompt v16.4.2
      ** Copyright (c) 2019 Microsoft Corporation
      **********************************************************************
      [vcvarsall.bat] Environment initialized for: 'x64'

      D:\Programs\Microsoft Visual Studio\2019\Community>set DESTDIR="D:\Projects\shogun"

      D:\Programs\Microsoft Visual Studio\2019\Community>git clone https://github.com/shogun-toolbox/shogun.git %DESTDIR%
      Cloning into 'D:\Projects\shogun'...
      remote: Enumerating objects: 160, done.
      remote: Counting objects: 100% (160/160), done.
      remote: Compressing objects: 100% (115/115), done.
      
      Receiving objects: 100% (185763/185763), 68.37 MiB | 10.55 MiB/s, done.
      Resolving deltas: 100% (148537/148537), done.

      D:\Programs\Microsoft Visual Studio\2019\Community>chdir /d %DESTDIR%

      D:\Projects\shogun>git submodule -q update --init

      D:\Projects\shogun>build -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_META_EXAMPLES=OFF -DENABLE_TESTING=ON
      'build' is not recognized as an internal or external command,
      operable program or batch file.

      D:\Projects\shogun>cmake -S %DESTDIR% -B %DESTDIR%\build -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_META_EXAMPLES=OFF -DENABLE_TESTING=ON
      -- Selecting Windows SDK version 10.0.18362.0 to target Windows 10.0.18363.
      -- The C compiler identification is MSVC 19.24.28314.0
      -- The CXX compiler identification is MSVC 19.24.28314.0
      -- Check for working C compiler: D:/Programs/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.24.28314/bin/Hostx64/x64/cl.exe
      -- Check for working C compiler: D:/Programs/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.24.28314/bin/Hostx64/x64/cl.exe -- works
      -- Detecting C compiler ABI info
      -- Detecting C compiler ABI info - done
      -- Detecting C compile features
      -- Detecting C compile features - done
      -- Check for working CXX compiler: D:/Programs/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.24.28314/bin/Hostx64/x64/cl.exe
      -- Check for working CXX compiler: D:/Programs/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.24.28314/bin/Hostx64/x64/cl.exe -- works
      -- Detecting CXX compiler ABI info
      -- Detecting CXX compiler ABI info - done
      -- Detecting CXX compile features
      -- Detecting CXX compile features - done
      -- Performing Test _cpp_latest_flag_supported
      -- Performing Test _cpp_latest_flag_supported - Success
      -- Performing Test HAVE_FOLDING_EXPRESSIONS
      -- Performing Test HAVE_FOLDING_EXPRESSIONS - Success
      -- Performing Test HAVE_IF_CONSTEXPR
      -- Performing Test HAVE_IF_CONSTEXPR - Success
      -- Performing Test HAVE_IF_INIT
      -- Performing Test HAVE_IF_INIT - Success
      -- Performing Test HAVE_STD_STRING_VIEW
      -- Performing Test HAVE_STD_STRING_VIEW - Success
      -- Could NOT find Doxygen (missing: DOXYGEN_EXECUTABLE) (Required is at least version "1.8.6")
      CMake Warning at CMakeLists.txt:411 (MESSAGE):
        Doxygen based documentation generation is enabled, but couldn't find
        doxygen.

        In order to turn off this warning either disable doxygen documentation
        generation with -DENABLE_DOXYGEN=OFF cmake option or install doxygen.


      -- Found PythonInterp: D:/Programs/Python37/python.exe (found version "3.7.4")
      -- Performing Test COMPILER_HAS_DEPRECATED_ATTR
      -- Performing Test COMPILER_HAS_DEPRECATED_ATTR - Failed
      -- Performing Test COMPILER_HAS_DEPRECATED
      -- Performing Test COMPILER_HAS_DEPRECATED - Success
      -- dir='D:/Projects/shogun/src'
      -- dir='D:/Projects/shogun/build/src'
      -- dir='D:/Projects/shogun/src/gpl'
      -- Looking for pthread.h
      -- Looking for pthread.h - not found
      -- Found Threads: TRUE
      -- Found OpenMP_C: -openmp (found version "2.0")
      -- Found OpenMP_CXX: -openmp (found version "2.0")
      -- Found OpenMP: TRUE (found version "2.0")
      -- Performing Test HAVE_CXA_DEMANGLE
      -- Performing Test HAVE_CXA_DEMANGLE - Failed
      -- Could NOT find CxaDemangle (missing: HAVE_CXA_DEMANGLE)
      -- Looking for xmmintrin.h
      -- Looking for xmmintrin.h - found
      -- Looking for emmintrin.h
      -- Looking for emmintrin.h - found
      -- Could NOT find CxaDemangle (missing: HAVE_CXA_DEMANGLE)
      -- Looking for signgam
      -- Looking for signgam - not found
      -- Looking for fdopen
      -- Looking for fdopen - found
      -- Looking for lgammal
      -- Looking for lgammal - found
      -- Using system's malloc
      -- Performing Test HAVE_STD_ALIGNED_ALLOC
      -- Performing Test HAVE_STD_ALIGNED_ALLOC - Failed
      -- Could NOT find Eigen3 (missing: EIGEN_INCLUDE_DIR) (Required is at least version "3.1.2")
      -- Could NOT find OPENCL (missing: OPENCL_LIBRARY)
      -- Could NOT find ViennaCL (missing: VIENNACL_INCLUDE_DIR VIENNACL_ENCODED_VERSION OpenCL_INCLUDE_DIRS OpenCL_LIBRARIES) (Required is at least version "1.5.0")
      -- Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
      -- Could NOT find rxcpp (missing: rxcpp_INCLUDE_DIR)
      -- Could NOT find TFLogger (missing: TFLogger_DIR)
      -- Looking for sgemm_
      -- Looking for sgemm_ - not found
      -- Could NOT find BLAS (missing: BLAS_LIBRARIES)
      -- LAPACK requires BLAS
      -- Performing Test HAVE_STD_VARIANT
      -- Performing Test HAVE_STD_VARIANT - Success
      -- Could NOT find spdlog (missing: spdlog_DIR)
      -- Could NOT find GLPK (missing: GLPK_LIBRARY GLPK_INCLUDE_DIR GLPK_PROPER_VERSION_FOUND)
      -- Could NOT find LibArchive (missing: LibArchive_LIBRARY LibArchive_INCLUDE_DIR)
      -- Could NOT find CPLEX (missing: CPLEX_LIBRARY CPLEX_INCLUDE_DIR)
      -- Could NOT find MOSEK (missing: MOSEK_DIR MOSEK_INCLUDE_DIR MOSEK_LIBRARY MOSEK_LIBRARIES)
      -- Could NOT find Protobuf (missing: Protobuf_LIBRARIES Protobuf_INCLUDE_DIR)
      CMake Warning at src/shogun/CMakeLists.txt:426 (find_package):
        By not providing "FindRapidJSON.cmake" in CMAKE_MODULE_PATH this project
        has asked CMake to find a package configuration file provided by
        "RapidJSON", but CMake did not find one.

        Could not find a package configuration file provided by "RapidJSON" with
        any of the following names:

          RapidJSONConfig.cmake
          rapidjson-config.cmake

        Add the installation prefix of "RapidJSON" to CMAKE_PREFIX_PATH or set
        "RapidJSON_DIR" to a directory containing one of the above files.  If
        "RapidJSON" provides a separate development package or SDK, be sure it has
        been installed.


      -- Could NOT find CURL (missing: CURL_LIBRARY CURL_INCLUDE_DIR)
      -- Could NOT find ZLIB (missing: ZLIB_LIBRARY ZLIB_INCLUDE_DIR)
      -- Could NOT find BZip2 (missing: BZIP2_LIBRARIES BZIP2_INCLUDE_DIR)
      -- Could NOT find LibLZMA (missing: LIBLZMA_LIBRARY LIBLZMA_INCLUDE_DIR LIBLZMA_HAS_AUTO_DECODER LIBLZMA_HAS_EASY_ENCODER LIBLZMA_HAS_LZMA_PRESET)
      -- Could NOT find SNAPPY (missing: SNAPPY_LIBRARIES SNAPPY_INCLUDE_DIR)
      -- Lzo includes and libraries NOT found.
      -- Could NOT find NLOPT (missing: NLOPT_LIBRARY NLOPT_INCLUDE_DIR)
      -- Could NOT find LPSOLVE (missing: LPSOLVE_LIBRARIES LPSOLVE_INCLUDE_DIR)
      -- Could NOT find ColPack (missing: COLPACK_LIBRARIES COLPACK_LIBRARY_DIR COLPACK_INCLUDE_DIR)
      -- Could NOT find ARPREC (missing: ARPREC_LIBRARIES ARPREC_INCLUDE_DIR)
      -- Linker: Default system linker
      -- Checking to see if CXX compiler accepts flag -flto=thin
      -- Checking to see if CXX compiler accepts flag -flto=thin - yes
      -- Found Jinja2: 1
      -- Could NOT find Ctags (missing: CTAGS_EXECUTABLE)
      Please install Ctags for trained models serialization tests.
      -- Found Sphinx: D:/Programs/Python37/Scripts/sphinx-build.exe
      -- Failed to locate pandoc executable (missing: PANDOC_EXECUTABLE)
      -- ===================================================================================================================
      -- Summary of Configuration Variables
      -- The following OPTIONAL packages have been found:

       * OpenMP
       * Threads
       * Jinja2
       * Sphinx

      -- The following REQUIRED packages have been found:

       * PythonInterp

      -- The following OPTIONAL packages have not been found:

       * Doxygen (required version >= 1.8.6)
       * CxaDemangle
       * Eigen3 (required version >= 3.1.2)
       * ViennaCL (required version >= 1.5.0)
       * rxcpp
       * TFLogger (required version >= 0.1.0)
       * BLAS
       * spdlog
       * GLPK
       * LibArchive
       * CPLEX
       * ARPACK
       * Mosek
       * Protobuf
       * RapidJSON
       * CURL
       * ZLIB
       * BZip2
       * LibLZMA
       * SNAPPY
       * LZO
       * NLopt
       * LpSolve
       * ColPack
       * ARPREC
       * Ctags
       * Pandoc

      -- ===================================================================================================================
      -- Integrations
      --   OpenCV Integration is OFF   enable with -DOpenCV=ON
      -- ===================================================================================================================
      -- Interfaces
      --   Python is OFF               enable with -DINTERFACE_PYTHON=ON
      --   Octave is OFF               enable with -DINTERFACE_OCTAVE=ON
      --   Java is OFF                 enable with -DINTERFACE_JAVA=ON
      --   Perl is OFF                 enable with -DINTERFACE_PERL=ON
      --   Ruby is OFF                 enable with -DINTERFACE_RUBY=ON
      --   C# is OFF                   enable with -DINTERFACE_CSHARP=ON
      --   R is OFF                    enable with -DINTERFACE_R=ON
      --   Scala is OFF                enable with -DINTERFACE_SCALA=ON
      --   CoreML is OFF               enable with -DINTERFACE_COREML=ON
      -- ===================================================================================================================
      -- To compile shogun type
      --   make
      --
      -- To install shogun to C:/Program Files (x86)/shogun type
      --   make install
      --
      -- or to install to a custom directory
      --   make install DESTDIR=/my/special/path
      --   (or rerun cmake with -DCMAKE_INSTALL_PREFIX=/my/special/path) to just change the prefix
      -- ===================================================================================================================
      -- Configuring done
      -- Generating done
      -- Build files have been written to: D:/Projects/shogun/build

      D:\Projects\shogun>msbuild %DESTDIR%\build\shogun.sln /verbosity:minimal /t:Clean /p:Configuration=Release /p:Platform=x64
      Microsoft (R) Build Engine version 16.4.0+e901037fe for .NET Framework
      Copyright (C) Microsoft Corporation. All rights reserved.


      D:\Projects\shogun>
      
   </pre>
</details>
