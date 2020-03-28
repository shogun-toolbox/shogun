# Installing Shogun

For certain systems, we offer pre-built packages of Shogun.
This is the easiest way to start using it.
In other cases, we describe how to build Shogun from source code.


# Quicklinks
 * [Ready-to-install packages](#binaries)
   - [Anaconda](#anaconda)
   - [Ubuntu](#ubuntu)
   - [Debian](#debian)
   - [Fedora](#fedora)
   - [macOS](#mac)
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

These packages include most of the optional dependencies and are currently available for Linux, macOS, and Windows.

### Ubuntu PPA <a name="ubuntu"></a>
We are working on integrating Shogun with Debian/Ubuntu.
In the meantime, we offer a [prepackaged PPA](https://launchpad.net/~shogun-toolbox/+archive/ubuntu/stable).
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
The latest packages for Debian `jessie` and `stretch` are available in our own repository at [http://apt.shogun.ml](http://apt.shogun.ml).
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


### macOS <a name="mac"></a>
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

The docker image follows both the `master` and the `develop` branch of the repository, just specify the desired branch name as a tag for the image. For example, to use the develop version of shogun simply pull the `shogun/shogun:develop` docker image.

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
Shogun is can be automatically built from the following language's source.

### Python PyPI <a name="pypi"></a>
You can install it from [PyPI](https://pypi.python.org/pypi/shogun-ml/).
There is limited control over options and it might take a long time as everything is done from scratch.

    pip install shogun-ml

We do not recommend this option and suggest to rather compile by hand as described below.


# Compiling manually <a name="manual"></a>

In case none of the binary packages listed on our website work for your system, or you want to modify Shogun, you will need to build it from the source.

## Requirements <a name="manual-requirements"></a>
The standard GNU/Linux tools and Python are minimal requirements to compile Shogun.

By default, building the meta examples is enabled, requiring `python-ply` and `ctags`. Disable using `-DBUILD_META_EXAMPLES=Off` if those requirements are a problem, also see [generating meta examples](#manual-examples).

To compile the interfaces, in addition to [swig](http://www.swig.org/) itself, you will need language-specific development packages installed, see [interfaces](#manual-interfaces) below.

There is a larger number of optional requirements.
The output of cmake output lists optional dependencies that were found and not found.
If a particular Shogun class is unavailable, this is likely due to an unmet dependency.
See our [docker configuration file](https://github.com/shogun-toolbox/shogun/blob/develop/configs/shogun/Dockerfile) for an example configuration used in our test builds.

You need at least 1GB free disk space. If you compile any interface, roughly 4 GB RAM needed (we are working on reducing this).
[CCache](https://ccache.samba.org/) will massively speed up the compilation process and is enabled by default if installed.

## Basics <a name="manual-basics"></a>
Shogun uses [CMake](https://cmake.org/) for its build. The general workflow is now explained.
For further details on testing etc, see [DEVELOPING.md](DEVELOPING.md).

Download the latest [stable release source code](https://github.com/shogun-toolbox/shogun/releases/latest), or (as demonstrated here) clone the latest development code.
Potentially update submodules

    git clone https://github.com/shogun-toolbox/shogun.git
    cd shogun
    git submodule update --init

Create the build directory in the source tree root

    mkdir build

Configure cmake, from the build directory, passing the Shogun source root as an argument.
It is recommended to use any of CMake GUIs (e.g. replace `cmake ..` with `ccmake ..`), in particular, if you feel unsure about possible parameters and configurations.
Note that all cmake options read as `-DOPTION=VALUE`.

    cd build
    cmake [options] ..

Compile

    make


Install (prepend `sudo` if installing system-wide), and you are done.

    make install

Sometimes you might need to clean up your build (e.g. in case of some major changes).
First, try

    make clean

If that does not help, try removing the build directory and starting from scratch afterward

    rm -rf build

If you prefer to not run the `sudo make install` command system-wide, you can either install Shogun to a custom location (`-DCMAKE_INSTALL_PREFIX=/custom/path`, defaults to `/usr/local`), or even skip `make install` at all.
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

To *use* the interfaces, in particular, if not installing to the default system-wide location, see [INTERFACES.md](INTERFACES.md).
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

In case you have a problem building Shogun, please open an [issue on GitHub](https://github.com/shogun-toolbox/shogun/issues) with your system details, *exact* commands used, and logs posted as a [gist](https://gist.github.com/).

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
to manually adjust the library and include paths, `-DCMAKE_INCLUDE_PATH=/my/include/path` and `-DCMAKE_LIBRARY_PATH=/my/library/path`.


## Customized Python environments <a name="manual-python"></a>
Often, there are multiple Python versions installed on the system.
There are various reasons for this, i.e. Linux without root access, MacOS + homebrew, using [Anaconda](https://www.continuum.io/downloads) or [virtualenv](https://virtualenv.pypa.io).
If Shogun is executed using a version of Python different from the one it was built, there will be import crashes.
If this is your setup, you need to make sure that Shogun is both **built** and **executed** against the Python environment of **your** choice.
For that, you need to do something similar to

    cmake -DPYTHON_INCLUDE_DIR=/path/to/python/include/dir -DPYTHON_LIBRARY=path/to/python/libpythonVERSION.{so|dynlib} -DPYTHON_EXECUTABLE=/path/to/python/executable -DPYTHON_PACKAGES_PATH=/path/to/python/dist-packages ..

For example, for `brew` installed Python under MacOS, use something like:

    cmake -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Headers -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib  -DINTERFACE_PYTHON=ON ..

Under Linux, where you want to use Python 3, which is not the system's default:

    cmake -DPYTHON_INCLUDE_DIR=/usr/include/python3.3 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DPYTHON_PACKAGES_PATH=/usr/local/lib/python3.3/dist-packages -DINTERFACE_PYTHON=ON ..

On a Linux cluster without root access, using [Anaconda](https://www.continuum.io/downloads) (note you will need to activate your environment every time you want to run Shogun):

    source path/to/anaconda/bin/activate
    cmake -DCMAKE_INSTALL_PREFIX=path/to/shogun/install/dir -DPYTHON_INCLUDE_DIR=path/to/anaconda/include/python2.7/ -DPYTHON_LIBRARY=path/to/anaconda/lib/libpython2.7.so  -DPYTHON_EXECUTABLE=path/to/anaconda/bin/python -DINTERFACE_PYTHON=ON ..

## Windows build <a name="manual-windows"></a>

Please see any of our Windows py3X [Azure Pipelines](https://dev.azure.com/shogunml/shogun/_build?definitionId=2) build to get any other information on the build process.
> It is recommended to use Visual Studio 16 2019" or "MSBuild".

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Open the start menu, and run _Anaconda Prompt (Miniconda3)_

3. In the Windows Start menu, find the path of the _Native Tools Command Prompt for VS_ shortcut that is relative to your system architecture and Visual Studio version (x32/x64, 2017/2019/..). For example, for _x64 Native Tools Command Prompt for VS 2019_, it looks so: `%comspec% /k "X:\Path\To\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"`. For detailed information check out [link 1](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line) and [link 2](https://docs.microsoft.com/en-us/dotnet/framework/tools/developer-command-prompt-for-vs)

4. Execute this path in the _Anaconda Prompt_ to run _Native Tools_ there 

5. Put these lines into the _Anaconda Prompt_:
> rem - Records comments (remarks); @rem - do not print command just its result (the rem result is empty)

> Be careful cmake can search out of the build directory to load libraries thus remove other directories that contain the shogun sources

```Batchfile
@rem [FILL THIS SECTION WITH THE VALUES YOU WANT
@rem For example: SET MAIN_DIR=D:\Build
SET MAIN_DIR=X:\Path\To\Build
@rem Basically your number of cores
SET MAX_CPU_COUNT=8
@rem Take a look at https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators
SET PLATFORM=-G "Visual Studio 16 2019" -A x64
@rem FILL THIS SECTION WITH THE VALUES YOU WANT]

SET REPO_DIR=%MAIN_DIR%\shogun
SET VENV_DIR=%MAIN_DIR%\envs\shogun
SET CLCACHE_DIR=%MAIN_DIR%\clcache
SET SourcesDirectory=%REPO_DIR%\build
SET BinariesDirectory=%MAIN_DIR%\binaries
SET targetPrefix=%BinariesDirectory%\opt
SET clcacheArtifactName=clcache-vs17
SET buildConfiguration=Release
SET SourceBranchName=develop

git clone https://github.com/shogun-toolbox/shogun %REPO_DIR%
CHDIR /d %REPO_DIR%
git submodule update --init --force --depth=5
conda create --quiet --prefix %VENV_DIR% --mkdir --yes python=3.6.* setuptools numpy scipy eigen snappy zlib ctags ply jinja2 gtest mkl-devel swig -c conda-forge
activate %VENV_DIR%

%REPO_DIR%\.ci\setup_clcache.cmd

MKDIR %BinariesDirectory%
CHDIR /d %BinariesDirectory%
%REPO_DIR%\.ci\get_latest_artifact.py %SourceBranchName% %clcacheArtifactName%
@rem If there is no tar program in your Windows do:
@rem conda install -c haasad eidl7zip --yes
@rem FOR /F "delims=" %i IN ('where .\%clcacheArtifactName%:*.tar*') DO 7za x %i -so | 7za x -aoa -si -ttar -o%CLCACHE_DIR%
@rem Or simply extract the file whose name is the output of the 'where .\%clcacheArtifactName%:*.tar*' command
FOR /F "delims=" %i IN ('where .\%clcacheArtifactName%:*.tar*') DO tar --extract --file=%i --directory %CLCACHE_DIR% --gzip

MKDIR %targetPrefix% %SourcesDirectory%
CHDIR /d %SourcesDirectory%
@rem It is necessary to have '/' not '\' in the DBLAS_LIBRARIES and DLAPACK_LIBRARIES attributes
cmake %PLATFORM% -DCMAKE_BUILD_TYPE=%buildConfiguration% -DCMAKE_PREFIX_PATH=%VENV_DIR%\Library -DENABLE_TESTING=ON -DCMAKE_INSTALL_PREFIX=%targetPrefix% -DBUILD_META_EXAMPLES=OFF -DBLAS_LIBRARIES=%VENV_DIR%/Library/lib/mkl_core_dll.lib -DLAPACK_LIBRARIES=%VENV_DIR%/Library/lib/mkl_core_dll.lib ..

cmake --build . --config %buildConfiguration% --target INSTALL -- -p:TrackFileAccess=false -p:CLToolExe=clcache.exe -maxcpucount:%MAX_CPU_COUNT%
```

5. Now you can use the Conda environment (located in VENV_DIR) containing shogun library in your Python projects!
