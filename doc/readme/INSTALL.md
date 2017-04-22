# Installing Shogun

For certain systems, we offer pre-built packages of Shogun.
This is the easiest way to start using it.
For other cases, we describe how to build Shogun from source code.


# Quicklinks
 * [Ready-to-install packages](#binaries)
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
   - [Winows](#manual-windows)

## Ready-to-install packages <a name="binaries"></a>

### Ubuntu ppa <a name="ubuntu"></a>
We are working on integrating Shogun with Debian/Ubuntu.
In the meantime, we offer a [prepackaged ppa](https://launchpad.net/~shogun-toolbox/+archive/ubuntu/stable).
These currently do contain the C++ library and Python bindings.
Add this to your system as

    sudo add-apt-repository ppa:shogun-toolbox/stable
    sudo apt-get update

Then, install as

    sudo apt-get install libshogun17

The Python (2) bindings can be installed as

    sudo apt-get install python-shogun

In addition to the latest stable release, we offer [nightly builds](https://launchpad.net/~shogun-toolbox/+archive/ubuntu/nightly) of our development branch.

### Debian <a name="debian"></a>
Latest packages for Debian jessie are available in our own repository at [http://apt.shogun.ml](http://apt.shogun.ml).
We provide both the stable and nightly packages, currenlty only for amd64 architecture.
In order to add the stable packages to your system, simply run the following commands

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3DD2174ACAB30365
    echo "deb http://apt.shogun.ml/ jessie main" | sudo tee /etc/apt/sources.list.d/shogun-toolbox.list  > /dev/null
    sudo apt-get update

After this just simply install the shogun library

    sudo apt-get install libshogun17

The nightly packages are available in the `nightly` component, i.e. `deb http://apt.shogun.ml/ jessie nightly`

### Fedora <a name="fedora"></a>
Shogun is part of [Fedora 25](https://admin.fedoraproject.org/pkgdb/package/rpms/shogun/).
Install as

    sudo dnf install shogun


### MacOS <a name="mac"></a>
Shogun is part of [homebrew-science](https://github.com/Homebrew/homebrew-science).
Install the latest stable version as

    sudo brew install shogun

Note: Shogun in homebrew is outdated.
Contact us if this changed or if you want to help changing it.

### Windows <a name="windows"></a>
Shogun natively compiles under Windows using MSVC, see the [AppVeyor CI build](https://ci.appveyor.com/project/vigsterkr/shogun) and the [Windows section](#manual-windows)
We currently do not support a binary installer.
If you are interested in packaging, documenting, or contributing otherwise, please contact us.

## Docker images <a name="docker"></a>
You can run Shogun in [our own cloud](cloud.shogun.ml) or set up your own using our
[Docker images](https://hub.docker.com/r/shogun/shogun/) as:

    sudo docker pull shogun/shogun:master
    sudo docker run -it shogun/shogun:master bash

We offer images for both the latest release and nightly development builds.

For the [developer version](https://hub.docker.com/r/shogun/shogun-dev/), replace `shogun/shogun:master` with `shogun/shogun-dev`.

Check the "details" tab before downloading to check if the latest build was successful (otherwise you might run into errors when running the docker image)."

Sometimes mounting a local folder into the docker image is useful.
You can do this via passing an additional option

```
-v /your/local/folder:/same/folder/in/docker
```

See the Docker documentation for further details.


## Integration with interface language build systems <a name="language"></a>
Shogun is can be automatically built from source from the following langauges.

### Python pypi <a name="pypi"></a>
You can install from [pipy](https://pypi.python.org/pypi/shogun-ml/).
There is limited control over options and it might take a long time as everything is done from scratch.

    pip install shogun-ml

We do not reccomend this option and suggest to rather compile by hand as described below.


# Compiling manually <a name="manual"></a>

In case none of the binary packages listed on our website work for your system, or you want to modify Shogun, you will need to build it from source.

## Requirements <a name="manual-requirements"></a>
The standard GNU/Linux tools and Python are minimal requirements to compile Shogun.
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
    git submodule update --init

Create the build directory in the source tree root

    cd shogun
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
The cmake options for building interfaces are `-DPythonModular -DOctaveModular -DRModular -DJavaModular -DRubyModular -DLuaModular -DCSharpModular` etc.
For example, replace the cmake step above by
```
cmake -DPythonModular=ON [potentially more options] ..
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

Getting a list of possible interfaces to enable:

    grep -E "OPTION.*(Modular)" CMakeLists.txt

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

    cmake -DPYTHON_INCLUDE_DIR=/path/to/python/include/dir -DPYTHON_LIBRARY=path/to/python/libpythonVERSION.{so|dynlib} -DPYTHON_EXECUTABLE:/path/to/python/executable -DPYTHON_PACKAGES_PATH=/path/to/python/dist-packages ..

For example, for `brew` installed Python under MacOS, use something like:

    cmake -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Headers -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib  -DPythonModular=ON ..

Under Linux, where you want to use Python 3, which is not the system's default:

    cmake -DPYTHON_INCLUDE_DIR=/usr/include/python3.3 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DPYTHON_PACKAGES_PATH=/usr/local/lib/python3.3/dist-packages -DPythonModular=ON ..

On a Linux cluster without root access, using [Anaconda](https://www.continuum.io/downloads) (note you will need to activate your environment everytime you want to run Shogun):

    source path/to/anaconda/bin/activate
    cmake -DCMAKE_INSTALL_PREFIX=path/to/shogun/install/dir -DPYTHON_INCLUDE_DIR=path/to/anaconda/include/python2.7/ -DPYTHON_LIBRARY=path/to/anaconda/lib/libpython2.7.so  -DPYTHON_EXECUTABLE=path/to/anaconda/bin/python -DPythonModular=On ..

## Windows build <a name="manual-windows"></a>

Please see our [AppVeyor](https://ci.appveyor.com/project/vigsterkr/shogun) build.
It is recommended to use "Visual Studio 14 2015" or "MSBuild".
You will need to adjust all path names to the Windows style, e.g.

    git clone https://github.com/shogun-toolbox/shogun.git C:\projects\shogun 
    git submodule -q update --init
    cd C:\projects\shogun
    md build && cd build

You need to specify a different generator in cmake (to match your IDE), e.g.

    cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DBUILD_META_EXAMPLES=OFF -DENABLE_TESTING=ON ..

Compiling works as

    msbuild "C:\projects\shogun\build\shogun.sln" /verbosity:minimal /t:Clean /p:Configuration=Release /p:Platform=x64

Note: If you use /m in msbuild command without specifying the number, it may occur out of memory errors.
