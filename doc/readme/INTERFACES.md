Running Shogun from the interfaces
==================================

We assume that installation (including the interfaces) was successful and all dependencies are installed. See [INSTALL.md](INSTALL.md) and our website.

Note that setting some the enviromental variables should not be necessary in case you installed Shogun to the default folder or installed it from a binary package.

## The shared library
All interfaces require the Shogun library to be visible to your system.
You can prepend the *folder* (!=full filename) of `libshogun.*` to an environmental variable.
On Linux, this is done with

    export LD_LIBRARY_PATH="path/to/libshogun.so:$LD_LIBRARY_PATH"

On MacOS

    export DYLD_LIBRARY_PATH="path/to/libshogun.dylib:$DYLD_LIBRARY_PATH"

Note that the `libshogun.*` was either copied to `path/to/shogun-install/lib/` when running `make install`.
You can also make it point to the build directory `path/to/build/src/shogun/` to make it available after a successful `make`.
All subsequent settings can be set to the build dir or the installation dir.

## Interfaces

We now describe how to run code that uses Shogun in all interfaces.
For language specific defails how to import and use Shogun in all interfaces, see the examples on our website.

### Native C++
Make sure you read up on how to compile C/C++ code.
Compilation requires the Shogun headers path, i.e. the path where for example `shogun/base/init.h` is located.
This is either in `path/to/src/shogun/` or in `path/to/shogun-install/include/shogun/` and is specified  via the `-I` flag.
Linking requires the `-lshogun` flag, which either needs the `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` set up as described above, or preferably passed via the `-L` flag.
Compiling and linking code with `gcc` works as

    gcc path/to/native_example.cpp -o native_example -I/path/to/headers -lshogun -L/path/to/libshogun.*

Running it:

    ./native_example

### Python
This needs `modshogun.py` to be visible, which is either in `path/to/build/src/interfaces/python_modular/` or in something similar to `path/to/shogun-install/lib/python2.7/dist-packages/`

    export PYTHONPATH="path/to/modshogun.py:$PYTHONPATH"

Running an example:

    python path/to/python_example.py

### Octave
This needs `modshogun.oct` to be visible, which is either in `path/to/build/src/interfaces/octave_modular/` or in something similar to `path/to/shogun-install/lib/x86_64-linux-gnu/octave/site/oct/api-v50+/x86_64-pc-linux-gnu/shogun/`

    export OCTAVE_PATH="path/to/modshogun.oct:$OCTAVE_PATH"

Running an example:

    python path/to/octave_example.py

### Ruby
This needs `modshogun.rb` to be visible, which is either in `path/to/build/src/interfaces/ruby_modular/` or in something similar to `path/to/shogun-install/lib/x86_64-linux-gnu/site_ruby`
    export RUBYLIB="path/to/modshogun.rb:$RUBYLIB"

Running an example:

    ruby path/to/ruby_example.rb

### R
This needs `modshogun.R` to be visible, which is either in `path/to/build/src/interfaces/r_modular/` or in something similar to `path/to/shogun-install/lib/R/site-library`
    export R_LIBS_USER="path/to/modshogun.R:$R_LIBS_USER"

Running an example:
    R --no-restore --no-save --no-readline --slave -f path/to/r_example.rb

### Lua
This needs `libmodshogun.so` (this is the interface file, not the shared library file `libshogun.so`) to be visible, which is either in `path/to/build/src/interfaces/lua_modular/` or in something similar to `path/to/shogun-install/lib/lua/5.1/`

    export LUA_CPATH="path/to/libmodshogun.so:$LUA_CPATH"

Running an example:

    R --no-restore --no-save --no-readline --slave -f path/to/r_example.R

### CSharp
This needs `modshogun.dll` to be visible, which is either in `path/to/build/src/interfaces/csharp_modular` or in something similar to `path/to/shogun-install/lib/cli/shogun/`

Compiling code works with the mono C# compiler and passing location of the above file

    mcs path/to/csharp_example.cs /lib:path/to/modshogun.dll/r:modshogun -out:csharp_example.exe

Running requires setting the mono path

    export MONO_PATH=/home/heiko/git/shogun/shogun_develop/shogun/build/src/interfaces/csharp_modular:$MONO_PATH

Running it:

    mono csharp_example

### Java
This needs `shogun.jar` to be visible, which is either in `path/to/build/src/interfaces/java_modular/` or in something similar to `path/to/shogun-install/share/java/` .
In addition, the location of the external dependency `jblas.jar` is needed,
usually in `/usr/share/java/`.

Compiling code works with the java compiler and passing location of `shogun.jar`,
`jblas.jar`, and the example itself in the class path

    javac -cp /path/to/jblas.jar:/path/to/modshogun.jar:path/to/java_example.java -d /path/to/output/ /path/to/java_example.java

Running it again requires the above class path and some more options

    java -Xmx1024m -cp /path/to/jblas.jar:/path/to/shogun.jar:path/to/java_example.java -Djava.library.path=/path/to/shogun.jar java_example

### Provided Examples
Stand-alone, executable code for all interface examples on our website (and more) can be generated locally, see [INSTALL.md](INSTALL.md).
As the examples load data files, they requires the `shogun-data` submodule to be checked out.

All examples should be run in the respective folder they are located in, for example (assuming that all described variables are set)

    cd /path/to/shogun-install/examples/meta/python/regression/
    python linear_ridge_regression.py

Or, for a compiled language with a manually compiled, not yet installed Shogun, running directly from the source tree

    cd /path/to/shogun-source/build/examples/meta/csharp/regression/
    mono linear_ridge_regression.cs
