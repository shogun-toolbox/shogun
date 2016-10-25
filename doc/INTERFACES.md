Running Shogun from the interfaces     {#interfaces}
==================================

This document explains how to use Shogun from its different interfaces. We
assume that installation (including the interfaces) was successful and all
dependencies are installed.

To get help installing Shogun and for example code for all interfaces, see our
website.

Note that setting some the enviromental variables should not be necessary in case
you installed Shogun to the default folder.

## The shared library
All interfaces require the Shogun library to be visible to your system. Either
`make install` to the default directory, or if using a custom directory

	$ export LD_LIBRARY_PATH="path/to/libshogun.so:$LD_LIBRARY_PATH"

The `libshogun.so` was either copied to `path/to/shogun-install/lib/` when
running `make install`, but you can also make it point to the build directory
 `path/to/build/src/shogun/` to make it available after a successful `make`.
All subsequent settings can be set to the build dir or the installation dir.

## Interfaces

We now describe how to run code that uses Shogun in all interfaces. For language
specific defails how to import and use Shogun in all interfaces, see the
examples on our website.

### Native C++
Compilation needs the Shogun headers path, i.e. the path where for example `shogun/base/init.h` is located. This is either in `path/to/src/shogun/` or in `path/to/shogun-install/include/shogun/` . Linking  with the `-lshogun` flag requires the LD_LIBRARY_PATH set up as described above.
Compiling and linking code works with gcc as

    $ gcc path/to/native_example.cpp -o native_example -I/path/to/headers -lshogun

Running it:

    $ ./native_example

### Python
This needs `modshogun.py` to be visible, which is either in `path/to/build/src/interfaces/python/` or in something similar to `path/to/shogun-install/lib/python2.7/dist-packages/`

    $ export PYTHONPATH="path/to/modshogun.py:$PYTHONPATH"

Running an example:

    $ python path/to/python_example.py

### Octave
This needs `modshogun.oct` to be visible, which is either in `path/to/build/src/interfaces/octave/` or in something similar to `path/to/shogun-install/lib/x86_64-linux-gnu/octave/site/oct/api-v50+/x86_64-pc-linux-gnu/shogun/`

    $ export OCTAVE_PATH="path/to/modshogun.oct:$OCTAVE_PATH"

Running an example:

    $ python path/to/octave_example.py

### Ruby

### R

### Lua

### CSharp
This needs `modshogun.dll` to be visible, which is either in `path/to/build/src/interfaces/csharp` or in something similar to `path/to/shogun-install/lib/cli/shogun/`

Compiling code works with the mono C# compiler and passing location of the above file

    $ mcs path/to/csharp_example.cs /lib:path/to/modshogun.dll/r:modshogun -out:csharp_example.exe

Running requires setting the mono path

    $ export MONO_PATH=/home/heiko/git/shogun/shogun_develop/shogun/build/src/interfaces/csharp_modular:$MONO_PATH

Running it:

    $ mono csharp_example

### Java
This needs `shogun.jar` to be visible, which is either in `path/to/build/src/interfaces/java/` or in something similar to `path/to/shogun-install//share/java/` .
In addition, the location of the external dependency `jblas.jar` is needed,
usually in `/usr/share/java/` .

Compiling code works with the java compiler and passing location of `shogun.jar`,
`jblas.jar`, and the example itself in the class path
    $javac -cp /path/to/jblas.jar:/path/to/modshogun.jar:path/to/java_example.java -d /path/to/output/ /path/to/java_example.java
					
Running it again requires the above class path and some more options

    $java -Xmx1024m -cp /path/to/jblas.jar:/path/to/shogun.jar:path/to/java_example.java -Djava.library.path=/path/to/shogun.jar java_example
