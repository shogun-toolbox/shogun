* Getting a list of possible interfaces to enable:

```
$ grep -E "OPTION.*(Modular|Static)" CMakeLists.txt
```

* If eigen3 or json-c are missing use the following to download and compile these dependencies:

```
$ cmake -DBUNDLE_EIGEN=ON -DBUNDLE_JSON=ON
```

* CMake setup for developers (debugging symbols on, optimization off, etc.):

```
$ mkdir build-debug
$ cd build-debug
$ cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON -DTRACE_MEMORY_ALLOCS=OFF -DPythonModular=ON -DBUILD_DASHBOARD_REPORTS=ON -DCMAKE_INSTALL_PREFIX="$BUILDDIR/install" ..
```

* Specify a different swig executable:

```
$ cmake -DSWIG_EXECUTABLE=/usr/bin/swig2.0
```

* Setup CMake for building the final binaries (debugging off, optimization on):

```
$ mkdir build-release
$ cd build-release
$ cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON -DCMAKE_INSTALL_PREFIX="$BUILDDIR/install" ..
```


* Compile everything and install:

```
$ make -j GoogleMock # only needed on first build: fetch and compile GoogleMock
$ make -j all # compiling everything
$ make -j install # install required for "make test"
$ make -j test # compile and run all tests and examples
```

* Alternative build targets:

```
$ make -j shogun # only compiling libshogun
$ make -j shogun-unit-test # build unit test binary
$ make -j unit-tests # build and run unit tests
```

* Testing/debugging:

```
$ ctest -D ExperimentalMemCheck # runs all tests with valgrind (depends on -DBUILD_DASHBOARD_REPORTS=ON)
$ cd tests/unit && valgrind --leak-check=full ./shogun-unit-test --gtest_filter=EPInferenceMethod.get_cholesky_probit_likelihood
$ cd tests/unit && valgrind --leak-check=full ./shogun-unit-test --gtest_filter=EPInferenceMethod.*
```

* To specify a different compiler, see [CMake FAQ, "How do I use a different compiler?"](http://www.cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F).
You might have to delete the build directory or clear the cmake cache otherwise for this to work.

```
$ CC=/path/to/gcc CXX=/path/to/g++ cmake ..
```

* Under OS X one often has the same Python major versions installed in `/usr` and `/usr/local` via brew etc,
so one might observe crashes if the wrong Python version is linked against. To use a custom Python installation 
for Python bindings one would under brew use:

```
$ cmake -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Headers -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib  -DPythonModular=ON ..
```

or, in general:

```
$ cmake -DPYTHON_INCLUDE_DIR=/path/to/python/include/dir -DPYTHON_LIBRARY=path/to/python/libpythonVERSION.so ..
```
