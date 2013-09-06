# getting a list of possible interfaces to enable:
grep -E "OPTION.*(Modular|Static)" CMakeLists.txt


# setup cmake for developers (debugging symbols on, optimization off, etc.):
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON -DTRACE_MEMORY_ALLOCS=OFF -DPythonModular=ON -DBUILD_DASHBOARD_REPORTS=ON -DCMAKE_INSTALL_PREFIX="$BUILDDIR/install" ..

# specify a different swig executable
cmake -DSWIG_EXECUTABLE=/usr/bin/swig2.0

# setup cmake for building the final binaries (debugging off, optimization on):
mkdir build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON -DCMAKE_INSTALL_PREFIX="$BUILDDIR/install" ..


# compile everything and install
make -j GoogleMock # only needed on first build: fetch and compile GoogleMock
make -j all # compiling everything
make -j install # install required for "make test"
make -j test # compile and run all tests and examples


# alternative build targets:
make -j shogun # only compiling libshogun
make -j shogun-unit-test # build unit test binary
make -j unit-tests # build and run unit tests


# testing/debugging
ctest -D ExperimentalMemCheck # runs all tests with valgrind (depends on -DBUILD_DASHBOARD_REPORTS=ON)
cd tests/unit && valgrind --leak-check=full ./shogun-unit-test --gtest_filter=EPInferenceMethod.get_cholesky_probit_likelihood
cd tests/unit && valgrind --leak-check=full ./shogun-unit-test --gtest_filter=EPInferenceMethod.*
