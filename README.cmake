# setup cmake for developers (debugging symbols on, optimization off, etc.):
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON -DCMAKE_INSTALL_PREFIX="$BUILDDIR/install" ..


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