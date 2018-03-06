@echo on

conda create -n shogun -q -y python=%PYTHON% ^
    setuptools numpy scipy eigen rxcpp ^
    cmake snappy zlib ctags ply
call activate shogun

mkdir %APPVEYOR_BUILD_FOLDER%\build
pushd %APPVEYOR_BUILD_FOLDER%\build

cmake -G "%VSVER%" ^
      -DCMAKE_BUILD_TYPE=%CONFIGURATION% ^
      -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%\Library ^
      -DBUILD_META_EXAMPLES=OFF ^
      -DENABLE_TESTING=ON .. || exit /B

cmake --build . --target install --config %CONFIGURATION% -- /maxcpucount:2 || exit /B
