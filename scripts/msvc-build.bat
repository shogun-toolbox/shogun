@echo on

mkdir %APPVEYOR_BUILD_FOLDER%\build
pushd %APPVEYOR_BUILD_FOLDER%\build

cmake -G "%GENERATOR%" ^
      -DCMAKE_BUILD_TYPE=%CONFIGURATION% ^
      -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%\Library ^
      -DBUILD_META_EXAMPLES=OFF ^
      -DENABLE_TESTING=ON .. || exit /B

cmake --build . --target install --config %CONFIGURATION% || exit /B
