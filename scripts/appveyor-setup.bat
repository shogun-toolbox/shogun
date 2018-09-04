@echo on

conda update -y -q conda
conda config --set auto_update_conda false
conda info -a

conda config --set show_channel_urls True

@rem Help with SSL timeouts to S3
conda config --set remote_connect_timeout_secs 12

conda config --add channels https://repo.continuum.io/pkgs/free
conda config --add channels conda-forge
conda info -a

if "%GENERATOR%"=="NMake Makefiles" set need_vcvarsall=1
if "%GENERATOR%"=="Ninja" set need_vcvarsall=1

if defined need_vcvarsall (
    @rem Select desired compiler version
    if "%APPVEYOR_BUILD_WORKER_IMAGE%" == "Visual Studio 2017" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
    ) else (
        call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
    )
)

@rem Use clcache for faster builds
pip install -q git+https://github.com/frerich/clcache.git
clcache -s
set CLCACHE_SERVER=1
set CLCACHE_HARDLINK=1
powershell.exe -Command "Start-Process clcache-server"

conda create -n shogun -q -y python=%PYTHON% ^
    setuptools numpy scipy eigen rxcpp ^
    cmake snappy zlib ctags ply ninja

call activate shogun
