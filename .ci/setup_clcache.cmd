@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Use clcache for faster builds
pip install -q git+https://github.com/frerich/clcache.git
clcache -s
set CLCACHE_SERVER=1
set CLCACHE_HARDLINK=1
powershell.exe -Command "Start-Process clcache-server"

if %errorlevel% neq 0 exit /b %errorlevel%
