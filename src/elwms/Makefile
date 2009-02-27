all: elwms.cpp.o
	c++ elwms.cpp.o -Xlinker --no-undefined -lm -pthread -llapack -lcblas \
	-llapack_atlas -lglpk -lm -L../libshogun -lshogun -L../libshogunui \
	-lshogunui -shared  -lpython2.5 -L/usr/lib/octave-3.0.1 -loctinterp -loctave \
	-lcruft -L/usr/lib/R/lib -lR -o sg.so \
	-Wl,--rpath-link,/opt/matlab7.4/extern/lib/glnx86 \
	-L/opt/matlab7.4/bin/glnx86 -lmx -lmex -lmat

elwms.cpp.o: elwms.cpp ../matlab/MatlabInterface.h ../matlab/MatlabInterface.cpp \
	../octave/OctaveInterface.h ../octave/OctaveInterface.cpp \
	../r/RInterface.h ../r/RInterface.cpp \
	../python/PythonInterface.h ../python/PythonInterface.cpp
	c++ -fPIC -g -Wall -Wno-unused-parameter -Wformat -Wformat-security \
	-Wimplicit -Wparentheses -Wshadow -O9 -fexpensive-optimizations \
	-frerun-cse-after-loop -fcse-follow-jumps -finline-functions \
	-fschedule-insns2 -fthread-jumps -fforce-addr -fstrength-reduce \
	-unroll-loops -march=i486 -mtune=i686 -pthread -DLINUX -DHAVE_POWL \
	-DHAVE_SQRTL -DHAVE_LOG2 -DHAVE_ATLAS -DHAVE_LAPACK -DUSE_GLPK \
	-DHAVE_LARGEFILE -DUSE_SHORTREAL_KERNELCACHE -DUSE_HMMPARALLELSTRUCTURES \
	-DUSE_HMMPARALLEL -DUSE_BIGSTATES -DUSE_HMMCACHE -DUSE_SVMLIGHT \
	-DHAVE_DOXYGEN -DHAVE_R -c -I. -I.. -I. -I..  -I../libshogun \
	-I../libshogunui -I/usr/share/R/include -DHAVE_PYTHON \
	-I/usr/include/python2.5 -I/usr/include -DHAVE_MATLAB \
	-I/usr/lib/python2.5/site-packages/numpy/core/include -DHAVE_OCTAVE \
	-I/usr/include/octave-3.0.1 -I/opt/matlab7.4/extern/include \
	-I/opt/matlab7.4/extern/src -o elwms.cpp.o elwms.cpp

clean:
	rm -f elwms.cpp.o sg.so
