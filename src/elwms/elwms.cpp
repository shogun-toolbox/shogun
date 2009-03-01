#ifdef HAVE_PYTHON
#include "../python/PythonInterface.cpp"
#endif

#ifdef HAVE_MATLAB
#include "../matlab/MatlabInterface.cpp"
#endif

#ifdef HAVE_OCTAVE
#include "../octave/OctaveInterface.cpp"
#endif

#ifdef HAVE_R
#include "../r/RInterface.cpp"
#endif
