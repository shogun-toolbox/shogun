#ifdef HAVE_PYTHON
#include "../python_static/PythonInterface.cpp"
#endif

#ifdef HAVE_MATLAB
#include "../matlab_static/MatlabInterface.cpp"
#endif

#ifdef HAVE_OCTAVE
#include "../octave_static/OctaveInterface.cpp"
#endif

#ifdef HAVE_R
#include "../r_static/RInterface.cpp"
#endif
