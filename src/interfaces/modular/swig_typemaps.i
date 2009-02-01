#ifdef HAVE_SWIG

#if defined(HAVE_PYTHON)
%include "lib/python_typemaps.i"
#elif defined(HAVE_OCTAVE)
%include "lib/octave_typemaps.i"
#elif defined(HAVE_R)
%include "lib/r_typemaps.i"
#else
#error("unknown swig interface")
#endif

#endif
