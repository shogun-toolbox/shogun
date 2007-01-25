%define DOCSTR
"The `Disance` module gathers all distances available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Distance
%{
#define SWIG_FILE_WITH_INIT
#include "distance/Distance.h" 
%}

%include "lib/common.i"

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "lib/common.i"
%include "lib/numpy.i"
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** dst, INT* m, INT* n)};
#endif

%feature("director") CDistance;
%feature("autodoc","1");

%include "lib/io.i" 
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "distance/Distance.h"

%include "distance/SimpleDistance.i"
%include "distance/RealDistance.i"
%include "distance/StringDistance.i"
%include "distance/Canberra.i"
%include "distance/Chebyshew.i"
%include "distance/Geodesic.i"
%include "distance/Jensen.i"
%include "distance/Manhattan.i"
%include "distance/Minkowski.i"
%include "distance/HammingWordDistance.i"
%include "distance/ManhattanWordDistance.i"
%include "distance/CanberraWordDistance.i"
