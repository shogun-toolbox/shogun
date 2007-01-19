%module(directors="1") Distance
%{
#define SWIG_FILE_WITH_INIT
#include "distance/Distance.h" 
%}

%include "lib/common.i"

#ifdef HAVE_PYTHON
%init %{
	  import_array();
%}

%include "lib/numpy.i"
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** dst, INT* m, INT* n)};
#endif

%feature("director") CDistance;

%include "base/SGObject.i"
%include "distance/Distance.h"
