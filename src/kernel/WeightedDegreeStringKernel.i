%{
#include "lib/common.h"
#include "kernel/WeightedDegreeStringKernel.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_weights, INT d)};

%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

%include "kernel/WeightedDegreeStringKernel.h"

