%{
#include "lib/common.h"
#include "kernel/WeightedDegreeStringKernel.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_weights, INT d)};
#endif

%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

%include "kernel/WeightedDegreeStringKernel.h"

