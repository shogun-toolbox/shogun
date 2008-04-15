%{
#include "lib/common.h"
#include "kernel/WeightedDegreeStringKernel.h"
%}

%include "lib/swig_typemaps.i"

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_weights, INT d)};

%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

%include "kernel/WeightedDegreeStringKernel.h"

