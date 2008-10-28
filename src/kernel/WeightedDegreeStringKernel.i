%{
#include "lib/common.h"
#include "kernel/WeightedDegreeStringKernel.h"
%}

%include "lib/swig_typemaps.i"

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_weights, int32_t d)};

%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

%include "kernel/WeightedDegreeStringKernel.h"

