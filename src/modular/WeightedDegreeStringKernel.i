%{
#include <shogun/lib/common.h>
#include <shogun/kernel/WeightedDegreeStringKernel.h>
%}

%include "swig_typemaps.i"

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* p_weights, int32_t d)};

%rename(WeightedDegreeStringKernel) CWeightedDegreeStringKernel;

%include <shogun/kernel/WeightedDegreeStringKernel.h>
