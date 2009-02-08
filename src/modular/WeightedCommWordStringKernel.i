%{
 #include <shogun/kernel/WeightedCommWordStringKernel.h>
%}

%include "swig_typemaps.i"

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* w, int32_t d)};

%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;

%include "StringKernel.i" 
%include <shogun/kernel/WeightedCommWordStringKernel.h>
