%{
 #include "kernel/WeightedCommWordStringKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* w, int32_t d)};

%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/WeightedCommWordStringKernel.h" 
