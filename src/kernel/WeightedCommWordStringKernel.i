%{
 #include "kernel/WeightedCommWordStringKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* w, int32_t d)};

%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/WeightedCommWordStringKernel.h" 
