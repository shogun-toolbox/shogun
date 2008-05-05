%{
 #include "kernel/WeightedCommWordStringKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* w, INT d)};

%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/WeightedCommWordStringKernel.h" 
