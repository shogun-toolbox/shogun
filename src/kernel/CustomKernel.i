%{
 #include "kernel/CustomKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (DREAL* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(const DREAL* km, int32_t rows, int32_t cols)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(const DREAL* km, int32_t len)};

%rename(CustomKernel) CCustomKernel;

%include "kernel/CustomKernel.h" 
