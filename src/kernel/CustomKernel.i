%{
 #include "kernel/CustomKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(const float64_t* km, int32_t rows, int32_t cols)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(const float64_t* km, int32_t len)};

%rename(CustomKernel) CCustomKernel;

%include "kernel/CustomKernel.h" 
