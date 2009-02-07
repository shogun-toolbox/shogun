%{
 #include "kernel/LinearKernel.h"
%}

%rename(LinearKernel) CLinearKernel;

%include "lib/swig_typemaps.i"

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst_w, int32_t* dst_dims)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src_w, int32_t src_w_dim)};

%include "kernel/LinearKernel.h"
