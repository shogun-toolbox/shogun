%{
 #include "kernel/LinearKernel.h"
%}

%rename(LinearKernel) CLinearKernel;

%include "lib/swig_typemaps.i"

%apply (DREAL** ARGOUT1, int32_t* DIM1) {(DREAL** dst_w, int32_t* dst_dims)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* src_w, int32_t src_w_dim)};

%include "kernel/LinearKernel.h"
