%{
 #include "kernel/LinearKernel.h"
%}

%rename(LinearKernel) CLinearKernel;

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* src_w, INT src_w_dim)};
#endif

%include "kernel/LinearKernel.h"
