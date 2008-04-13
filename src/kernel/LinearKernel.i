%{
 #include "kernel/LinearKernel.h"
%}

%rename(LinearKernel) CLinearKernel;

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* src_w, INT src_w_dim)};

%include "kernel/LinearKernel.h"
