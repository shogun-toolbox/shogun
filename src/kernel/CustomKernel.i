%{
 #include "kernel/CustomKernel.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(const DREAL* km, INT rows, INT cols)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(const DREAL* km, INT cols)};
#endif

%rename(CustomKernel) CCustomKernel;

%include "kernel/CustomKernel.h" 
