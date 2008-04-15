%{
 #include "kernel/CustomKernel.h" 
%}

%include "lib/swig_typemaps.i"

%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(const DREAL* km, INT rows, INT cols)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(const DREAL* km, INT len)};

%rename(CustomKernel) CCustomKernel;

%include "kernel/CustomKernel.h" 
