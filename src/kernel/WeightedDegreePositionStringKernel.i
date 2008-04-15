%{
#include "kernel/WeightedDegreePositionStringKernel.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_POIM2(self) -> [] of float") get_POIM2;
#endif

%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* weights, INT d, INT len)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* shifts, INT len)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* distrib, INT num_sym, INT num_feat)};
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** poim, INT* result_len)};

%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;

%include "kernel/WeightedDegreePositionStringKernel.h"

