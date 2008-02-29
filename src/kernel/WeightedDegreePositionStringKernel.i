%{
#include "kernel/WeightedDegreePositionStringKernel.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* weights, INT d, INT len)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* shifts, INT len)};
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* distrib, INT num_sym, INT num_feat)};
%feature("autodoc", "get_POIM2(self) -> [] of float") get_POIM2;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** poim, INT* result_len)};
#endif

%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;

%include "kernel/WeightedDegreePositionStringKernel.h"

