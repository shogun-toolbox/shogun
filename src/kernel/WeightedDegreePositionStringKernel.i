%{
#include "kernel/WeightedDegreePositionStringKernel.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_POIM2(self) -> [] of float") get_POIM2;
#endif

%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* weights, int32_t d, int32_t len)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* shifts, int32_t len)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* distrib, int32_t num_sym, int32_t num_feat)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** poim, int32_t* result_len)};

%rename(WeightedDegreePositionStringKernel) CWeightedDegreePositionStringKernel;

%include "kernel/WeightedDegreePositionStringKernel.h"

