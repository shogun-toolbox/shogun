%{
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/RealFeatures.h>
%}

%include "swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* d1, int32_t* d2)};

%rename(RealFeatures) CRealFeatures;

%include "SimpleFeatures.i"
%include <shogun/features/RealFeatures.h>
