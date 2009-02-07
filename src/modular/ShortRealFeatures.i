%{
#include "features/SimpleFeatures.h"
#include "features/ShortRealFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (float32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float32_t** dst, int32_t* d1, int32_t* d2)};

%rename(ShortRealFeatures) CShortRealFeatures;

%include "features/SimpleFeatures.i"
%include "features/ShortRealFeatures.h"
