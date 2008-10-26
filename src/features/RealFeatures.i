%{
#include "features/SimpleFeatures.h"
#include "features/RealFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (DREAL* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(DREAL* src, int32_t num_feat, int32_t num_vec)};
%apply (DREAL** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(DREAL** dst, int32_t* d1, int32_t* d2)};

%rename(RealFeatures) CRealFeatures;

%include "features/SimpleFeatures.i"
%include "features/RealFeatures.h"
