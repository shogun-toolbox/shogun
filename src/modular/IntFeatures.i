%{
#include "features/SimpleFeatures.h"
#include "features/IntFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** dst, int32_t* d1, int32_t* d2)};

%rename(IntFeatures) CIntFeatures;

%include "features/SimpleFeatures.i"
%include "features/IntFeatures.h"
