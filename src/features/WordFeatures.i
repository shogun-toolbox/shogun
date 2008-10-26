%{
 #include "features/WordFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
#endif

%apply (uint16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint16_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint16_t** dst, int32_t* d1, int32_t* d2)};

%rename(WordFeatures) CWordFeatures;

%include "features/WordFeatures.h"
