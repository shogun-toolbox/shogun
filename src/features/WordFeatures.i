%{
 #include "features/WordFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
#endif

%apply (uint16_t* IN_ARRAY2, INT DIM1, INT DIM2) {(uint16_t* src, INT num_feat, INT num_vec)};
%apply (uint16_t** ARGOUT2, INT* DIM1, INT* DIM2) {(uint16_t** dst, INT* d1, INT* d2)};

%rename(WordFeatures) CWordFeatures;

%include "features/WordFeatures.h"
