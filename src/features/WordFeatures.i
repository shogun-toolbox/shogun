%{
 #include "features/WordFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
#endif

%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD* src, INT num_feat, INT num_vec)};
%apply (WORD** ARGOUT2, INT* DIM1, INT* DIM2) {(WORD** dst, INT* d1, INT* d2)};

%rename(WordFeatures) CWordFeatures;

%include "features/WordFeatures.h"
