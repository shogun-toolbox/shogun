%{
 #include "features/WordFeatures.h"
%}

%include "lib/swig_typemaps.i"

%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD* src, INT num_feat, INT num_vec)};

%rename(WordFeatures) CWordFeatures;

%include "features/WordFeatures.h"
