%{
 #include "features/ShortFeatures.h"
%}

%include "lib/swig_typemaps.i"

%apply (SHORT* IN_ARRAY2, INT DIM1, INT DIM2) {(SHORT* src, INT num_feat, INT num_vec)};

%rename(ShortFeatures) CShortFeatures;

%include "features/ShortFeatures.h"
