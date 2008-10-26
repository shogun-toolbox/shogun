%{
 #include "features/ShortFeatures.h"
%}

%include "lib/swig_typemaps.i"

%apply (SHORT* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(SHORT* src, int32_t num_feat, int32_t num_vec)};

%rename(ShortFeatures) CShortFeatures;

%include "features/ShortFeatures.h"
