%{
 #include <shogun/features/ShortFeatures.h>
%}

%include "swig_typemaps.i"

%apply (int16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int16_t* src, int32_t num_feat, int32_t num_vec)};

%rename(ShortFeatures) CShortFeatures;

%include <shogun/features/ShortFeatures.h>
