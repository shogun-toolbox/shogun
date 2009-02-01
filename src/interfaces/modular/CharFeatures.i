%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (char* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(char* src, int32_t num_feat, int32_t num_vec)};

%rename(CharFeatures) CCharFeatures;

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"
