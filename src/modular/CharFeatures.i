%include "common.i"

%{
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/CharFeatures.h>
%}

%include "swig_typemaps.i"

%apply (char* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(char* src, int32_t num_feat, int32_t num_vec)};

%rename(CharFeatures) CCharFeatures;

%include "SimpleFeatures.i"
%include <shogun/features/CharFeatures.h>
