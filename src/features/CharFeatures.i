%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (char* IN_ARRAY2, INT DIM1, INT DIM2) {(char* src, INT num_feat, INT num_vec)};

%rename(CharFeatures) CCharFeatures;

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"
