%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* src, INT num_feat, INT num_vec)};

%rename(CharFeatures) CCharFeatures;

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"
