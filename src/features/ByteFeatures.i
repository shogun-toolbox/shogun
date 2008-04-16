%{
#include "features/SimpleFeatures.h"
#include "features/ByteFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE* src, INT num_feat, INT num_vec)};

%rename(ByteFeatures) CByteFeatures;

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

