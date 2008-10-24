%{
#include "features/SimpleFeatures.h"
#include "features/ByteFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (uint8_t* IN_ARRAY2, INT DIM1, INT DIM2) {(uint8_t* src, INT num_feat, INT num_vec)};

%rename(ByteFeatures) CByteFeatures;

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

