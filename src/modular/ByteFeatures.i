%{
#include "features/SimpleFeatures.h"
#include "features/ByteFeatures.h" 
%}

%include "lib/swig_typemaps.i"

%apply (uint8_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint8_t* src, int32_t num_feat, int32_t num_vec)};

%rename(ByteFeatures) CByteFeatures;

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

