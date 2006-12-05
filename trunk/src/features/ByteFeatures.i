%{
#include "features/SimpleFeatures.h"
#include "features/ByteFeatures.h" 
%}

%include "lib/numpy.i"
%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE* src, INT num_feat, INT num_vec)};

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

%pythoncode %{
  class ByteFeatures(CByteFeatures):
     def __init__(self,p1,p2): 
        CByteFeatures.__init__(self,p2,0)
        self.copy_feature_matrix(p1)
%}
