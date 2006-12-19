%{
#include "features/SimpleFeatures.h"
#include "features/ByteFeatures.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/numpy.i"
%apply (BYTE* IN_ARRAY2, INT DIM1, INT DIM2) {(BYTE* src, INT num_feat, INT num_vec)};
#endif

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class ByteFeatures(CByteFeatures):
     def __init__(self,p1,p2): 
        CByteFeatures.__init__(self,p2,0)
        self.copy_feature_matrix(p1)
%}
#endif
