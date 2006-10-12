%{
 #include "features/ByteFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/ByteFeatures.h"

%pythoncode %{
  class ByteFeatures(CByteFeatures):
     def __init__(self,p1,p2): 
        CByteFeatures.__init__(self,p2,0)
        self.set_feature_matrix(p1)
%}
