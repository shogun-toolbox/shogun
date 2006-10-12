%{
    #include "features/RealFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/RealFeatures.h"

%pythoncode %{
  class RealFeatures(CRealFeatures):
     def __init__(self,p1): 
        CRealFeatures.__init__(self,0)
        self.set_feature_matrix(p1)
%}
