%{
 #include "features/ShortFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/ShortFeatures.h" 

%pythoncode %{
  class ShortFeatures(CShortFeatures):
     def __init__(self,p1): 
        CShortFeatures.__init__(self,0)
        self.set_feature_matrix(p1)
%}
