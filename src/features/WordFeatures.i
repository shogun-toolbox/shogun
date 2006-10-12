%{
 #include "features/WordFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/WordFeatures.h" 

%pythoncode %{
  class WordFeatures(CWordFeatures):
     def __init__(self,p1): 
        CWordFeatures.__init__(self,0)
        self.set_feature_matrix(p1)
%}
