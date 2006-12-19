%{
 #include "features/WordFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/WordFeatures.h" 

#ifdef HAVE_PYTHON
%pythoncode %{
  class WordFeatures(CWordFeatures):
     def __init__(self,p1): 
        CWordFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
#endif
