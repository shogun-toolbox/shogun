%{
 #include "features/ShortFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "features/ShortFeatures.h" 

#ifdef HAVE_PYTHON
%pythoncode %{
  class ShortFeatures(CShortFeatures):
     def __init__(self,p1): 
        CShortFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
#endif
