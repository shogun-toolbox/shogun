%{
 #include "features/ShortFeatures.h"
%}

%include "lib/swig_typemaps.i"

%apply (SHORT* IN_ARRAY2, INT DIM1, INT DIM2) {(SHORT* src, INT num_feat, INT num_vec)};

%include "features/ShortFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class ShortFeatures(CShortFeatures):
     def __init__(self,p1):
        CShortFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
#endif
