%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* src, INT num_feat, INT num_vec)};
#endif

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class CharFeatures(CCharFeatures):
     def __init__(self,p1,p2): 
        CCharFeatures.__init__(self,p2,0)
        self.copy_feature_matrix(p1)
%}
#endif
