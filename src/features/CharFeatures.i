%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

%include "lib/numpy.i"
%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* src, INT num_feat, INT num_vec)};

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

%pythoncode %{
  class CharFeatures(CCharFeatures):
     def __init__(self,p1,p2): 
        CCharFeatures.__init__(self,p2,0)
        self.copy_feature_matrix(p1)
%}
