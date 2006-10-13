%include "lib/common.i"

%{
#include "features/CharFeatures.h" 
%}


%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

%pythoncode %{
  class CharFeatures(CCharFeatures):
     def __init__(self,p1,p2): 
        CCharFeatures.__init__(self,p2,0)
        self.copy_feature_matrix(p1)
%}
