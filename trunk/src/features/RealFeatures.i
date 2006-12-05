%{
#include "features/SimpleFeatures.h" 
#include "features/RealFeatures.h" 
%}

%include "lib/numpy.i"
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* src, INT num_feat, INT num_vec)};
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** dst, INT* d1, INT* d2)};

%include "features/SimpleFeatures.i"
%include "features/RealFeatures.h"

%pythoncode %{
  class RealFeatures(CRealFeatures):
     def __init__(self,p1): 
        CRealFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
