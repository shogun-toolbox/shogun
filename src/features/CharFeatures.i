%{
#include "features/CharFeatures.h" 
%}

%include "lib/common.i"
%include "lib/numpy.i"

/*%rename(CCharFeatures_) CCharFeatures;*/

%include "features/CharFeatures.h"
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* feature_matrix, int d1, int d2)};

%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL*, INT)}; 
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* series, INT size)};

/*
%pythoncode %{
  class CCharFeatures(CCharFeatures_):
     def __init__(self,p1,p2): 
        CCharFeatures_.__init__(self,p1,0)
        self.set_feature_matrix(p2)
%}
*/  
