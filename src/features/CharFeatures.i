%include "lib/common.i"

%{
#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (CHAR* IN_ARRAY2, INT DIM1, INT DIM2) {(CHAR* src, INT num_feat, INT num_vec)};

%include "features/SimpleFeatures.i"
%include "features/CharFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class CharFeatures(CCharFeatures):
     def __init__(self,features,alphabet): 
        CCharFeatures.__init__(self,alphabet)
        self.copy_feature_matrix(features)
%}
#endif
