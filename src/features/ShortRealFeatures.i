%{
#include "features/SimpleFeatures.h"
#include "features/ShortRealFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (SHORTREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(SHORTREAL* src, INT num_feat, INT num_vec)};
%apply (SHORTREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(SHORTREAL** dst, INT* d1, INT* d2)};

%include "features/SimpleFeatures.i"
%include "features/ShortRealFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class ShortRealFeatures(CShortRealFeatures):
     def __init__(self,p1):
        CShortRealFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
#endif
