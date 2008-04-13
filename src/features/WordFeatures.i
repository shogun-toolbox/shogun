%{
 #include "features/WordFeatures.h"
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
#endif

#ifdef HAVE_OCTAVE
%include "lib/octave_typemaps.i"
#endif

%apply (WORD* IN_ARRAY2, INT DIM1, INT DIM2) {(WORD* src, INT num_feat, INT num_vec)};

%include "features/WordFeatures.h"

#ifdef HAVE_PYTHON
%pythoncode %{
  class WordFeatures(CWordFeatures):
     def __init__(self,p1):
        CWordFeatures.__init__(self,0)
        self.copy_feature_matrix(p1)
%}
#endif
