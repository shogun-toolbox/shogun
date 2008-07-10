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

%rename(ShortRealFeatures) CShortRealFeatures;

%include "features/SimpleFeatures.i"
%include "features/ShortRealFeatures.h"
