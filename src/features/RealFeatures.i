%{
#include "features/SimpleFeatures.h"
#include "features/RealFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
#endif

%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* src, INT num_feat, INT num_vec)};
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** dst, INT* d1, INT* d2)};

%rename(RealFeatures) CRealFeatures;

%include "features/SimpleFeatures.i"
%include "features/RealFeatures.h"
