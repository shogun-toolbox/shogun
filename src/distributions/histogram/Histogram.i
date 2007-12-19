%{
#include "distributions/histogram/Histogram.h"
%}


#ifdef HAVE_PYTHON
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst, INT* num)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* src, INT num)};
#endif

%rename(Histogram) CHistogram;

%include "distributions/histogram/Histogram.h"
