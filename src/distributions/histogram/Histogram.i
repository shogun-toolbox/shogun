%{
#include "distributions/histogram/Histogram.h"
%}


#ifdef HAVE_PYTHON
%feature("autodoc", "get_histogram(self) -> numpy 1dim array of float") get_histogram;
%apply (DREAL** ARGOUT1, int32_t* DIM1) {(DREAL** dst, int32_t* num)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* src, int32_t num)};
#endif

%rename(Histogram) CHistogram;

%include "distributions/histogram/Histogram.h"
