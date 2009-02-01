%{
#include "distributions/histogram/Histogram.h"
%}


#ifdef HAVE_PYTHON
%feature("autodoc", "get_histogram(self) -> numpy 1dim array of float") get_histogram;
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src, int32_t num)};
#endif

%rename(Histogram) CHistogram;

%include "distributions/histogram/Histogram.h"
