%define DOCSTR
"The `Distribution` module gathers all distributions available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distribution
%{
#define SWIG_FILE_WITH_INIT
#include "distributions/Distribution.h"
%}
%rename(Distribution) CDistribution;
%feature("autodoc","0");

%include "lib/common.i"
%include "lib/swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "distributions/Distribution_doxygen.i"
#endif


#ifdef HAVE_PYTHON
%init %{
	import_array();
%}
%feature("autodoc", "get_log_likelihood(self) -> numpy 1dim array of float") get_log_likelihood;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num)};

%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"

%include "distributions/Distribution.h"
%include "distributions/histogram/Histogram.i"
%include "distributions/hmm/HMM.i"
%include "distributions/hmm/GHMM.i"
%include "distributions/hmm/LinearHMM.i"
