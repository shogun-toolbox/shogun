%define DOCSTR
"The `Distribution` module gathers all distributions available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distribution
%{
#define SWIG_FILE_WITH_INIT
#include <shogun/distributions/Distribution.h>
%}
%rename(Distribution) CDistribution;
%feature("autodoc","0");

%include "common.i"
%include "swig_typemaps.i"

#ifdef HAVE_DOXYGEN
%include "Distribution_doxygen.i"
#endif


#ifdef HAVE_PYTHON
%init %{
	import_array();
%}
%feature("autodoc", "get_log_likelihood(self) -> numpy 1dim array of float") get_log_likelihood;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst, int32_t* num)};

%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"

%include <shogun/distributions/Distribution.h>
%include "Histogram.i"
%include "HMM.i"
%include "GHMM.i"
%include "LinearHMM.i"
