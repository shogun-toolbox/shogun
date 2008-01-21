%define DOCSTR
"The `Distribution` module gathers all distributions available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") Distribution
%{
#define SWIG_FILE_WITH_INIT
#include "distributions/Distribution.h"
%}
%feature("director") CDistribution;
%rename(Distribution) CDistribution;
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "distributions/Distribution_doxygen.i"
#endif


#ifdef HAVE_PYTHON
%init %{
	import_array();
%}

%include "lib/common.i"
%include "lib/python_typemaps.i"

%feature("autodoc", "get_log_likelihood(self) -> numpy 1dim array of float") get_log_likelihood;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst, INT* num)};
#endif


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
