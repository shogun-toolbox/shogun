/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
%define DOCSTR
"The `Distribution` module gathers all distributions available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Distribution
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_PYTHON
%feature("autodoc", "get_log_likelihood(self) -> numpy 1dim array of float") get_log_likelihood;
%feature("autodoc", "get_histogram(self) -> numpy 1dim array of float") get_histogram;
%feature("autodoc", "get_log_transition_probs(self) -> numpy 1dim array of %float") get_log_transition_probs;
%feature("autodoc", "get_transition_probs(self) -> numpy 1dim array of %float") get_transition_probs;
#endif

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Distribution_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Features_includes.i"
%include "Distribution_includes.i"
%include "Preprocessor_includes.i"

%import "Features.i"

/* Remove C Prefix */
%rename(BaseDistribution) CDistribution;
%rename(Histogram) CHistogram;
%rename(HMM) CHMM;
%rename(GHMM) CGHMM;
%rename(LinearHMM) CLinearHMM;
%rename(PositionalPWM) CPositionalPWM;
%rename(Gaussian) CGaussian;
%rename(GMM) CGMM;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distributions/Distribution.h>
%include <shogun/distributions/Histogram.h>
%include <shogun/distributions/HMM.h>
%include <shogun/distributions/GHMM.h>
%include <shogun/distributions/LinearHMM.h>
%include <shogun/distributions/PositionalPWM.h>
%include <shogun/distributions/Gaussian.h>
%include <shogun/clustering/GMM.h>
