/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_log_likelihood(self) -> numpy 1dim array of float") get_log_likelihood;
%feature("autodoc", "get_histogram(self) -> numpy 1dim array of float") get_histogram;
%feature("autodoc", "get_log_transition_probs(self) -> numpy 1dim array of %float") get_log_transition_probs;
%feature("autodoc", "get_transition_probs(self) -> numpy 1dim array of %float") get_transition_probs;
#endif

/* Remove C Prefix */
%rename(Distribution) CDistribution;
%rename(Histogram) CHistogram;
%rename(HMM) CHMM;
%rename(LinearHMM) CLinearHMM;
%rename(PositionalPWM) CPositionalPWM;
%rename(Gaussian) CGaussian;
%rename(GMM) CGMM;
%rename(KernelDensity) CKernelDensity;
%rename(GaussianDistribution) CGaussianDistribution;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distributions/Distribution.h>
%include <shogun/distributions/Histogram.h>
%include <shogun/distributions/HMM.h>
%include <shogun/distributions/LinearHMM.h>
%include <shogun/distributions/PositionalPWM.h>
%include <shogun/distributions/Gaussian.h>
%include <shogun/distributions/KernelDensity.h>
%include <shogun/clustering/GMM.h>
%include <shogun/distributions/classical/ProbabilityDistribution.h>
%include <shogun/distributions/classical/GaussianDistribution.h>
