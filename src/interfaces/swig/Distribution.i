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

%shared_ptr(shogun::Distribution)
SHARED_RANDOM_INTERFACE(shogun::Distribution)
%shared_ptr(shogun::HMM)
%shared_ptr(shogun::PositionalPWM)
%shared_ptr(shogun::Gaussian)
%shared_ptr(shogun::GMM)
%shared_ptr(shogun::ProbabilityDistribution)
SHARED_RANDOM_INTERFACE(shogun::ProbabilityDistribution)
%shared_ptr(shogun::KernelDensity)
%shared_ptr(shogun::GaussianDistribution)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/distributions/Distribution.h>

/** Instantiate RandomMixin */
RANDOM_INTERFACE(Distribution)

%include <shogun/distributions/HMM.h>
%include <shogun/distributions/PositionalPWM.h>
%include <shogun/distributions/Gaussian.h>
%include <shogun/distributions/KernelDensity.h>
%include <shogun/clustering/GMM.h>
%include <shogun/distributions/classical/ProbabilityDistribution.h>
RANDOM_INTERFACE(ProbabilityDistribution)
%include <shogun/distributions/classical/GaussianDistribution.h>
