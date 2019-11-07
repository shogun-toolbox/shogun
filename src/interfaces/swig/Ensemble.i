/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::CombinationRule)

%shared_ptr(shogun::WeightedMajorityVote)

%shared_ptr(shogun::MajorityVote)

%shared_ptr(shogun::MeanRule)


/* Include Class Headers to make them visible from within the target language */
%include <shogun/ensemble/CombinationRule.h>

%include <shogun/ensemble/WeightedMajorityVote.h>

%include <shogun/ensemble/MajorityVote.h>

%include <shogun/ensemble/MeanRule.h>
