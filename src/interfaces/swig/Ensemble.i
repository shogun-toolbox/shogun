/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(CombinationRule) CCombinationRule;

%rename(WeightedMajorityVote) CWeightedMajorityVote;

%rename(MajorityVote) CMajorityVote;

%rename(MeanRule) CMeanRule;


/* Include Class Headers to make them visible from within the target language */
%include <shogun/ensemble/CombinationRule.h>

%include <shogun/ensemble/WeightedMajorityVote.h>

%include <shogun/ensemble/MajorityVote.h>

%include <shogun/ensemble/MeanRule.h>
