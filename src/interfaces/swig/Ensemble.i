/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg, 2013 Viktor Gal
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
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
