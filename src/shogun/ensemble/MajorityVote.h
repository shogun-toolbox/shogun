/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#ifndef _MAJORITY_VOTE_H_
#define _MAJORITY_VOTE_H_

#include <shogun/lib/config.h>

#include <shogun/ensemble/WeightedMajorityVote.h>

namespace shogun
{
	/**
	 * @brief CMajorityVote is a CWeightedMajorityVote combiner, where each
	 * Machine's weight in the ensemble is 1.0
	 */
	class CMajorityVote : public CWeightedMajorityVote
	{
		public:
			CMajorityVote();

			virtual ~CMajorityVote();

			/**
			 * Combines a matrix of an ensemble of Machines output, where each
			 * column is a given Machine's classification/regression output
			 * for the input Features.
			 *
			 * @param ensemble_result SGMatrix
			 * @return a vector where the nth element is the combined value of the Machines for the nth feature vector
			 */
			virtual SGVector<float64_t> combine(const SGMatrix<float64_t>& ensemble_result) const;

			/**
			 * Combines a vector of Machine ouputs for a given feature vector.
			 * The nth element of the vector is the nth Machine's output in the ensemble.
			 *
			 * @param ensemble_result SGVector<float64_t> with the Machine's output
			 * @return the combined value
			 */
			virtual float64_t combine(const SGVector<float64_t>& ensemble_result) const;

			/** name **/
			virtual const char* get_name() const { return "MajorityVote"; }
	};
}

#endif /* _MAJORITY_VOTE_H_ */
