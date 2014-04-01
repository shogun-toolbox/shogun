/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#ifndef _WEIGHTED_MAJORITY_VOTE_H_
#define _WEIGHTED_MAJORITY_VOTE_H_

#include <shogun/lib/config.h>

#include <shogun/ensemble/CombinationRule.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
	/**
	 * @brief Weighted Majority Vote implementation.
	 *
	 * For a given feature vector the combined value is going to be:
	 * \f[
	 *	 label = max_{j=0..C} \sum_{i=0}^{N} w_i d_{i,j}
	 * \f]
	 * where \f$N\f$ is the number of Machines in the ensemble
	 * \f$w_i\f$ is the weight of ith Machine
	 * \f$d_{i,j}\f$ decision of the ith Machine for jth class.
	 * \f$C\f$ is the number of classes
	 *
	 */
	class CWeightedMajorityVote : public CCombinationRule
	{
		public:
			/**
			 * Default ctor
			 */
			CWeightedMajorityVote();

			/**
			 * CWeightedMajorityVote constructor
			 *
			 * @param weights a vector of weights, where the nth element is the
			 * weight of the nth Machine in ensemble.
			 */
			CWeightedMajorityVote(SGVector<float64_t>& weights);

			virtual ~CWeightedMajorityVote();

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

			/**
			 * Set weight vector for the labels
			 *
			 * @param w weights
			 */
			void set_weights(SGVector<float64_t>& w);

			/**
			 * Get weight vector
			 *
			 * @return weight vector
			 */
			SGVector<float64_t> get_weights() const;

			/** name **/
			virtual const char* get_name() const { return "WeightedMajorityVote"; }

		protected:
			/**
			 * Weigthed majority voting implementation
			 *
			 * @param ensemble_result a vector of outputs in the ensemble for a given feature vector.
			 * @return the combined value
			 */
			virtual float64_t weighted_combine(const SGVector<float64_t>& ensemble_result) const;

			/** the weight vector */
			mutable SGVector<float64_t> m_weights;

		private:
			void init();
			void register_parameters();
	};
}

#endif /* _WEIGHTED_MAJORITY_VOTE_H_ */
