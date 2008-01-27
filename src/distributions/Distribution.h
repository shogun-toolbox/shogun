/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTRIBUTION_H___
#define _DISTRIBUTION_H___

#include "features/Features.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

/** class Distribution */
class CDistribution : public CSGObject
{
	public:
		/** default constructor */
		CDistribution();
		virtual ~CDistribution();

		/** train distribution
		 *
		 * abstrace base method
		 *
		 * @return if training was successful
		 */
		virtual bool train()=0;

		/** get number of parameters in model
		 *
		 * abstract base method
		 *
		 * @return number of parameters in model
		 */
		virtual INT get_num_model_parameters()=0;

		/** get number of parameters in model that are relevant,
		 * i.e. > ALMOST_NEG_INFTY
		 *
		 * @return number of relevant model parameters
		 */
		virtual INT get_num_relevant_model_parameters();

		/** get model parameter (logarithmic)
		 *
		 * abstrac base method
		 *
		 * @return model parameter (logarithmic)
		 */
		virtual DREAL get_log_model_parameter(INT num_param)=0;

		/** get derivative of likelihood function (logarithmic)
		 *
		 * abstract base method
		 *
		 * @param num_param which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		virtual DREAL get_log_derivative(INT num_param, INT num_example)=0;

		/** compute log likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return log likeliehood for example
		 */
		virtual DREAL get_log_likelihood_example(INT num_example)=0;

		/** compute log likelihood for whole sample
		 *
		 * @return log likelihood for whole sample
		 */
		virtual DREAL get_log_likelihood_sample();

		/** compute log likelihood for each example
		 *
		 * @param dst where likelihood will be stored
		 * @param num where number of likelihoods will be stored
		 */
		virtual void get_log_likelihood(DREAL** dst, INT *num);

		/** get model parameter
		 *
		 * @param num_param which param
		 * @return model parameter
		 */
		virtual inline DREAL get_model_parameter(INT num_param)
		{
			return exp(get_log_model_parameter(num_param));
		}

		/** get derivative of likelihood function
		 *
		 * @param num_param which param
		 * @param num_example which example
		 * @return derivative of likelihood function
		 */
		virtual inline DREAL get_derivative(INT num_param, INT num_example)
		{
			return exp(get_log_derivative(num_param, num_example));
		}

		/** compute likelihood for example
		 *
		 * @param num_example which example
		 * @return likelihood for example
		 */
		virtual inline DREAL get_likelihood_example(INT num_example)
		{
			return exp(get_log_likelihood_example(num_example));
		}

		/** set feature vectors
		 *
		 * @param f new feature vectors
		 */
		virtual inline void set_features(CFeatures* f) { features=f; }

		/** get feature vectors
		 *
		 * @return feature vectors
		 */
		virtual inline CFeatures* get_features() { return features; }

		/** set pseudo count
		 *
		 * @param pseudo new pseudo count
		 */
		virtual inline void set_pseudo_count(DREAL pseudo) { pseudo_count=pseudo; }

		/** get pseudo count
		 *
		 * @return pseudo count
		 */
		virtual inline DREAL get_pseudo_count() { return pseudo_count; }

	protected:
		/** feature vectors */
		CFeatures* features;
		/** pseudo count */
		DREAL pseudo_count;
};
#endif

