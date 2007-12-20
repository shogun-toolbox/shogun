/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTRIBUTION_H___
#define _DISTRIBUTION_H___

#include "features/Features.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

class CDistribution : public CSGObject
{
	public:
		CDistribution();
		virtual ~CDistribution();

		virtual bool train()=0;

		/// get number of parameters in model
		virtual INT get_num_model_parameters()=0;

		/// get number of parameters in model that are relevant,
		/// i.e. > ALMOST_NEG_INFTY
		virtual INT get_num_relevant_model_parameters();

		//get model parameter (logarithmic)
		virtual DREAL get_log_model_parameter(INT num_param)=0;

		//get derivative of likelihood function (logarithmic)
		virtual DREAL get_log_derivative(INT num_param, INT num_example)=0;
		
		/// compute log likelihood for example
		virtual DREAL get_log_likelihood_example(INT num_example)=0;

		/// compute log likelihood for whole sample
		virtual DREAL get_log_likelihood_sample();

		/// compute log likelihood for each example
		virtual DREAL* get_log_likelihood();

		//get model parameter
		virtual inline DREAL get_model_parameter(INT num_param)
		{
			return exp(get_log_model_parameter(num_param));
		}

		//get derivative of likelihood function
		virtual inline DREAL get_derivative(INT num_param, INT num_example)
		{
			return exp(get_log_derivative(num_param, num_example));
		}

		/// compute likelihood for example
		virtual inline DREAL get_likelihood_example(INT num_example)
		{
			return exp(get_log_likelihood_example(num_example));
		}

		/// set and get feature vectors
		virtual inline void set_features(CFeatures* f) { features=f; }
		virtual inline CFeatures* get_features() { return features; }

		/// set and get pseudo count
		virtual inline void set_pseudo_count(DREAL pseudo) { pseudo_count=pseudo; }
		virtual inline DREAL get_pseudo_count() { return pseudo_count; }

	protected:
		CFeatures* features;
		DREAL pseudo_count;
};
#endif

