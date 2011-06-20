/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _GMM_H__
#define _GMM_H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "distributions/Distribution.h"
#include "distributions/Gaussian.h"
#include "lib/common.h"

namespace shogun
{
/** @brief Gaussian distribution interface.
 *
 * Takes input of number of Gaussians to fit
 */
class CGMM : public CDistribution
{
	public:
		/** default constructor */
		CGMM();
		/** constructor
		 *
		 * @param n number of Gaussians
		 * @param max_iter maximum iterations
		 * @param min_change minimal expected log likelihood change
		 */
		CGMM(int32_t n, int32_t max_iter, float64_t min_change);
		virtual ~CGMM();

		/** cleanup */
		void cleanup();

		/** learn distribution
		 *
		 * @param data training data
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get number of parameters in model
		 *
		 * @return number of parameters in model
		 */
		virtual int32_t get_num_model_parameters();

		/** get model parameter (logarithmic)
		 *
		 * @return model parameter (logarithmic) if num_param < m_dim returns
		 * an element from the mean, else return an element from the covariance
		 */
		virtual float64_t get_log_model_parameter(int32_t num_param);

		/** get partial derivative of likelihood function (logarithmic)
		 *
		 * @param num_param derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		virtual float64_t get_log_derivative(
			int32_t num_param, int32_t num_example);

		/** compute log likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return log likelihood for example
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example);

		/** compute likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return likelihood for example
		 */
		virtual float64_t get_likelihood_example(int32_t num_example)
		{
			return CMath::exp(get_log_likelihood_example(num_example));
		}

		/** get nth mean
		 *
		 * @param num which mean to retrieve
		 */
		virtual SGVector<float64_t> get_nth_mean(int32_t num)
		{
			ASSERT(num<m_n);
			return m_components[num]->get_mean();
		}

		/** get nth cov
		 *
		 * @param num which covariance to retrieve
		 */
		virtual SGMatrix<float64_t> get_nth_cov(int32_t num)
		{
			ASSERT(num<m_n);
			return m_components[num]->get_cov();
		}

		/** get coefficients
		 *
		 */
		virtual SGVector<float64_t> get_coef()
		{
			return SGVector<float64_t>(m_coefficients,m_coef_size);
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "GMM"; }

	private:
		/** Initialize parameters for serialization */
		void register_params();

	protected:
		/** Mixture components */
		CGaussian** m_components;
		/** Number of mixture components */
		int32_t m_n;
		/** Mixture coefficients */
		float64_t* m_coefficients;
		/** Coefficient vector size */
		int32_t m_coef_size;
		/** Maximum number of iterations */
		int32_t m_max_iter;
		/** Minimum expected log-likelihood change */
		float64_t m_minimal_change;
};
}
#endif //HAVE_LAPACK
#endif //_GMM_H__
