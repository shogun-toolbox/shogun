/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef _GAUSSIAN_H__
#define _GAUSSIAN_H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "distributions/Distribution.h"
#include "features/DotFeatures.h"
#include "lib/common.h"
#include "lib/lapack.h"
#include "lib/Mathematics.h"

namespace shogun
{
class CDotFeatures;
/** @brief Gaussian distribution interface.
 *
 * Takes as input a mean vector and covariance matrix.
 * Also possible to train from data.
 * Likelihood is computed using the Gaussian PDF \f$(2\pi)^{-\frac{k}{2}}|\Sigma|^{-\frac{1}{2}}e^{-\frac{1}{2}(x-\mu)'\Sigma^{-1}(x-\mu)}\f$
 */
class CGaussian : public CDistribution
{
	public:
		/** default constructor */
		CGaussian();
		/** constructor
		 *
		 * @param mean mean of the Gaussian
		 * @param mean_length
		 * @param cov covariance of the Gaussian
		 * @param cov_rows
		 * @param cov_cols
		 */
		CGaussian(float64_t* mean, int32_t mean_length,
					float64_t* cov, int32_t cov_rows, int32_t cov_cols);
		virtual ~CGaussian();

		/** Compute the inverse covariance and constant part */
		void init();

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

		/** compute PDF
		 *
		 * @param point
		 * @param point_len
		 * @return computed PDF
		 */
		virtual inline float64_t compute_PDF(float64_t* point, int32_t point_len)
		{
			return CMath::exp(compute_log_PDF(point, point_len));
		}

		/** compute log PDF
		 *
		 * @param point
		 * @param point_len
		 * @return computed log PDF
		 */
		virtual float64_t compute_log_PDF(float64_t* point, int32_t point_len);

		/** get mean
		 *
		 * @param mean copy of the mean
		 * @param mean_length
		 */
		virtual inline void get_mean(float64_t** mean, int32_t* mean_length)
		{
			*mean = new float64_t[m_mean_length];
			memcpy(*mean, m_mean, sizeof(float64_t)*m_mean_length);
			*mean_length = m_mean_length;
		}

		/** set mean
		 *
		 * @param mean new mean
		 * @param mean_length has to match current mean length
		 */
		virtual inline void set_mean(float64_t* mean, int32_t mean_length)
		{
			ASSERT(mean_length == m_mean_length);
			memcpy(m_mean, mean, sizeof(float64_t)*m_mean_length);
		}

		/** get cov
		 *
		 * @param cov copy of the cov
		 * @param cov_rows
		 * @param cov_cols
		 */
		virtual inline void get_cov(float64_t** cov, int32_t* cov_rows, int32_t* cov_cols)
		{
			*cov = new float64_t[m_cov_rows*m_cov_cols];
			memcpy(*cov, m_cov, sizeof(float64_t)*m_cov_rows*m_cov_cols);
			*cov_rows = m_cov_rows;
			*cov_cols = m_cov_cols;
		}

		/** set cov
		 *
		 * @param cov new cov
		 * @param cov_rows has to match current cov rows
		 * @param cov_cols has to be equal to cov_rows
		 */
		virtual inline void set_cov(float64_t* cov, int32_t cov_rows, int32_t cov_cols)
		{
			ASSERT(cov_rows = cov_cols);
			ASSERT(cov_rows = m_cov_rows);
			memcpy(m_cov, cov, sizeof(float64_t)*m_cov_rows*m_cov_cols);
			init();
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "Gaussian"; }

	private:
		/** Initialize parameters for serialization */
		void register_params();

	protected:
		/** constant part */
		float64_t m_constant;
		/** covariance */
		float64_t* m_cov;
		/** covariance row num */
		int32_t m_cov_rows;
		/** covariance col num */
		int32_t m_cov_cols;
		/** covariance inverse */
		float64_t* m_cov_inverse;
		/** covariance inverse row num */
		int32_t m_cov_inverse_rows;
		/** covariance inverse col num */
		int32_t m_cov_inverse_cols;
		/** mean */
		float64_t* m_mean;
		/** mean length */
		int32_t m_mean_length;
};
}
#endif //HAVE_LAPACK
#endif //_GAUSSIAN_H__
