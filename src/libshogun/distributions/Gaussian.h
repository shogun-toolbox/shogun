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
		CGaussian(SGVector<float64_t> mean_vector, SGMatrix<float64_t> cov_matrix);

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
		 */
		virtual SGVector<float64_t> get_mean()
		{
			return SGVector<float64_t>(m_mean,m_mean_length);
		}

		/** set mean
		 *
		 * @param mean new mean
		 * @param mean_length has to match current mean length
		 */
		virtual void set_mean(SGVector<float64_t> mean_vector)
		{
			ASSERT(mean_vector.vlen == m_mean_length);
			memcpy(m_mean, mean_vector.vector, sizeof(float64_t)*m_mean_length);
		}

		/** get cov
		 *
		 */
		virtual SGMatrix<float64_t> get_cov()
		{
			return SGMatrix<float64_t>(m_cov,m_cov_rows,m_cov_cols);
		}

		/** set cov
		 *
		 */
		virtual inline void set_cov(SGMatrix<float64_t> cov_matrix)
		{
			ASSERT(cov_matrix.num_rows = cov_matrix.num_cols);
			ASSERT(cov_matrix.num_rows = m_cov_rows);
			memcpy(m_cov, cov_matrix.matrix, sizeof(float64_t)*m_cov_rows*m_cov_cols);
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
