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

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/distributions/Distribution.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
class CDotFeatures;

/** Covariance type */
enum ECovType
{
	/// full covariance
	FULL,
	/// diagonal covariance
	DIAG,
	/// spherical covariance
	SPHERICAL
};

/** @brief Gaussian distribution interface.
 *
 * Takes as input a mean vector and covariance matrix.
 * Also possible to train from data.
 * Likelihood is computed using the Gaussian PDF \f$(2\pi)^{-\frac{k}{2}}|\Sigma|^{-\frac{1}{2}}e^{-\frac{1}{2}(x-\mu)'\Sigma^{-1}(x-\mu)}\f$
 * The actual computations depend on the type of covariance used.
 */
class CGaussian : public CDistribution
{
	public:
		/** default constructor */
		CGaussian();
		/** constructor
		 *
		 * @param mean mean of the Gaussian
		 * @param cov covariance of the Gaussian
		 * @param cov_type covariance type (full, diagonal or shperical)
		 */
		CGaussian(SGVector<float64_t> mean, SGMatrix<float64_t> cov, ECovType cov_type=FULL);
		virtual ~CGaussian();

		/** Compute the constant part */
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
		 * @param point point for which to compute the PDF
		 * @return computed PDF
		 */
		virtual inline float64_t compute_PDF(SGVector<float64_t> point)
		{
			return CMath::exp(compute_log_PDF(point));
		}

		/** compute log PDF
		 *
		 * @param point point for which to compute the log PDF
		 * @return computed log PDF
		 */
		virtual float64_t compute_log_PDF(SGVector<float64_t> point);

		/** get mean
		 *
		 * @return mean
		 */
		virtual inline SGVector<float64_t> get_mean()
		{
			return m_mean;
		}

		/** set mean
		 *
		 * @param mean new mean
		 */
		virtual inline void set_mean(SGVector<float64_t> mean)
		{
			m_mean.destroy_vector();
			if (mean.vlen==1)
				m_cov_type=SPHERICAL;

			m_mean=mean;
		}

		/** get covariance
		 *
		 * @param cov covariance, memory needs to be freed by user
		 */
		virtual SGMatrix<float64_t> get_cov();

		/** set covariance
		 *
		 * Doesn't store the covariance, but decomposes, thus the covariance can be freed after exit without harming the object
		 *
		 * @param cov new covariance
		 */
		virtual inline void set_cov(SGMatrix<float64_t> cov)
		{
			ASSERT(cov.num_rows==cov.num_cols);
			ASSERT(cov.num_rows==m_mean.vlen);
			decompose_cov(cov);
			init();
			if (cov.do_free)
				cov.free_matrix();
		}

		/** get covariance type
		 *
		 * @return covariance type
		 */
		inline ECovType get_cov_type()
		{
			return m_cov_type;
		}

		/** set covariance type
		 *
		 * Will only take effect after covariance is changed
		 *
		 * @param cov_type new covariance type
		 */
		inline void set_cov_type(ECovType cov_type)
		{
			m_cov_type = cov_type;
		}

		/** get diagonal
		 *
		 * @return diagonal
		*/
		inline SGVector<float64_t> get_d()
		{
			return m_d;
		}

		/** set diagonal
		 *
		 * @param d new diagonal
		 */
		inline void set_d(SGVector<float64_t> d)
		{
			m_d.destroy_vector();
			m_d = d;
			init();
		}

		/** get unitary matrix
		 *
		 * @return unitary matrix
		*/
		inline SGMatrix<float64_t> get_u()
		{
			return m_u;
		}

		/** set unitary matrix
		 *
		 * @param u new unitary matrix
		 */
		inline void set_u(SGMatrix<float64_t> u)
		{
			m_u.destroy_matrix();
			m_u = u;
		}

		/** sample from distribution
		 *
		 * @return sample
		 */
		SGVector<float64_t> sample();

		/** @return object name */
		inline virtual const char* get_name() const { return "Gaussian"; }

	private:
		/** Initialize parameters for serialization */
		void register_params();

		/** decompose covariance matrix according to type
		 *
		 * @param cov covariance
		 */
		void decompose_cov(SGMatrix<float64_t> cov);

	protected:
		/** constant part */
		float64_t m_constant;
		/** diagonal */
		SGVector<float64_t> m_d;
		/** unitary matrix */
		SGMatrix<float64_t> m_u;
		/** mean */
		SGVector<float64_t> m_mean;
		/** covariance type */
		ECovType m_cov_type;
};
}
#endif //HAVE_LAPACK
#endif //_GAUSSIAN_H__
