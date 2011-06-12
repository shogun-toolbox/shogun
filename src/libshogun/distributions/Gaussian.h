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

enum ECovType
{
	FULL,
	DIAG,
	SPHERICAL
};

/** @brief Gaussian distribution interface.
 *
 * Takes as input a mean vector and covariance matrix.
 * Also possible to train from data.
 * Likelihood is computed using the Gaussian PDF \f$(2\pi)^{-\frac{k}{2}}|\Sigma|^{-\frac{1}{2}}e^{-\frac{1}{2}(x-\mu)'\Sigma^{-1}(x-\mu)}\f$
 * The actuall computations depend on the type of covariance used.
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
		 * @param cov_type covariance type
		 */
<<<<<<< HEAD
		CGaussian(SGVector<float64_t> mean_vector, SGMatrix<float64_t> cov_matrix);

=======
		CGaussian(float64_t* mean, int32_t mean_length,
					float64_t* cov, int32_t cov_rows, int32_t cov_cols,
					ECovType cov_type=FULL);
>>>>>>> Rewritten Gaussian class to work with different covariance types in log domain.
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
<<<<<<< HEAD
			return SGVector<float64_t>(m_mean,m_mean_length);
=======
			*mean=new float64_t[m_mean_length];
			memcpy(*mean, m_mean, sizeof(float64_t)*m_mean_length);
			*mean_length=m_mean_length;
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start
		}

		/** set mean
		 *
		 * @param mean new mean
		 * @param mean_length has to match current mean length
		 */
		virtual void set_mean(SGVector<float64_t> mean_vector)
		{
<<<<<<< HEAD
			ASSERT(mean_vector.vlen == m_mean_length);
			memcpy(m_mean, mean_vector.vector, sizeof(float64_t)*m_mean_length);
=======
			if (m_mean_length>0)
			{
				ASSERT(mean_length==m_mean_length);
			}
			else
			{
				m_mean_length=mean_length;
				m_mean=new float64_t[mean_length];
			}
			memcpy(m_mean, mean, sizeof(float64_t)*m_mean_length);
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start
		}

		/** get cov
		 *
		 */
<<<<<<< HEAD
		virtual SGMatrix<float64_t> get_cov()
		{
			return SGMatrix<float64_t>(m_cov,m_cov_rows,m_cov_cols);
		}
=======
		virtual void get_cov(float64_t** cov, int32_t* cov_rows, int32_t* cov_cols);
>>>>>>> Rewritten Gaussian class to work with different covariance types in log domain.

		/** set cov
		 *
		 */
		virtual inline void set_cov(SGMatrix<float64_t> cov_matrix)
		{
<<<<<<< HEAD
<<<<<<< HEAD
			ASSERT(cov_matrix.num_rows = cov_matrix.num_cols);
			ASSERT(cov_matrix.num_rows = m_cov_rows);
			memcpy(m_cov, cov_matrix.matrix, sizeof(float64_t)*m_cov_rows*m_cov_cols);
=======
			ASSERT(cov_rows = cov_cols);
			ASSERT(cov_rows = m_mean_length);
=======
			ASSERT(cov_rows==cov_cols);
			ASSERT(cov_rows==m_mean_length);
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start
			decompose_cov(cov, cov_rows);
>>>>>>> Rewritten Gaussian class to work with different covariance types in log domain.
			init();
		}

		/** get cov type
		 *
		 * @return covariance type
		 */
		inline ECovType get_cov_type()
		{
			return m_cov_type;
		}

		/** set cov type
		 *
		 * Will only take effect after covariance is changed
		 *
		 * @param cov_type new covariance type
		 */
		inline void set_cov_type(ECovType cov_type)
		{
			m_cov_type = cov_type;
		}

		/** set diagonal
		 *
		 * @param d diagonal
		 * @param d_length diagonal length
		 */
		inline void set_d(float64_t* d, int32_t d_length)
		{
			if (m_d_length>0)
			{
				ASSERT(d_length==m_d_length);
			}
			else
			{
				m_d_length=d_length;
				m_d=new float64_t[d_length];
			}
			memcpy(m_d, d, sizeof(float64_t)*m_d_length);			
		}

		/** set unitary matrix
		 *
		 * @param d diagonal
		 * @param d_length diagonal length
		 */
		inline void set_u(float64_t* u, int32_t u_rows, int32_t u_cols)
		{
			if (m_u_rows>0)
			{
				ASSERT(u_rows==u_cols);
				ASSERT(u_rows==m_u_rows);
			}
			else
			{
				m_u_rows=u_rows;
				m_u_cols=u_cols;
				m_u=new float64_t[m_u_rows*m_u_cols];
			}
			memcpy(m_u, u, sizeof(float64_t)*m_u_rows*m_u_cols);			
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "Gaussian"; }

	private:
		/** Initialize parameters for serialization */
		void register_params();

		/** decompose covariance matrix according to type */
		void decompose_cov(float64_t* cov, int32_t cov_size);

	protected:
		/** constant part */
		float64_t m_constant;
		/** diagonal */
		float64_t* m_d;
		/** diagonal length */
		int32_t m_d_length;
		/** unitary matrix */
		float64_t* m_u;
		/** unitary matrix row num */
		int32_t  m_u_rows;
		/** unitary matrix col num */
		int32_t m_u_cols;
		/** mean */
		float64_t* m_mean;
		/** mean length */
		int32_t m_mean_length;
		/** covariance type */
		ECovType m_cov_type;
};
}
#endif //HAVE_LAPACK
#endif //_GAUSSIAN_H__
