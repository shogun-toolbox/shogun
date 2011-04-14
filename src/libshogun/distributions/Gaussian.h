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

#include "distributions/Distribution.h"
#include "features/DotFeatures.h"
#include "lib/common.h"
#include "lib/lapack.h"
#include "lib/Mathematics.h"

namespace shogun
{
class CDotFeatures;
/** @brief gaussian distribution interface.
 *
 * A takes as input a mean vector and covariance matrix.
 * Also possible to train from data.
 */
class CGaussian : public CDistribution
{
	public:
		/** default constructor */
		CGaussian();
		/** constructor
		 *
		 * @param mean and covariance
		 */
		CGaussian(float64_t* mean, float64_t* cov, int32_t dim);
		virtual ~CGaussian();

		/** Set the distribution mean and covariance */
		void init(float64_t* mean, float64_t* cov, int32_t dim);

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
		 * @return computed PDF
		 */
		virtual float64_t compute_PDF(float64_t* point, int32_t point_len);

		/** set data vectors
		 *
		 * @param f new data vectors
		 */
		virtual inline void set_data(CDotFeatures* f)
		{
			SG_UNREF(m_data);
			SG_REF(f);
			m_data=f;
		}

		/** get data vectors
		 *
		 * @return data vectors
		 */
		virtual inline CDotFeatures* get_data()
		{
			SG_REF(m_data);
			return m_data;
		}

		/** get mean
		 *
		 * @param copy of the mean
		 */
		virtual inline void get_mean(float64_t** mean)
		{
			*mean = new float64_t[m_dim];
			memcpy(*mean, m_mean, sizeof(float64_t)*m_dim);
		}

		/** set mean
		 *
		 * @param new mean
		 */
		virtual inline void set_mean(float64_t* mean)
		{
			memcpy(m_mean, mean, sizeof(float64_t)*m_dim);
		}

		/** get cov
		 *
		 * @param copy of the cov
		 */
		virtual inline void get_cov(float64_t** cov)
		{
			*cov = new float64_t[m_dim*m_dim];
			memcpy(*cov, m_cov, sizeof(float64_t)*m_dim*m_dim);
		}

		/** set cov
		 *
		 * @param new cov
		 */
		virtual inline void set_cov(float64_t* cov)
		{			
			memcpy(m_cov, cov, sizeof(float64_t)*m_dim*m_dim);
			memcpy(m_cov_inverse, cov, sizeof(float64_t)*m_dim*m_dim);
			int32_t result = clapack_dpotrf(CblasRowMajor, CblasLower, m_dim, m_cov_inverse, m_dim);
			m_constant = 1;

			for (int i = 0; i < m_dim; i++)
				m_constant *= m_cov_inverse[i*m_dim+i];

			m_constant = 1/m_constant;
			m_constant *= pow(2*M_PI, (float64_t) -m_dim/2);

			result = clapack_dpotri(CblasRowMajor, CblasLower, m_dim, m_cov_inverse, m_dim);
		}

		/** get dim
		 *
		 * @return dimension
		 */
		virtual inline int32_t get_dim()
		{
			return m_dim;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "Gaussian"; }
	private:
		/** Initialize parameters for serialization */
		void init();
	protected:
		/** constant part */
		float64_t m_constant;
		/** covariance */
		float64_t* m_cov;
		/** covariance inverse */
		float64_t* m_cov_inverse;
		/** mean */
		float64_t* m_mean;
		/** dimensionality */
		int32_t m_dim;
		/** data features */
		CDotFeatures* m_data;
};
}
#endif
