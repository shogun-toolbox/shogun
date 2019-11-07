/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Alesis Novik, Heiko Strathmann, Evgeniy Andreev,
 *          Viktor Gal, Weijie Lin, Evan Shelhamer, Thoralf Klein
 */

#ifndef _GAUSSIAN_H__
#define _GAUSSIAN_H__

#include <shogun/lib/config.h>

#include <shogun/distributions/Distribution.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomMixin.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
class DotFeatures;

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
class Gaussian : public RandomMixin<Distribution>
{
	public:
		/** default constructor */
		Gaussian();
		/** constructor
		 *
		 * @param mean mean of the Gaussian
		 * @param cov covariance of the Gaussian
		 * @param cov_type covariance type (full, diagonal or shperical)
		 */
		Gaussian(const SGVector<float64_t> mean, SGMatrix<float64_t> cov, ECovType cov_type=FULL);
		virtual ~Gaussian();

		/** Compute the constant part */
		void init();

		/** learn distribution
		 *
		 * @param data training data
		 *
		 * @return whether training was successful
		 */
		virtual bool train(std::shared_ptr<Features> data=NULL);

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

		/** update parameters in the em maximization step for mixture model of which
		 * this distribution is a part
		 *
		 * @param alpha_k "belongingness" values of various data points
		 * @return sum of values in alpha_k
		 */
		virtual float64_t update_params_em(const SGVector<float64_t> alpha_k);

		/** compute PDF
		 *
		 * @param point point for which to compute the PDF
		 * @return computed PDF
		 */
		virtual float64_t compute_PDF(SGVector<float64_t> point)
		{
			return std::exp(compute_log_PDF(point));
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
		virtual SGVector<float64_t> get_mean();

		/** set mean
		 *
		 * @param mean new mean
		 */
		virtual void set_mean(const SGVector<float64_t> mean);

		/** get covariance
		 *
		 * @return cov covariance, memory needs to be freed by user
		 */
		virtual SGMatrix<float64_t> get_cov();

		/** set covariance
		 *
		 * Doesn't store the covariance, but decomposes, thus the covariance can be freed after exit without harming the object
		 *
		 * @param cov new covariance
		 */
		virtual void set_cov(SGMatrix<float64_t> cov);

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
		void set_d(const SGVector<float64_t> d);

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
			m_u = u;
		}

		/** sample from distribution
		 *
		 * @return sample
		 */
		SGVector<float64_t> sample();

		/** @param distribution is casted to Gaussian, NULL if not possible
		 * Note that the object is SG_REF'ed
		 * @return casted Gaussian object
		 */
#ifndef SWIG
		[[deprecated("use .as template function")]]
#endif
		static std::shared_ptr<Gaussian> obtain_from_generic(const std::shared_ptr<Distribution>& distribution);

		/** @return object name */
		virtual const char* get_name() const { return "Gaussian"; }

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
#endif //_GAUSSIAN_H__
