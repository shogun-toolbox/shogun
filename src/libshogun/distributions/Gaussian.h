#ifndef _GAUSSIAN_H__
#define _GAUSSIAN_H__

#include "distributions/Distribution.h"
#include "lib/lapack.h"
#include "lib/Mathematics.h"

namespace shogun
{
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
		 * @return model parameter (logarithmic)
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

		/** @return object name */
		inline virtual const char* get_name() const { return "Gaussian"; }
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
};
}
#endif
