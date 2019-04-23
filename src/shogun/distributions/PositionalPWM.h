/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Evan Shelhamer,
 *          Yuyu Zhang
 */

#ifndef _POSITIONAL_PWM_H__
#define _POSITIONAL_PWM_H__

#include <shogun/lib/config.h>

#include <shogun/distributions/Distribution.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
/** @brief Positional PWM */
class PositionalPWM : public Distribution
{
	public:
		/** default constructor */
		PositionalPWM();

		virtual ~PositionalPWM();

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

		/** get log likelihood window
		 * @param window
		 * @param len
		 * @param pos
		 */
		float64_t get_log_likelihood_window(uint8_t* window, int32_t len, float64_t pos);

		/** get sigma
		 */
		virtual float64_t get_sigma()
		{
			return m_sigma;
		}

		/** set sigma
		 *
		 * @param sigma new sigma
		 */
		virtual void set_sigma(float64_t sigma)
		{
			m_sigma=sigma;
		}

		/** get mean
		 */
		virtual float64_t get_mean()
		{
			return m_mean;
		}

		/** set mean
		 *
		 * @param mean new mean
		 */
		virtual void set_mean(float64_t mean)
		{
			m_mean=mean;
		}

		/** set pwm
		 *
		 * @param pwm new pwm (values must be in logspace)
		 */
		virtual void set_pwm(SGMatrix<float64_t> pwm)
		{
			m_pwm = pwm;
		}

		/** get pwm
		 *
		 * @return current pwm
		 */
		virtual SGMatrix<float64_t> get_pwm() const
		{
			return m_pwm;
		}

		/** get w
		 *
		 * @return current w
		 */
		virtual SGMatrix<float64_t> get_w() const
		{
			return m_w;
		}

		/** get poim u
		 *
		 * @param d degree for which poim should be obtained
		 *
		 * @return poim u
		 */
		virtual SGMatrix<float64_t> get_scoring(int32_t d);

		/** compute w
		 * @param num_pos
		 */
		void compute_w(int32_t num_pos);

		/** compute scoring
		 * @param max_degree
		 */
		void compute_scoring(int32_t max_degree);

		/** @return object name */
		virtual const char* get_name() const { return "PositionalPWM"; }

	private:
		/** Initialize parameters for serialization */
		void register_params();

	protected:

		/** pwm */
		SGMatrix<float64_t> m_pwm;

		/** sigma */
		float64_t m_sigma;

		/** mean */
		float64_t m_mean;

		/** w */
		SGMatrix<float64_t> m_w;

		/** poim */
		SGVector<float64_t> m_poim;

};
}
#endif //_POSITIONAL_PWM_H__
