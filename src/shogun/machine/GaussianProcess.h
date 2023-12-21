/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin, Roman Votyakov
 *
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 */

#ifndef _GAUSSIANPROCESSMACHINE_H_
#define _GAUSSIANPROCESSMACHINE_H_

#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/gp/Inference.h>
#include <shogun/mathematics/Seedable.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/machine/NonParametricMachine.h>

namespace shogun
{

	/** @brief A base class for Gaussian Processes.
	 *
	 * Instead of a distribution over weights, the GP specifies a distribution
	 * over functions:
	 *
	 * \f[
	 * f(x) \sim \mathcal{GP} (m(x), k(x,x'))
	 * \f]
	 *
	 * where \f$m(x)\f$ - mean function, \f$k(x, x')\f$ - covariance function.
	 */
	class GaussianProcess : public RandomMixin<NonParametricMachine>
	{
	public:
		/** default constructor */
		GaussianProcess();

		/** constructor
		 *
		 * @param method inference method
		 */
		GaussianProcess(std::shared_ptr<Inference> method);

		~GaussianProcess() override;

		/** returns name of the machine
		 *
		 * @return name GaussianProcess
		 */
		const char* get_name() const override
		{
			return "GaussianProcess";
		}

		/** returns a mean \f$\mu\f$ of a Gaussian distribution
		 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
		 * posterior marginal \f$p(f_*|X,y,x_*)\f$.
		 *
		 * @param data testing features
		 *
		 * @return posterior means
		 */
		SGVector<float64_t>
		get_posterior_means(const std::shared_ptr<Features>& data);

		/** returns a variance \f$\sigma^2\f$ of a Gaussian distribution
		 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
		 * posterior marginal \f$p(f_*|X,y,x_*)\f$.
		 *
		 * @param data testing features
		 *
		 * @return posterior variances
		 */
		SGVector<float64_t>
		get_posterior_variances(const std::shared_ptr<Features>& data);

		/** get inference method
		 *
		 * @return inference method, which is used by Gaussian process machine
		 */
		std::shared_ptr<Inference> get_inference_method() const
		{

			return m_method;
		}

		/** set inference method
		 *
		 * @param method inference method
		 */
		void set_inference_method(std::shared_ptr<Inference> method)
		{

			m_method = method;
		}

		/** set training labels
		 *
		 * @param lab labels to set
		 */
		void set_labels(std::shared_ptr<Labels> lab) override
		{
			NonParametricMachine::set_labels(lab);
			m_method->set_labels(lab);
		}

		virtual SGVector<float64_t>
		get_mean_vector(const std::shared_ptr<Features>& data)
		{
			not_implemented(SOURCE_LOCATION);
		}

		virtual SGVector<float64_t>
		get_variance_vector(const std::shared_ptr<Features>& data)
		{
			not_implemented(SOURCE_LOCATION);
		}

		virtual SGVector<float64_t>
		get_probabilities(const std::shared_ptr<Features>& data)
		{
			not_implemented(SOURCE_LOCATION);
		}

		bool train_require_labels() const override
		{
			return false;
		}

	private:
		void init();

	protected:
		/** inference method */
		std::shared_ptr<Inference> m_method;
		/** Whether predictive variance is computed in predictions. If true, the
		 * values are stored in the current_values vector of the predicted
		 * labels
		 */
		bool m_compute_variance;
		/**use in inference method*/
		std::shared_ptr<Features> m_inducing_features;
	};
} // namespace shogun
#endif /* _GAUSSIANPROCESSMACHINE_H_ */
