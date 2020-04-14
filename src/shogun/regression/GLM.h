/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Ahmed Khalifa
 */

#ifndef _GENERALIZEDLINEARMODEL_H__
#define _GENERALIZEDLINEARMODEL_H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/optimization/DescendUpdater.h>
#include <shogun/regression/Regression.h>

namespace shogun
{
	enum DistributionFamily
	{
		NORMAL_DISTRIBUTION,
		EXPONENTIAL_DISTRIBUTION,
		GAMMA_DISTRIBUTION,
		BINOMIAL_DISTRIBUTION,
		GAUSS_DISTRIBUTION,
		POISSON_DISTRIBUTION
	};
	enum LinkFunction
	{
		LOG,
		LOGIT,
		IDENTITY,
		INVERSE
	};
	/** @brief Class GLM implements Generalized Linear Models, such as poisson,
	 * gamma, binomial
	 */
	class GLM : public LinearMachine
	{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		GLM();
		SGVector<float64_t> log_likelihood(
		    const std::shared_ptr<DenseFeatures<float64_t>> features,
		    std::shared_ptr<Labels> label);
		/** Constructor
		 *
		 * @param descend_updater chosen Descend Updater algorithm
		 * @param link_fn the link function
		 * @param Family the family
		 * @param alpha Weighting parameter between L1 and L2 Penalty
		 * @param lambda Regularization parameter lambda
		 */
		GLM(const std::shared_ptr<DescendUpdater>& descend_updater,
		    DistributionFamily family, LinkFunction link_fn, float64_t alpha,
		    float64_t lambda);

		virtual ~GLM(){};

		/** train model
		 * @param data training data
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data = NULL)
		{
			return true;
		};

		/** @return object name */
		virtual const char* get_name() const
		{
			return "GLM";
		}

	protected:
		std::shared_ptr<DescendUpdater>
		    m_descend_updater; // TODO: Choose Default value
		float64_t m_alpha = 0.5;
		float64_t m_lambda = 0.1;
		DistributionFamily m_family = POISSON_DISTRIBUTION;
		LinkFunction m_link_fn = LOG;

	private:
		void init();
	};
} // namespace shogun
#endif /* _GLM_H_ */