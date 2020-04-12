/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Ahmed Khalifa
 */
/*
References:
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM
http://glm-tools.github.io/pyglmnet/api.html
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
	enum Family
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

		/** Default constructor */
		GLM();

		/** Constructor
		 *
		 * @param descend_updater chosen Descend Updater algorithm
		 * @param Linkfn the link function check
		 * https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
		 * @param Family the family check
		 * https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
		 * @param alpha alpha
		 * @param lambda lambda
		 */
		GLM(std::shared_ptr<DescendUpdater>,
		    Family family = POISSON_DISTRIBUTION, LinkFunction Link_fn = LOG,
		    float64_t alpha = 0.5, float64_t lambda = 0.1);

		/** standard constructor
		 * @param data features
		 * @param labs labels
		 * @param learn_rate Learning rate
		 * @param Linkfn the link function check
		 * https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
		 * @param Family the distribution/family check
		 * https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
		 * @param alpha alpha
		 * @param lambda lambda
		 */

		/** default destructor */
		virtual ~GLM()
		{
		}

		/** train model
		 *
		 * @param data training data
		 *
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
		std::shared_ptr<DescendUpdater> m_descend_updater;
		float64_t m_alpha;
		float64_t m_lambda;
		Family m_family;
		LinkFunction m_linkfn;

	private:
		void init();
	};
} // namespace shogun
#endif /* _GLM_H_ */