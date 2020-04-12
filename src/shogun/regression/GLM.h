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
#include <shogun/optimization/DescendUpdaterWithCorrection.h>
#include <shogun/regression/Regression.h>

namespace shogun
{
enum Family
{
	normal,
	exponential,
	gamma,
	binomial,
	gaussian,
	poisson
};
enum LinkFunction
{
	log,
	logit,
	identity,
	inverse
};
/** @brief Class GLM implements Generalized Linear Models, such as poisson, gamma, binomial
 */
class GLM : public LinearMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);


		/** Default constructor */
		GLM();

		/** default constructor
	 	*
	 	* @param descend_updater chosen Descend Updater algorithm
	 	* @param Linkfn the link function check https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
	 	* @param Family the family check https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
	 	* @param alpha alpha
	 	* @param lambda lambda
	 	*/
		GLM(
			DescendUpdaterWithCorrection* descend_updater, Family family= poisson, LinkFunction Link_fn= log, 
			float64_t alpha= 0.5, float64_t lambda= 0.1);

		/** standard constructor
		* @param data features
		* @param labs labels
		* @param learn_rate Learning rate
	 	* @param Linkfn the link function check https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
	 	* @param Family the distribution/family check https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family
	 	* @param alpha alpha
	 	* @param lambda lambda
		*/
		
		/** default destructor */
		virtual ~GLM() {}

		/** train model
		 *
		 * @param data training data 
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(Features* data=NULL) {return true;};

		/** @return object name */
		virtual const char* get_name() const { return "GLM"; }



	protected:

		DescendUpdaterWithCorrection* m_descend_updater;
		float64_t m_alpha;
		float64_t m_lambda;
		Family m_family;
		LinkFunction m_linkfn;

	private:
		void init();

};
}
#endif /* _GLM_H_ */
