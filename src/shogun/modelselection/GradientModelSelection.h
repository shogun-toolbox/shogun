/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Heiko Strathmann, Sergey Lisitsyn, Roman Votyakov,
 *          Soeren Sonnenburg, Wu Lin
 */


#ifndef CGRADIENTMODELSELECTION_H_
#define CGRADIENTMODELSELECTION_H_

#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderMinimizer.h>

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>

namespace shogun
{

class GradientModelSelectionCostFunction;

/** @brief Model selection class which searches for the best model by a
 * gradient-search.
 */
class GradientModelSelection : public ModelSelection
{
friend class GradientModelSelectionCostFunction;

public:
	/** default constructor */
	GradientModelSelection();

	/** constructor
	 *
	 * NOTE: if model_parameters is NULL, then gradient model selection is
	 * performed on all parameters of the machine.
	 *
	 * @param machine_eval machine evaluation object
	 * @param model_parameters parameters
	 */
	GradientModelSelection(std::shared_ptr<MachineEvaluation> machine_eval,
			std::shared_ptr<ModelSelectionParameters> model_parameters=NULL);

	virtual ~GradientModelSelection();

	/** method to select model via gradient search
	 *
	 * @param print_state if true, the output is verbose
	 *
	 * @return best combination of model parameters
	 */
	virtual std::shared_ptr<ParameterCombination> select_model(bool print_state=false);

	/** returns the name of the model selection object
	 *
	 *  @return name GradientModelSelection
	 */
	virtual const char* get_name() const { return "GradientModelSelection"; }

	/** set minimizer
	 *
	 * @param minimizer minimizer used in model selection
	 */
	virtual void set_minimizer(std::shared_ptr<FirstOrderMinimizer> minimizer);

private:
	/** initialize object */
	void init();

protected:

	/** minimizer */
	std::shared_ptr<FirstOrderMinimizer> m_mode_minimizer;

	/** get the cost given current model parameter
	 * Note that this function will compute the gradient
	 *
	 * @param model_vals model parameters
	 * @param model_grads grad of the model parameters
	 * @param func_data data needed for the callback function
	 */
	virtual float64_t get_cost(SGVector<float64_t> model_vars, SGVector<float64_t> model_grads, void* func_data);
};
}
#endif /* CGRADIENTMODELSELECTION_H_ */
