/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Jacob Walker, Sergey Lisitsyn, Roman Votyakov,
 *          Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef __MODELSELECTION_H_
#define __MODELSELECTION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/evaluation/MachineEvaluation.h>

namespace shogun
{
class ModelSelectionParameters;
class ParameterCombination;

/** @brief Abstract base class for model selection.
 *
 * Takes a parameter tree which specifies parameters for model selection, and a
 * cross-validation instance and searches for the best combination of parameters
 * in the abstract method select_model(), which has to be implemented in
 * concrete sub-classes.
 */
class ModelSelection: public SGObject
{
public:
	/** default constructor */
	ModelSelection();

	/** constructor
	 *
	 * @param machine_eval object that computes the actual evaluation
	 * @param model_parameters parameter tree with model parameters to optimize
	 */
	ModelSelection(std::shared_ptr<MachineEvaluation> machine_eval,
			std::shared_ptr<ModelSelectionParameters> model_parameters);

	/** destructor */
	virtual ~ModelSelection();

	/** abstract method to select model
	 *
	 * @param print_state if true, the current combination is printed
	 *
	 * @return best combination of model parameters
	 */
	virtual std::shared_ptr<ParameterCombination> select_model(bool print_state=false)=0;

private:
	/** initializer */
	void init();

protected:
	/** model parameters */
	std::shared_ptr<ModelSelectionParameters> m_model_parameters;
	/** cross validation */
	std::shared_ptr<MachineEvaluation> m_machine_eval;
};
}
#endif /* __MODELSELECTION_H_ */
