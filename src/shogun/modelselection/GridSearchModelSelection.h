/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Soeren Sonnenburg, Jacob Walker,
 *          Yuyu Zhang, Roman Votyakov
 */

#ifndef __GRIDSEARCHMODELSELECTION_H_
#define __GRIDSEARCHMODELSELECTION_H_

#include <shogun/lib/config.h>

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/base/DynArray.h>

namespace shogun
{
class ModelSelectionParameters;

/** @brief Model selection class which searches for the best model by a grid-
 * search. See ModelSelection for details.
 */
class GridSearchModelSelection : public ModelSelection
{
public:
	/** constructor */
	GridSearchModelSelection();

	/** constructor
	 *
	 * @param machine_eval machine evaluation object
	 * @param model_parameters parameters
	 */
	GridSearchModelSelection(std::shared_ptr<MachineEvaluation> machine_eval,
			std::shared_ptr<ModelSelectionParameters> model_parameters);

	/** destructor */
	virtual ~GridSearchModelSelection();

	/** method to select model via grid search
	 *
	 * @param print_state if true, the current combination is printed
	 *
	 * @return best combination of model parameters
	 */
	virtual std::shared_ptr<ParameterCombination> select_model(bool print_state=false);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const { return "GridSearchModelSelection"; }
};
}
#endif /* __GRIDSEARCHMODELSELECTION_H_ */
