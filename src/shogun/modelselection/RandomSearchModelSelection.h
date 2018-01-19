/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Roman Votyakov, 
 *          Heiko Strathmann, Yuyu Zhang
 */

#ifndef RANDOMSEARCHMODELSELECTION_H_
#define RANDOMSEARCHMODELSELECTION_H_

#include <shogun/lib/config.h>

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/base/DynArray.h>

namespace shogun
{
class CModelSelectionParameters;

/** @brief Model selection class which searches for the best model by a random
 * search. See CModelSelection for details.
 */
class CRandomSearchModelSelection : public CModelSelection
{
public:
	/** constructor */
	CRandomSearchModelSelection();

	/** constructor
	 *
	 * @param machine_eval machine to use
	 * @param model_parameters model parameters to use
	 * @param ratio ratio in range [0,1]
	 */
	CRandomSearchModelSelection(CMachineEvaluation* machine_eval,
			CModelSelectionParameters* model_parameters, float64_t ratio);

	/** destructor */
	virtual ~CRandomSearchModelSelection();

	/** returns ratio
	 *
	 * @return ratio
	 */
	float64_t get_ratio() const { return m_ratio; }

	/** sets ratio
	 *
	 * @param ratio ratio to set
	 */
	void set_ratio(float64_t ratio)
	{
		REQUIRE(ratio>0.0 && ratio<1.0, "Ratio should be in [0,1] range\n")
		m_ratio=ratio;
	}

	/** method to select model via random search model selection
	 *
	 * @param print_state if true, the current combination is printed
	 *
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const { return "RandomSearchModelSelection"; }

protected:
	/** ratio */
	float64_t m_ratio;
};
}
#endif /* RANDOMSEARCHMODELSELECTION_H_ */
