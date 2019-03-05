/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Heiko Strathmann, Sergey Lisitsyn, Roman Votyakov, 
 *          Yuyu Zhang, Giovanni De Toni
 */

#ifndef CGRADIENTEVALUATION_H_
#define CGRADIENTEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/evaluation/EvaluationResult.h>

namespace shogun
{

/** @brief Class evaluates a machine using its associated differentiable
 * function for the function value and its gradient with respect to parameters.
 */
class CGradientEvaluation: public CMachineEvaluation
{
public:
	/** default constructor */
	CGradientEvaluation();

	/** constructor
	 *
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CGradientEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CEvaluation* evaluation_criterion, bool autolock=true);

	virtual ~CGradientEvaluation();

	/** returns the name of the machine evaluation
	 *
	 *  @return name GradientEvaluation
	 */
	virtual const char* get_name() const { return "GradientEvaluation"; }

	/** set differentiable function
	*
	* @param diff differentiable function
	*/
	inline void set_function(CDifferentiableFunction* diff)
	{
		SG_REF(diff);
		SG_UNREF(m_diff);
		m_diff=diff;
	}

	/** get differentiable function
	*
	* @return differentiable function
	*/
	inline CDifferentiableFunction* get_function()
	{
		SG_REF(m_diff);
		return m_diff;
	}

private:
	/** initialises and registers parameters */
	void init();

	/** evaluates differentiable function for value and derivative.
	 *
	 * @return GradientResult containing value and gradient
	 */
	virtual CEvaluationResult* evaluate_impl();

	/** updates parameter dictionary of differentiable function */
	void update_parameter_dictionary();

private:
	/** differentiable function */
	CDifferentiableFunction* m_diff;

	/** parameter dictionary of differentiable function */
	CMap<AnyParameter*, CSGObject*>*  m_parameter_dictionary;
};
}
#endif /* CGRADIENTEVALUATION_H_ */
