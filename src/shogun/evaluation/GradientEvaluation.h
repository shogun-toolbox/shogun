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
class GradientEvaluation: public MachineEvaluation
{
public:
	/** default constructor */
	GradientEvaluation();

	/** constructor
	 *
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	GradientEvaluation(std::shared_ptr<Machine> machine,
		std::shared_ptr<Features> features, std::shared_ptr<Labels> labels,
		std::shared_ptr<Evaluation> evaluation_criterion, bool autolock=true);

	virtual ~GradientEvaluation();

	/** returns the name of the machine evaluation
	 *
	 *  @return name GradientEvaluation
	 */
	virtual const char* get_name() const { return "GradientEvaluation"; }

	/** set differentiable function
	*
	* @param diff differentiable function
	*/
	inline void set_function(std::shared_ptr<DifferentiableFunction> diff)
	{


		m_diff=diff;
	}

	/** get differentiable function
	*
	* @return differentiable function
	*/
	inline std::shared_ptr<DifferentiableFunction> get_function()
	{

		return m_diff;
	}

private:
	/** initialses and registers parameters */
	void init();

	/** evaluates differentiable function for value and derivative.
	 *
	 * @return GradientResult containing value and gradient
	 */
	virtual std::shared_ptr<EvaluationResult> evaluate_impl();

	/** updates parameter dictionary of differentiable function */
	void update_parameter_dictionary();

private:
	/** differentiable function */
	std::shared_ptr<DifferentiableFunction> m_diff;

	/** parameter dictionary of differentiable function */
	std::shared_ptr<CMap<TParameter*, SGObject*>>  m_parameter_dictionary;
};
}
#endif /* CGRADIENTEVALUATION_H_ */
