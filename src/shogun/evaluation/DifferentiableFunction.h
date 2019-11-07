/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Yuyu Zhang
 */

#ifndef CDIFFERENTIABLEFUNCTION_H_
#define CDIFFERENTIABLEFUNCTION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief An abstract class that describes a differentiable function used for
 * GradientEvaluation.
 */
class DifferentiableFunction : public SGObject
{
public:
	/** default constructor */
	DifferentiableFunction(): SGObject() {}

	virtual ~DifferentiableFunction() {}

	/** get the gradient
	 *
	 * @param parameters parameter's dictionary
	 *
	 * @return map of gradient. Keys are names of parameters, values are values
	 * of derivative with respect to that parameter.
	 */

	virtual std::map<std::string, SGVector<float64_t>> get_gradient(
		std::map<Parameters::value_type, std::shared_ptr<SGObject>> parameters)=0; 	

	/** get the function value
	 *
	 * @return vector that represents the function value
	 */
	virtual SGVector<float64_t> get_value()=0;
};
}
#endif /* CDIFFERENTIABLEFUNCTION_H_ */
