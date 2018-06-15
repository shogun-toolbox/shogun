/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Wu Lin
 */

#include <shogun/lib/config.h>
#ifndef STANSTOCHASTICMINIMIZERS_UNITTEST_H
#define STANSTOCHASTICMINIMIZERS_UNITTEST_H
#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
using namespace shogun;

class SquareErrorTestCostFunction : public StanFirstOrderSAGCostFunction
{
public:
	SquareErrorTestCostFunction(
		SGMatrix<float64_t> X, SGMatrix<float64_t> y,
		StanVector& trainable_parameters,
		StanFunctionsVector<float64_t> cost_for_ith_point,
		FunctionReturnsStan<const StanVector& > total_cost)
	    : StanFirstOrderSAGCostFunction(
	          X, y, trainable_parameters, cost_for_ith_point, total_cost){};

	virtual const char* get_name() const
	{
		return "SquareErrorTestCostFunction";
	}
};

#endif /** STANSTOCHASTICMINIMIZERS_UNITTEST_H */
