/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <shogun/lib/config.h>
#ifndef StanFirstOrderSAGCostFunction_UNITTEST_H
#define StanFirstOrderSAGCostFunction_UNITTEST_H
#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
using namespace shogun;

class LeastSquareTestCostFunction : public StanFirstOrderSAGCostFunction
{
public:
	LeastSquareTestCostFunction(
		SGMatrix<float64_t> X, SGMatrix<float64_t> y,
		StanVector& trainable_parameters,
		StanFunctionsVector<float64_t> cost_for_ith_point,
		FunctionReturnsStan<const StanVector& > total_cost)
	    : StanFirstOrderSAGCostFunction(
	          X, y, trainable_parameters, cost_for_ith_point, total_cost){};

	virtual const char* get_name() const
	{
		return "LeastSquareTestCostFunction";
	}
};

#endif /** StanFirstOrderSAGCostFunction_UNITTEST_H */
