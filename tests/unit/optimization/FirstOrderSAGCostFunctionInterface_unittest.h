/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk
 */

#include <shogun/lib/config.h>
#ifndef FIRSTORDERSAGCOSTFUNCTIONINTERFACE_UNITTEST_H
#define FIRSTORDERSAGCOSTFUNCTIONINTERFACE_UNITTEST_H
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
using namespace shogun;

class LeastSquareTestCostFunction : public FirstOrderSAGCostFunctionInterface
{
public:
	LeastSquareTestCostFunction(){};
	LeastSquareTestCostFunction(
	    SGMatrix<float64_t> X, SGMatrix<float64_t> y,
	    StanVector* trainable_parameters,
	    Eigen::Matrix<std::function<stan::math::var(int32_t)>, Eigen::Dynamic,
	                  1>* cost_for_ith_point,
	    std::function<stan::math::var(StanVector*)>* total_cost)
	    : FirstOrderSAGCostFunctionInterface(
	          X, y, trainable_parameters, cost_for_ith_point, total_cost){};

	virtual SGVector<float64_t> obtain_variable_reference();
	virtual const char* get_name() const
	{
		return "LeastSquareTestCostFunction";
	}
};

#endif /** FIRSTORDERSAGCOSTFUNCTIONINTERFACE_UNITTEST_H */
