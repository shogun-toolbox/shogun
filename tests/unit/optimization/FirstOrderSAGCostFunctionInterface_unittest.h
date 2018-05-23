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
	virtual SGVector<float64_t> obtain_variable_reference();
	virtual const char* get_name() const
	{
		return "LeastSquareTestCostFunction";
	}
};

#endif /** FIRSTORDERSAGCOSTFUNCTIONINTERFACE_UNITTEST_H */
