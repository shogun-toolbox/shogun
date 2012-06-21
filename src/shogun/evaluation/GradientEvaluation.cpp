/*
 * GradientEvaluation.cpp
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#include "GradientEvaluation.h"
#include <shogun/evaluation/GradientResult.h>

namespace shogun {

CGradientEvaluation::CGradientEvaluation() {
	// TODO Auto-generated constructor stub

}

CGradientEvaluation::~CGradientEvaluation() {
	// TODO Auto-generated destructor stub
}

CEvaluationResult* CGradientEvaluation::evaluate()
{
	CGradientResult* result = new CGradientResult();

	SGVector<float64_t> quan = diff->get_quantity();
	CMap<shogun::SGString<char>, double> grad = diff->get_gradient();

	result->quantity = quan.clone();
	result->gradient = grad;

	SG_REF(result);

	return result;
}

} /* namespace shogun */
