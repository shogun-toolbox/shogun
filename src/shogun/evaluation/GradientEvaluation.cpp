/*
 * GradientEvaluation.cpp
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#include "GradientEvaluation.h"
#include <shogun/evaluation/GradientResult.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/EvaluationResult.h>


namespace shogun {

CGradientEvaluation::CGradientEvaluation() {
	// TODO Auto-generated constructor stub

}

CGradientEvaluation::CGradientEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
		CSplittingStrategy* splitting_strategy, CEvaluation* evaluation_crit, bool autolock) :
		CMachineEvaluation(machine, features, labels, NULL, NULL, true)
{


}

CGradientEvaluation::~CGradientEvaluation() {
	// TODO Auto-generated destructor stub
}

CEvaluationResult* CGradientEvaluation::evaluate()
{
	CGradientResult* result = new CGradientResult();

	SGVector<float64_t> quan = diff->get_quantity();
	result->gradient = diff->get_gradient();

	result->quantity = quan.clone();
	//result->gradient = grad.c;

	//CEvaluationResult* stupid = result;
	//result = (CGradientResult*)stupid;
/*	for(int i = 0; i < 2; i++)
	{
		CMapNode<SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);

		//char* name = node->key.;
		SG_SPRINT("%s\n", node->key.string);
		SG_SPRINT("%i\n", node->key.slen);
		SG_SPRINT("%f\n", node->data);
	}*/

	SG_REF(result);
	return result;
}

} /* namespace shogun */
