/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/GradientResult.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/EvaluationResult.h>


using namespace shogun;

CGradientEvaluation::CGradientEvaluation() : CMachineEvaluation(NULL,
		NULL, NULL, NULL, NULL, true)
{
	init();
}

CGradientEvaluation::CGradientEvaluation(CMachine* machine, CFeatures* features,
		CLabels* labels, CEvaluation* evaluation_crit, bool autolock) :
		CMachineEvaluation(machine, features, labels, NULL, evaluation_crit, true)
{
	init();
}

void CGradientEvaluation::init()
{
	m_diff = NULL;

	SG_ADD((CSGObject**)&m_diff, "differentiable_function",
			"Differentiable Function", MS_NOT_AVAILABLE);
}

CGradientEvaluation::~CGradientEvaluation()
{
	SG_UNREF(m_diff);
}

CEvaluationResult* CGradientEvaluation::evaluate()
{
	CGradientResult* result = new CGradientResult();

	SGVector<float64_t> quan = m_diff->get_quantity();

	result->gradient = m_diff->get_gradient(result->parameter_dictionary);

	result->quantity = quan.clone();

	result->total_variables = 0;

	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		result->total_variables += node->data.vlen;
	}


	SG_REF(result);
	return result;
}

