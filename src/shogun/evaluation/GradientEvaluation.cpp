/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/GradientResult.h>

using namespace shogun;

CGradientEvaluation::CGradientEvaluation() : CMachineEvaluation()
{
	init();
}

CGradientEvaluation::CGradientEvaluation(CMachine* machine, CFeatures* features,
		CLabels* labels, CEvaluation* evaluation_crit, bool autolock) :
		CMachineEvaluation(machine, features, labels, NULL, evaluation_crit, autolock)
{
	init();
}

void CGradientEvaluation::init()
{
	m_diff=NULL;
	m_parameter_dictionary=NULL;

	SG_ADD((CSGObject**)&m_diff, "differentiable_function",	"Differentiable "
			"function", MS_AVAILABLE);
}

CGradientEvaluation::~CGradientEvaluation()
{
	SG_UNREF(m_diff);
	SG_UNREF(m_parameter_dictionary);
}

void CGradientEvaluation::update_parameter_dictionary()
{
	SG_UNREF(m_parameter_dictionary);

	m_parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	m_diff->build_gradient_parameter_dictionary(m_parameter_dictionary);
	SG_REF(m_parameter_dictionary);
}

CEvaluationResult* CGradientEvaluation::evaluate()
{
	if (update_parameter_hash())
		update_parameter_dictionary();

	// create gradient result object
	CGradientResult* result=new CGradientResult();
	SG_REF(result);

	// set function value
	result->set_value(m_diff->get_value());

	CMap<TParameter*, SGVector<float64_t> >* gradient=m_diff->get_gradient(
			m_parameter_dictionary);

	// set gradient and parameter dictionary
	result->set_gradient(gradient);
	result->set_paramter_dictionary(m_parameter_dictionary);

	SG_UNREF(gradient);

	return result;
}
