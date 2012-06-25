/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include "MachineEvaluation.h"
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CMachineEvaluation::CMachineEvaluation()
{
	init();
}

CMachineEvaluation::CMachineEvaluation(CMachine* machine, CFeatures* features,
		CLabels* labels, CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock)
{
	init();

	m_machine=machine;
	m_features=features;
	m_labels=labels;
	m_splitting_strategy=splitting_strategy;
	m_evaluation_criterion=evaluation_criterion;
	m_autolock=autolock;

	SG_REF(m_machine);
	SG_REF(m_features);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterion);
}

CMachineEvaluation::CMachineEvaluation(CMachine* machine, CLabels* labels,
		CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock)
{
	init();

	m_machine=machine;
	m_labels=labels;
	m_splitting_strategy=splitting_strategy;
	m_evaluation_criterion=evaluation_criterion;
	m_autolock=autolock;

	SG_REF(m_machine);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterion);
}

CMachineEvaluation::~CMachineEvaluation() {
	// TODO Auto-generated destructor stub
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_evaluation_criterion);
}

void CMachineEvaluation::init()
{
	m_machine=NULL;
	m_features=NULL;
	m_labels=NULL;
	m_splitting_strategy=NULL;
	m_evaluation_criterion=NULL;
	m_do_unlock=false;
	m_autolock=true;

	m_parameters->add((CSGObject**) &m_machine, "machine",
			"Used learning machine");
	m_parameters->add((CSGObject**) &m_features, "features", "Used features");
	m_parameters->add((CSGObject**) &m_labels, "labels", "Used labels");
	m_parameters->add((CSGObject**) &m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy");
	m_parameters->add((CSGObject**) &m_evaluation_criterion,
			"evaluation_criterion", "Used evaluation criterion");
	m_parameters->add(&m_do_unlock, "do_unlock",
			"Whether machine should be unlocked after evaluation");
	m_parameters->add(&m_autolock, "m_autolock",
			"Whether machine should automatically try to be locked before "
			"evaluation");

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("m_do_unlock", CT_SCALAR, ST_NONE, PT_BOOL, 1),
			new SGParamInfo()
	);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("m_autolock", CT_SCALAR, ST_NONE, PT_BOOL, 1),
			new SGParamInfo()
	);
}

CMachine* CMachineEvaluation::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

EEvaluationDirection CMachineEvaluation::get_evaluation_direction()
{
	return m_evaluation_criterion->get_evaluation_direction();
}
