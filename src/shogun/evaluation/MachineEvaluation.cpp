/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Some code adapted from CrossValidation class by
 * Heiko Strathmann
 */

#include "MachineEvaluation.h"
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/base/Parameter.h>
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

	m_machine = machine;
	m_features = features;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_evaluation_criterion = evaluation_criterion;
	m_autolock = autolock;

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

	m_machine = machine;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_evaluation_criterion = evaluation_criterion;
	m_autolock = autolock;

	SG_REF(m_machine);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterion);
}

CMachineEvaluation::~CMachineEvaluation()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_evaluation_criterion);
}

void CMachineEvaluation::init()
{
	m_machine = NULL;
	m_features = NULL;
	m_labels = NULL;
	m_splitting_strategy = NULL;
	m_evaluation_criterion = NULL;
	m_do_unlock = false;
	m_autolock = true;

	SG_ADD((CSGObject**)&m_machine, "machine", "Used learning machine",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_features, "features", "Used features",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels, "labels", "Used labels",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_evaluation_criterion, "evaluation_criterion",
			"Used evaluation criterion", MS_NOT_AVAILABLE);
	SG_ADD(&m_do_unlock, "do_unlock",
			"Whether machine should be unlocked after evaluation",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_autolock, "m_autolock",
			"Whether machine should automatically try to be locked before ",
			MS_NOT_AVAILABLE);

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
