/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Heiko Strathmann, Giovanni De Toni
 */

#include <shogun/base/Parameter.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>

#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>

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
	m_cancel_computation = false;
	m_pause_computation_flag = false;

	SG_ADD(&m_machine, "machine", "Used learning machine");
	SG_ADD(&m_features, "features", "Used features");
	SG_ADD(&m_labels, "labels", "Used labels");
	SG_ADD(&m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy");
	SG_ADD(&m_evaluation_criterion, "evaluation_criterion",
			"Used evaluation criterion");
}

CEvaluationResult* CMachineEvaluation::evaluate() const
{
	SG_TRACE("entering {}::evaluate()", get_name());

	require(
	    m_machine, "{}::evaluate() is only possible if a machine is "
	               "attached",
	    get_name());

	require(
	    m_features, "{}::evaluate() is only possible if features are "
	                "attached",
	    get_name());

	require(
	    m_labels, "{}::evaluate() is only possible if labels are "
	              "attached",
	    get_name());

	CEvaluationResult* result = evaluate_impl();

	SG_TRACE("leaving {}::evaluate()", get_name());
	return result;
};

CMachine* CMachineEvaluation::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

EEvaluationDirection CMachineEvaluation::get_evaluation_direction() const
{
	return m_evaluation_criterion->get_evaluation_direction();
}
