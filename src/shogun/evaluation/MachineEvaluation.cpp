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
#include <shogun/base/init.h>
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
	m_cancel_computation = false;
	m_pause_computation_flag = false;

	SG_ADD((CSGObject**)&m_machine, "machine", "Used learning machine");
	SG_ADD((CSGObject**)&m_features, "features", "Used features");
	SG_ADD((CSGObject**)&m_labels, "labels", "Used labels");
	SG_ADD((CSGObject**)&m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy");
	SG_ADD((CSGObject**)&m_evaluation_criterion, "evaluation_criterion",
			"Used evaluation criterion");
	SG_ADD(&m_do_unlock, "do_unlock",
			"Whether machine should be unlocked after evaluation");
	SG_ADD(&m_autolock, "m_autolock",
			"Whether machine should automatically try to be locked before ");

}

CEvaluationResult* CMachineEvaluation::evaluate()
{
	SG_DEBUG("entering %s::evaluate()\n", get_name())

	REQUIRE(
	    m_machine, "%s::evaluate() is only possible if a machine is "
	               "attached\n",
	    get_name());

	REQUIRE(
	    m_features, "%s::evaluate() is only possible if features are "
	                "attached\n",
	    get_name());

	REQUIRE(
	    m_labels, "%s::evaluate() is only possible if labels are "
	              "attached\n",
	    get_name());

	auto sub = connect_to_signal_handler();
	CEvaluationResult* result = evaluate_impl();
	sub.unsubscribe();
	reset_computation_variables();

	SG_DEBUG("leaving %s::evaluate()\n", get_name())
	return result;
};

CMachine* CMachineEvaluation::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

EEvaluationDirection CMachineEvaluation::get_evaluation_direction()
{
	return m_evaluation_criterion->get_evaluation_direction();
}
