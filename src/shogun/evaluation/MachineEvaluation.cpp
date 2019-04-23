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

MachineEvaluation::MachineEvaluation()
{
	init();
}

MachineEvaluation::MachineEvaluation(std::shared_ptr<Machine> machine, std::shared_ptr<Features> features,
		std::shared_ptr<Labels> labels, std::shared_ptr<SplittingStrategy> splitting_strategy,
		std::shared_ptr<Evaluation> evaluation_criterion, bool autolock)
{
	init();

	m_machine = machine;
	m_features = features;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_evaluation_criterion = evaluation_criterion;
	m_autolock = autolock;






}

MachineEvaluation::MachineEvaluation(std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels,
		std::shared_ptr<SplittingStrategy> splitting_strategy,
		std::shared_ptr<Evaluation> evaluation_criterion, bool autolock)
{
	init();

	m_machine = machine;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_evaluation_criterion = evaluation_criterion;
	m_autolock = autolock;





}

MachineEvaluation::~MachineEvaluation()
{





}

void MachineEvaluation::init()
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

	SG_ADD(&m_machine, "machine", "Used learning machine");
	SG_ADD(&m_features, "features", "Used features");
	SG_ADD(&m_labels, "labels", "Used labels");
	SG_ADD(&m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy");
	SG_ADD(&m_evaluation_criterion, "evaluation_criterion",
			"Used evaluation criterion");
	SG_ADD(&m_do_unlock, "do_unlock",
			"Whether machine should be unlocked after evaluation");
	SG_ADD(&m_autolock, "autolock",
			"Whether machine should automatically try to be locked before ");

}

std::shared_ptr<EvaluationResult> MachineEvaluation::evaluate()
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
	auto result = evaluate_impl();
	sub.unsubscribe();
	reset_computation_variables();

	SG_DEBUG("leaving %s::evaluate()\n", get_name())
	return result;
};

std::shared_ptr<Machine> MachineEvaluation::get_machine() const
{

	return m_machine;
}

EEvaluationDirection MachineEvaluation::get_evaluation_direction()
{
	return m_evaluation_criterion->get_evaluation_direction();
}
