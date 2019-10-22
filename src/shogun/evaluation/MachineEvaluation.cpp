/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Heiko Strathmann, Giovanni De Toni
 */

#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>

#include <rxcpp/rx-lite.hpp>
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

std::shared_ptr<EvaluationResult> MachineEvaluation::evaluate() const
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

	auto result = evaluate_impl();

	SG_TRACE("leaving {}::evaluate()", get_name());
	return result;
};

std::shared_ptr<Machine> MachineEvaluation::get_machine() const
{

	return m_machine;
}

EEvaluationDirection MachineEvaluation::get_evaluation_direction() const
{
	return m_evaluation_criterion->get_evaluation_direction();
}
