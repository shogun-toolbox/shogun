/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Chiyuan Zhang, Fernando Iglesias,
 *          Soeren Sonnenburg, Heiko Strathmann, Jiaolong Xu, Evgeniy Andreev,
 *          Evan Shelhamer, Shell Hu, Thoralf Klein, Viktor Gal
 */

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/MulticlassMachine.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/labels/MultilabelLabels.h>

using namespace shogun;

MulticlassMachine::MulticlassMachine()
: BaseMulticlassMachine(), m_multiclass_strategy(std::make_shared<MulticlassOneVsRestStrategy>()),
	m_machine(NULL)
{

	register_parameters();
}

MulticlassMachine::MulticlassMachine(
		std::shared_ptr<MulticlassStrategy >strategy,
		std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labs)
: BaseMulticlassMachine(), m_multiclass_strategy(strategy)
{

	set_labels(labs);

	m_machine = machine;
	register_parameters();
}

MulticlassMachine::~MulticlassMachine()
{


}

void MulticlassMachine::set_labels(std::shared_ptr<Labels> lab)
{
    Machine::set_labels(lab);

}

void MulticlassMachine::register_parameters()
{
	SG_ADD(&m_multiclass_strategy,"multiclass_strategy", "Multiclass strategy");
	SG_ADD(&m_machine, "machine", "The base machine");
}

void MulticlassMachine::init_strategy()
{
    int32_t num_classes = m_labels->as<MulticlassLabels>()->get_num_classes();
    m_multiclass_strategy->set_num_classes(num_classes);
}

std::shared_ptr<BinaryLabels> MulticlassMachine::get_submachine_outputs(int32_t i)
{
	auto machine = m_machines.at(i);
	ASSERT(machine)
	return machine->apply_binary();
}

float64_t MulticlassMachine::get_submachine_output(int32_t i, int32_t num)
{
	auto machine = get_machine(i);
	float64_t output = 0.0;
	return machine->apply_one(num);
}

std::shared_ptr<MulticlassLabels> MulticlassMachine::apply_multiclass(std::shared_ptr<Features> data)
{
	SG_TRACE("entering {}::apply_multiclass({} at {})",
			get_name(), data ? data->get_name() : "NULL", fmt::ptr(data.get()));

	std::shared_ptr<MulticlassLabels> return_labels=NULL;

	if (data)
		init_machines_for_apply(data);
	else
		init_machines_for_apply(NULL);

	if (is_ready())
	{
		/* num vectors depends on whether data is provided */
		int32_t num_vectors=data ? data->get_num_vectors() :
				get_num_rhs_vectors();

		int32_t num_machines=m_machines.size();
		if (num_machines <= 0)
			error("num_machines = {}, did you train your machine?", num_machines);

		auto result=std::make_shared<MulticlassLabels>(num_vectors);

		// if outputs are prob, only one confidence for each class
		int32_t num_classes=m_multiclass_strategy->get_num_classes();
		EProbHeuristicType heuris = get_prob_heuris();

		if (heuris!=PROB_HEURIS_NONE)
			result->allocate_confidences_for(num_classes);
		else
			result->allocate_confidences_for(num_machines);

		std::vector<std::shared_ptr<BinaryLabels>> outputs(num_machines);
		SGVector<float64_t> As(num_machines);
		SGVector<float64_t> Bs(num_machines);

		for (int32_t i=0; i<num_machines; ++i)
		{
			outputs[i] = get_submachine_outputs(i);

			if (heuris==OVA_SOFTMAX)
			{
				Statistics::SigmoidParamters params = Statistics::fit_sigmoid(outputs[i]->get_values());
				As[i] = params.a;
				Bs[i] = params.b;
			}

			if (heuris!=PROB_HEURIS_NONE && heuris!=OVA_SOFTMAX)
				outputs[i]->scores_to_probabilities(0,0);
		}

		SGVector<float64_t> output_for_i(num_machines);
		SGVector<float64_t> r_output_for_i(num_machines);
		if (heuris!=PROB_HEURIS_NONE)
			r_output_for_i.resize_vector(num_classes);

		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				output_for_i[j] = outputs[j]->get_value(i);

			if (heuris==PROB_HEURIS_NONE)
			{
				r_output_for_i = output_for_i;
			}
			else
			{
				if (heuris==OVA_SOFTMAX)
					m_multiclass_strategy->rescale_outputs(output_for_i,As,Bs);
				else
					m_multiclass_strategy->rescale_outputs(output_for_i);

				// only first num_classes are returned
				for (int32_t r=0; r<num_classes; r++)
					r_output_for_i[r] = output_for_i[r];

				SG_DEBUG("{}::apply_multiclass(): sum(r_output_for_i) = {}",
					get_name(), SGVector<float64_t>::sum(r_output_for_i.vector,num_classes));
			}

			// use rescaled outputs for label decision
			result->set_label(i, m_multiclass_strategy->decide_label(r_output_for_i));
			result->set_multiclass_confidences(i, r_output_for_i);
		}
		outputs.clear();

		return_labels=result;
	}
	else
		error("Not ready");


	SG_TRACE("leaving {}::apply_multiclass({} at {})",
				get_name(), data ? data->get_name() : "NULL", fmt::ptr(data.get()));
	return return_labels;
}

std::shared_ptr<MultilabelLabels> MulticlassMachine::apply_multilabel_output(std::shared_ptr<Features> data, int32_t n_outputs)
{
	std::shared_ptr<MultilabelLabels> return_labels=NULL;

	if (data)
		init_machines_for_apply(data);
	else
		init_machines_for_apply(NULL);

	if (is_ready())
	{
		/* num vectors depends on whether data is provided */
		int32_t num_vectors=data ? data->get_num_vectors() :
				get_num_rhs_vectors();

		int32_t num_machines=m_machines.size();
		if (num_machines <= 0)
			error("num_machines = {}, did you train your machine?", num_machines);
		require(n_outputs<=num_machines,"You request more outputs than machines available");

		auto result=std::make_shared<MultilabelLabels>(num_vectors, n_outputs);
		std::vector<std::shared_ptr<BinaryLabels>> outputs(num_machines);

		for (int32_t i=0; i < num_machines; ++i)
			outputs[i] = get_submachine_outputs(i);

		SGVector<float64_t> output_for_i(num_machines);
		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				output_for_i[j] = outputs[j]->get_value(i);

			result->set_label(i, m_multiclass_strategy->decide_label_multiple_output(output_for_i, n_outputs));
		}
		for (int32_t i=0; i < num_machines; ++i)
			outputs[i].reset();

		return_labels=result;
	}
	else
		error("Not ready");

	return return_labels;
}

bool MulticlassMachine::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(m_multiclass_strategy)
	init_strategy();

	if ( !data && !is_ready() )
		error("Please provide training data.");
	else
		init_machine_for_train(data);

	m_machines.clear();
	auto train_labels = std::make_shared<BinaryLabels>(get_num_rhs_vectors());

	m_machine->set_labels(train_labels);

	m_multiclass_strategy->train_start(
	    multiclass_labels(m_labels), train_labels);
	while (m_multiclass_strategy->train_has_more())
	{
		SGVector<index_t> subset=m_multiclass_strategy->train_prepare_next();
		if (subset.vlen)
		{
			train_labels->add_subset(subset);
			add_machine_subset(subset);
		}

		m_machine->train();
		m_machines.push_back(get_machine_from_trained(m_machine));

		if (subset.vlen)
		{
			train_labels->remove_subset();
			remove_machine_subset();
		}
	}

	m_multiclass_strategy->train_stop();


	return true;
}

float64_t MulticlassMachine::apply_one(int32_t vec_idx)
{
	init_machines_for_apply(NULL);

	ASSERT(m_machines.size()>0)
	SGVector<float64_t> outputs(m_machines.size());

	for (int32_t i=0; i<m_machines.size(); i++)
		outputs[i] = get_submachine_output(i, vec_idx);

	float64_t result = m_multiclass_strategy->decide_label(outputs);

	return result;
}
