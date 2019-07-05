/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Saurabh Mahindre, Heiko Strathmann, Thoralf Klein,
 *          Olivier NGuyen, Bjoern Esser, Weijie Lin
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/base/progress.h>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/evaluation/Evaluation.h>

using namespace shogun;

BaggingMachine::BaggingMachine() : RandomMixin<Machine>()
{
	init();
	register_parameters();
}

BaggingMachine::BaggingMachine(std::shared_ptr<Features> features, std::shared_ptr<Labels> labels)
    : BaggingMachine()
{
	set_labels(labels);
	m_features = features;
}

std::shared_ptr<BinaryLabels> BaggingMachine::apply_binary(std::shared_ptr<Features> data)
{
	SGMatrix<float64_t> output = apply_outputs_without_combination(data);

	auto mean_rule = std::make_shared<MeanRule>();

	SGVector<float64_t> labels = m_combination_rule->combine(output);
	SGVector<float64_t> probabilities = mean_rule->combine(output);

	float64_t threshold = 0.5;
	return std::make_shared<BinaryLabels>(probabilities, threshold);
}

std::shared_ptr<MulticlassLabels> BaggingMachine::apply_multiclass(std::shared_ptr<Features> data)
{
	SGMatrix<float64_t> bagged_outputs =
	    apply_outputs_without_combination(data);

	require(m_labels, "Labels not set.");
	require(
	    m_labels->get_label_type() == LT_MULTICLASS,
	    "Labels ({}) are not compatible with multiclass.",
	    m_labels->get_name());

	auto labels_multiclass = std::dynamic_pointer_cast<MulticlassLabels>(m_labels);
	auto num_samples = bagged_outputs.size() / m_num_bags;
	auto num_classes = labels_multiclass->get_num_classes();

	auto pred = std::make_shared<MulticlassLabels>(num_samples);
	pred->allocate_confidences_for(num_classes);

	SGMatrix<float64_t> class_probabilities(num_classes, num_samples);
	class_probabilities.zero();

	for (auto i = 0; i < num_samples; ++i)
	{
		for (auto j = 0; j < m_num_bags; ++j)
		{
			int32_t class_idx = bagged_outputs(i, j);
			class_probabilities(class_idx, i) += 1;
		}
	}

	linalg::scale(class_probabilities, class_probabilities, 1.0 / m_num_bags);

	for (auto i = 0; i < num_samples; ++i)
		pred->set_multiclass_confidences(i, class_probabilities.get_column(i));

	SGVector<float64_t> combined = m_combination_rule->combine(bagged_outputs);
	pred->set_labels(combined);

	return pred;
}

std::shared_ptr<RegressionLabels> BaggingMachine::apply_regression(std::shared_ptr<Features> data)
{
	return std::make_shared<RegressionLabels>(apply_get_outputs(data));
}

SGVector<float64_t> BaggingMachine::apply_get_outputs(std::shared_ptr<Features> data)
{
	ASSERT(data != NULL);
	require(m_combination_rule != NULL, "Combination rule is not set!");

	auto output = apply_outputs_without_combination(data);
	return m_combination_rule->combine(output);
}

SGMatrix<float64_t>
BaggingMachine::apply_outputs_without_combination(std::shared_ptr<Features> data)
{
	ASSERT(m_num_bags == m_bags.size());

	SGMatrix<float64_t> output(data->get_num_vectors(), m_num_bags);
	output.zero();

#pragma omp parallel for
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		auto m = m_bags.at(i);
		auto l = m->apply(data);
		SGVector<float64_t> lv;
		if (l!=NULL)
			lv = l->as<DenseLabels>()->get_labels();
		else
			error("NULL returned by apply method");

		float64_t* bag_results = output.get_column_vector(i);
		sg_memcpy(bag_results, lv.vector, lv.vlen * sizeof(float64_t));
	}

	return output;
}

bool BaggingMachine::train_machine(std::shared_ptr<Features> data)
{
	require(m_machine != NULL, "Machine is not set!");
	require(m_num_bags > 0, "Number of bag is not set!");

	if (data)
	{
		m_features = data;

		ASSERT(m_features->get_num_vectors() == m_labels->get_num_labels());
	}

	// if bag size is not provided, set it equal to number of training vectors
	if (m_bag_size == 0)
		m_bag_size = m_features->get_num_vectors();

	// clear the array, if previously trained
	m_bags.clear();

	// reset the oob index vector
	m_all_oob_idx = SGVector<bool>(m_features->get_num_vectors());
	m_all_oob_idx.zero();


	m_oob_indices = std::make_shared<DynamicObjectArray>();

	SGMatrix<index_t> rnd_indicies(m_bag_size, m_num_bags);
	random::fill_array(rnd_indicies, 0, m_bag_size - 1, m_prng);

	auto pb = SG_PROGRESS(range(m_num_bags));
#pragma omp parallel for
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		auto c=std::dynamic_pointer_cast<Machine>(m_machine->clone());
		ASSERT(c != NULL);
		SGVector<index_t> idx(
		    rnd_indicies.get_column_vector(i), m_bag_size, false);

		std::shared_ptr<Features> features;
		std::shared_ptr<Labels> labels;

		if (env()->get_num_threads() == 1)
		{
			features = m_features;
			labels = m_labels;
		}
		else
		{
			features = m_features->shallow_subset_copy();
			labels = m_labels->shallow_subset_copy();
		}

		labels->add_subset(idx);
		/* TODO:
		   if it's a binary labeling ensure that
		   there's always samples of both classes
		if ((m_labels->get_label_type() == LT_BINARY))
		{
		    while (true) {
		        if (!m_labels->ensure_valid()) {
		            m_labels->remove_subset();
		            idx.random(0, m_features->get_num_vectors());
		            m_labels->add_subset(idx);
		            continue;
		        }
		        break;
		    }
		}
		*/
		features->add_subset(idx);
		set_machine_parameters(c, idx);
		c->set_labels(labels);
		c->train(features);
		features->remove_subset();
		labels->remove_subset();

#pragma omp critical
		{
		// get out of bag indexes
		auto oob = get_oob_indices(idx);
		m_oob_indices->push_back(oob);

		// add trained machine to bag array
		m_bags.push_back(c);
		}

		pb.print_progress();
	}
	pb.complete();

	return true;
}

void BaggingMachine::set_machine_parameters(std::shared_ptr<Machine> m, SGVector<index_t> idx)
{
}

void BaggingMachine::register_parameters()
{
	SG_ADD(&m_features, kFeatures, "Train features for bagging");
	SG_ADD(
	    &m_num_bags, kNBags, "Number of bags", ParameterProperties::HYPER);
	SG_ADD(
	    &m_bag_size, kBagSize, "Number of vectors per bag",
	    ParameterProperties::HYPER);
	SG_ADD(&m_bags, kBags, "Bags array");
	SG_ADD(
	    &m_combination_rule, kCombinationRule,
	    "Combination rule to use for aggregating", ParameterProperties::HYPER);
	SG_ADD(&m_all_oob_idx, kAllOobIdx, "Indices of all oob vectors");
	SG_ADD(&m_oob_indices, kOobIndices, "OOB indices for each machine");
	SG_ADD(&m_machine, kMachine, "machine to use for bagging");
	SG_ADD(&m_oob_evaluation_metric, kOobEvaluationMetric,
	    "metric to calculate the oob error");
	watch_method(kOobError, &BaggingMachine::get_oob_error);
}

void BaggingMachine::set_num_bags(int32_t num_bags)
{
	m_num_bags = num_bags;
}

int32_t BaggingMachine::get_num_bags() const
{
	return m_num_bags;
}

void BaggingMachine::set_bag_size(int32_t bag_size)
{
	m_bag_size = bag_size;
}

int32_t BaggingMachine::get_bag_size() const
{
	return m_bag_size;
}

std::shared_ptr<Machine> BaggingMachine::get_machine() const
{
	return m_machine;
}

void BaggingMachine::set_machine(std::shared_ptr<Machine> machine)
{
	m_machine = machine;
}

void BaggingMachine::init()
{
	m_machine = nullptr;
	m_features = nullptr;
	m_combination_rule = nullptr;
	m_labels = nullptr;
	m_num_bags = 0;
	m_bag_size = 0;
	m_all_oob_idx = SGVector<bool>();
	m_oob_indices = nullptr;
	m_oob_evaluation_metric = nullptr;
}

void BaggingMachine::set_combination_rule(std::shared_ptr<CombinationRule> rule)
{
	m_combination_rule = rule;
}

std::shared_ptr<CombinationRule> BaggingMachine::get_combination_rule() const
{
	return m_combination_rule;
}

float64_t BaggingMachine::get_oob_error() const
{
	require(
	    m_oob_evaluation_metric, "Out of bag evaluation metric is not set!");
	require(m_combination_rule, "Combination rule is not set!");
	require(m_bags.size() > 0, "BaggingMachine is not trained!");

	SGMatrix<float64_t> output(
	    m_features->get_num_vectors(), m_bags.size());
	if (m_labels->get_label_type() == LT_REGRESSION)
		output.zero();
	else
		output.set_const(NAN);

	/* TODO: add parallel support of applying the OOBs
	  only possible when add_subset is thread-safe
	#pragma omp parallel for num_threads(env()->get_num_threads())
	*/
	for (index_t i = 0; i < m_bags.size(); i++)
	{
		auto m = m_bags.at(i);
		auto current_oob
			= m_oob_indices->get_element<DynamicArray<index_t>>(i);

		SGVector<index_t> oob(
		    current_oob->get_array(), current_oob->get_num_elements(), false);
		m_features->add_subset(oob);

		auto l = m->apply(m_features);
		SGVector<float64_t> lv;
		if (l!=NULL)
			lv = std::dynamic_pointer_cast<DenseLabels>(l)->get_labels();
		else
			error("NULL returned by apply method");

		// assign the values in the matrix (NAN) that are in-bag!
		for (index_t j = 0; j < oob.vlen; j++)
			output(oob[j], i) = lv[j];

		m_features->remove_subset();



	}

	std::vector<index_t> idx;
	for (index_t i = 0; i < m_features->get_num_vectors(); i++)
	{
		if (m_all_oob_idx[i])
			idx.push_back(i);
	}

	SGVector<float64_t> combined = m_combination_rule->combine(output);
	SGVector<float64_t> lab(idx.size());
	for (int32_t i = 0; i < lab.vlen; i++)
		lab[i] = combined[idx[i]];

	std::shared_ptr<Labels> predicted = NULL;
	switch (m_labels->get_label_type())
	{
		case LT_BINARY:
			predicted = std::make_shared<BinaryLabels>(lab);
			break;

		case LT_MULTICLASS:
			predicted = std::make_shared<MulticlassLabels>(lab);
			break;

		case LT_REGRESSION:
			predicted = std::make_shared<RegressionLabels>(lab);
			break;

		default:
			error("Unsupported label type");
	}


	m_labels->add_subset(SGVector<index_t>(idx.data(), idx.size(), false));
	float64_t res = m_oob_evaluation_metric->evaluate(predicted, m_labels);
	m_labels->remove_subset();

	return res;
}

std::shared_ptr<DynamicArray<index_t>> BaggingMachine::get_oob_indices(const SGVector<index_t>& in_bag)
{
	SGVector<bool> out_of_bag(m_features->get_num_vectors());
	out_of_bag.set_const(true);

	// mark the ones that are in_bag
	for (index_t i = 0; i < in_bag.vlen; i++)
		out_of_bag[in_bag[i]] &= false;

	auto oob = std::make_shared<DynamicArray<index_t>>();
	// store the indicies of vectors that are out of the bag
	for (index_t i = 0; i < out_of_bag.vlen; i++)
	{
		if (out_of_bag[i])
		{
			oob->push_back(i);
			m_all_oob_idx[i] = true;
		}
	}
	return oob;
}
