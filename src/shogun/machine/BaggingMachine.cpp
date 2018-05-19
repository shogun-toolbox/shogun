/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Saurabh Mahindre, Heiko Strathmann, Thoralf Klein,
 *          Olivier NGuyen, Bjoern Esser, Weijie Lin
 */

#include <shogun/ensemble/CombinationRule.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/base/progress.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <shogun/evaluation/Evaluation.h>

using namespace shogun;

CBaggingMachine::CBaggingMachine()
	: CMachine()
{
	init();
	register_parameters();
}

CBaggingMachine::CBaggingMachine(CFeatures* features, CLabels* labels)
	: CMachine()
{
	init();
	register_parameters();

	set_labels(labels);

	SG_REF(features);
	m_features = features;
}

CBaggingMachine::~CBaggingMachine()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_combination_rule);
	SG_UNREF(m_bags);
	SG_UNREF(m_oob_indices);
}

CBinaryLabels* CBaggingMachine::apply_binary(CFeatures* data)
{
	SGMatrix<float64_t> output = apply_outputs_without_combination(data);

	CMeanRule* mean_rule = new CMeanRule();

	SGVector<float64_t> labels = m_combination_rule->combine(output);
	SGVector<float64_t> probabilities = mean_rule->combine(output);

	float64_t threshold = 0.5;
	CBinaryLabels* pred = new CBinaryLabels(probabilities, threshold);

	SG_UNREF(mean_rule);

	return pred;
}

CMulticlassLabels* CBaggingMachine::apply_multiclass(CFeatures* data)
{
	SGMatrix<float64_t> bagged_outputs =
	    apply_outputs_without_combination(data);

	REQUIRE(m_labels, "Labels not set.\n");
	REQUIRE(
	    m_labels->get_label_type() == LT_MULTICLASS,
	    "Labels (%s) are not compatible with multiclass.\n",
	    m_labels->get_name());

	auto labels_multiclass = dynamic_cast<CMulticlassLabels*>(m_labels);
	auto num_samples = bagged_outputs.size() / m_num_bags;
	auto num_classes = labels_multiclass->get_num_classes();

	CMulticlassLabels* pred = new CMulticlassLabels(num_samples);
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

CRegressionLabels* CBaggingMachine::apply_regression(CFeatures* data)
{
	return new CRegressionLabels(apply_get_outputs(data));
}

SGVector<float64_t> CBaggingMachine::apply_get_outputs(CFeatures* data)
{
	ASSERT(data != NULL);
	REQUIRE(m_combination_rule != NULL, "Combination rule is not set!");

	SGMatrix<float64_t> output = apply_outputs_without_combination(data);
	SGVector<float64_t> combined = m_combination_rule->combine(output);

	return combined;
}

SGMatrix<float64_t>
CBaggingMachine::apply_outputs_without_combination(CFeatures* data)
{
	ASSERT(m_num_bags == m_bags->get_num_elements());

	SGMatrix<float64_t> output(data->get_num_vectors(), m_num_bags);
	output.zero();

	#pragma omp parallel for
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags->get_element(i));
		CLabels* l = m->apply(data);
		SGVector<float64_t> lv;
		if (l!=NULL)
			lv = dynamic_cast<CDenseLabels*>(l)->get_labels();
		else
			SG_ERROR("NULL returned by apply method\n");

		float64_t* bag_results = output.get_column_vector(i);
		sg_memcpy(bag_results, lv.vector, lv.vlen*sizeof(float64_t));

		SG_UNREF(l);
		SG_UNREF(m);
	}

	return output;
}

bool CBaggingMachine::train_machine(CFeatures* data)
{
	REQUIRE(m_machine != NULL, "Machine is not set!");
	REQUIRE(m_num_bags > 0, "Number of bag is not set!");

	if (data)
	{
		SG_REF(data);
		SG_UNREF(m_features);
		m_features = data;

		ASSERT(m_features->get_num_vectors() == m_labels->get_num_labels());
	}

	// if bag size is not provided, set it equal to number of training vectors
	if (m_bag_size==0)
		m_bag_size = m_features->get_num_vectors();

	// clear the array, if previously trained
	m_bags->reset_array();

	// reset the oob index vector
	m_all_oob_idx = SGVector<bool>(m_features->get_num_vectors());
	m_all_oob_idx.zero();

	SG_UNREF(m_oob_indices);
	m_oob_indices = new CDynamicObjectArray();

	SGMatrix<index_t> rnd_indicies(m_bag_size, m_num_bags);
	for (index_t i = 0; i < m_num_bags*m_bag_size; ++i)
		rnd_indicies.matrix[i] = CMath::random(0, m_bag_size-1);

	auto pb = progress(range(m_num_bags));
	#pragma omp parallel for
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* c=dynamic_cast<CMachine*>(m_machine->clone());
		ASSERT(c != NULL);
		SGVector<index_t> idx(rnd_indicies.get_column_vector(i), m_bag_size, false);

		CFeatures* features;
		CLabels* labels;

		if (get_global_parallel()->get_num_threads()==1)
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
		set_machine_parameters(c,idx);
		c->set_labels(labels);
		c->train(features);
		features->remove_subset();
		labels->remove_subset();

		#pragma omp critical
		{
		// get out of bag indexes
		CDynamicArray<index_t>* oob = get_oob_indices(idx);
		m_oob_indices->push_back(oob);

		// add trained machine to bag array
		m_bags->push_back(c);
		}

		if (get_global_parallel()->get_num_threads()!=1)
		{
			SG_UNREF(features);
			SG_UNREF(labels);
		}

		SG_UNREF(c);
		pb.print_progress();
	}
	pb.complete();
	
	return true;
}

void CBaggingMachine::set_machine_parameters(CMachine* m, SGVector<index_t> idx)
{
}

void CBaggingMachine::register_parameters()
{
	SG_ADD(
	    &m_features, "features", "Train features for bagging",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_num_bags, "num_bags", "Number of bags", MS_AVAILABLE);
	SG_ADD(&m_bag_size, "bag_size", "Number of vectors per bag", MS_AVAILABLE);
	SG_ADD(&m_bags, "bags", "Bags array", MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_combination_rule, "combination_rule",
	    "Combination rule to use for aggregating", MS_AVAILABLE);
	SG_ADD(&m_all_oob_idx, "all_oob_idx", "Indices of all oob vectors",
			MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_oob_indices, "oob_indices", "OOB indices for each machine",
	    MS_NOT_AVAILABLE);
}

void CBaggingMachine::set_num_bags(int32_t num_bags)
{
	m_num_bags = num_bags;
}

int32_t CBaggingMachine::get_num_bags() const
{
	return m_num_bags;
}

void CBaggingMachine::set_bag_size(int32_t bag_size)
{
	m_bag_size = bag_size;
}

int32_t CBaggingMachine::get_bag_size() const
{
	return m_bag_size;
}

CMachine* CBaggingMachine::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

void CBaggingMachine::set_machine(CMachine* machine)
{
	SG_REF(machine);
	SG_UNREF(m_machine);
	m_machine = machine;
}

void CBaggingMachine::init()
{
	m_bags = new CDynamicObjectArray();
	m_machine = NULL;
	m_features = NULL;
	m_combination_rule = NULL;
	m_labels = NULL;
	m_num_bags = 0;
	m_bag_size = 0;
	m_all_oob_idx = SGVector<bool>();
	m_oob_indices = NULL;
}

void CBaggingMachine::set_combination_rule(CCombinationRule* rule)
{
	SG_REF(rule);
	SG_UNREF(m_combination_rule);
	m_combination_rule = rule;
}

CCombinationRule* CBaggingMachine::get_combination_rule() const
{
	SG_REF(m_combination_rule);
	return m_combination_rule;
}

float64_t CBaggingMachine::get_oob_error(CEvaluation* eval) const
{
	REQUIRE(m_combination_rule != NULL, "Combination rule is not set!");
	REQUIRE(m_bags->get_num_elements() > 0, "BaggingMachine is not trained!");

	SGMatrix<float64_t> output(m_features->get_num_vectors(), m_bags->get_num_elements());
	if (m_labels->get_label_type() == LT_REGRESSION)
		output.zero();
	else
		output.set_const(NAN);

	/* TODO: add parallel support of applying the OOBs
	  only possible when add_subset is thread-safe
	#pragma omp parallel for num_threads(parallel->get_num_threads())
	*/
	for (index_t i = 0; i < m_bags->get_num_elements(); i++)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags->get_element(i));
		CDynamicArray<index_t>* current_oob
			= dynamic_cast<CDynamicArray<index_t>*>(m_oob_indices->get_element(i));

		SGVector<index_t> oob(current_oob->get_array(), current_oob->get_num_elements(), false);
		m_features->add_subset(oob);

		CLabels* l = m->apply(m_features);
		SGVector<float64_t> lv;
		if (l!=NULL)
			lv = dynamic_cast<CDenseLabels*>(l)->get_labels();
		else
			SG_ERROR("NULL returned by apply method\n");

		// assign the values in the matrix (NAN) that are in-bag!
		for (index_t j = 0; j < oob.vlen; j++)
			output(oob[j], i) = lv[j];

		m_features->remove_subset();
		SG_UNREF(current_oob);
		SG_UNREF(m);
		SG_UNREF(l);
	}

	std::vector<index_t> idx;
	for (index_t i = 0; i < m_features->get_num_vectors(); i++)
	{
		if (m_all_oob_idx[i])
			idx.push_back(i);
	}

	SGVector<float64_t> combined = m_combination_rule->combine(output);
	SGVector<float64_t> lab(idx.size());
	for (int32_t i=0;i<lab.vlen;i++)
		lab[i]=combined[idx[i]];

	CLabels* predicted = NULL;
	switch (m_labels->get_label_type())
	{
		case LT_BINARY:
			predicted = new CBinaryLabels(lab);
			break;

		case LT_MULTICLASS:
			predicted = new CMulticlassLabels(lab);
			break;

		case LT_REGRESSION:
			predicted = new CRegressionLabels(lab);
			break;

		default:
			SG_ERROR("Unsupported label type\n");
	}
	SG_REF(predicted);

	m_labels->add_subset(SGVector<index_t>(idx.data(), idx.size(), false));
	float64_t res = eval->evaluate(predicted, m_labels);
	m_labels->remove_subset();

	SG_UNREF(predicted);
	return res;
}

CDynamicArray<index_t>* CBaggingMachine::get_oob_indices(const SGVector<index_t>& in_bag)
{
	SGVector<bool> out_of_bag(m_features->get_num_vectors());
	out_of_bag.set_const(true);

	// mark the ones that are in_bag
	for (index_t i = 0; i < in_bag.vlen; i++)
		out_of_bag[in_bag[i]] &= false;

	CDynamicArray<index_t>* oob = new CDynamicArray<index_t>();
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

