/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/machine/BaggingMachine.h>
#include <shogun/base/Parameter.h>

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
	SGVector<float64_t> combined_vector = apply_get_outputs(data);

	CBinaryLabels* pred = new CBinaryLabels(combined_vector);
	return pred;
}

CMulticlassLabels* CBaggingMachine::apply_multiclass(CFeatures* data)
{
	SGVector<float64_t> combined_vector = apply_get_outputs(data);

	CMulticlassLabels* pred = new CMulticlassLabels(combined_vector);
	return pred;
}

CRegressionLabels* CBaggingMachine::apply_regression(CFeatures* data)
{
	SGVector<float64_t> combined_vector = apply_get_outputs(data);

	CRegressionLabels* pred = new CRegressionLabels(combined_vector);

	return pred;
}

SGVector<float64_t> CBaggingMachine::apply_get_outputs(CFeatures* data)
{
	ASSERT(data != NULL);
	REQUIRE(m_combination_rule != NULL, "Combination rule is not set!");
	ASSERT(m_num_bags == m_bags->get_num_elements());
  	
	SGMatrix<float64_t> output(data->get_num_vectors(), m_num_bags);
	output.zero();

	#pragma omp parallel for num_threads(parallel->get_num_threads())
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags->get_element(i));
		CLabels* l = m->apply(data);
		SGVector<float64_t> lv = l->get_values();
		float64_t* bag_results = output.get_column_vector(i);
		memcpy(bag_results, lv.vector, lv.vlen*sizeof(float64_t));

		SG_UNREF(l);
		SG_UNREF(m);
	}

	SGVector<float64_t> combined = m_combination_rule->combine(output);

	return combined;
}

bool CBaggingMachine::train_machine(CFeatures* data)
{
	REQUIRE(m_machine != NULL, "Machine is not set!");
	REQUIRE(m_bag_size > 0, "Bag size is not set!");
	REQUIRE(m_num_bags > 0, "Number of bag is not set!");

	if (data)
	{
		SG_REF(data);
		SG_UNREF(m_features);
		m_features = data;

		ASSERT(m_features->get_num_vectors() == m_labels->get_num_labels());
	}

	// bag size << number of feature vector
	ASSERT(m_bag_size < m_features->get_num_vectors());

	// clear the array, if previously trained
	m_bags->reset_array();

	// reset the oob index vector
	m_all_oob_idx = SGVector<bool>(m_features->get_num_vectors());
	m_all_oob_idx.zero();

	SG_UNREF(m_oob_indices);
	m_oob_indices = new CDynamicObjectArray();
	
	/*
	  TODO: enable multi-threaded learning. This requires views support
		on CFeatures 
	#pragma omp parallel for num_threads(parallel->get_num_threads())
	*/
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* c = dynamic_cast<CMachine*>(m_machine->clone());
		ASSERT(c != NULL);
		SGVector<index_t> idx(m_bag_size);
		idx.random(0, m_features->get_num_vectors()-1);
		m_labels->add_subset(idx);
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
		m_features->add_subset(idx);
		c->set_labels(m_labels);
		c->train(m_features);
		m_features->remove_subset();
		m_labels->remove_subset();

		// get out of bag indexes
		CDynamicArray<index_t>* oob = get_oob_indices(idx);
		m_oob_indices->push_back(oob);

		// add trained machine to bag array
		m_bags->append_element(c);
	}

	return true;
}

void CBaggingMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_features, "features", "Train features for bagging", 
		MS_NOT_AVAILABLE);
	SG_ADD(&m_num_bags, "num_bags", "Number of bags", MS_AVAILABLE);
	SG_ADD(&m_bag_size, "bag_size", "Number of vectors per bag", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_bags, "bags", "Bags array", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_combination_rule, "combination_rule", 
		"Combination rule to use for aggregating", MS_AVAILABLE);
	SG_ADD(&m_all_oob_idx, "all_oob_idx", "Indices of all oob vectors",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_oob_indices, "oob_indices", 
			"OOB indices for each machine", MS_NOT_AVAILABLE);
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
		oob.display_vector();
		m_features->add_subset(oob);

		CLabels* l = m->apply(m_features);
		SGVector<float64_t> lv = l->get_values();

		// assign the values in the matrix (NAN) that are in-bag!
		for (index_t j = 0; j < oob.vlen; j++)
			output(oob[j], i) = lv[j];

		m_features->remove_subset();
		SG_UNREF(current_oob);
		SG_UNREF(m);
		SG_UNREF(l);
	}
	output.display_matrix();

	DynArray<index_t> idx;
	for (index_t i = 0; i < m_features->get_num_vectors(); i++)
	{
		if (m_all_oob_idx[i])
			idx.push_back(i);
	}

	SGVector<float64_t> combined = m_combination_rule->combine(output);
	CLabels* predicted = NULL;
	switch (m_labels->get_label_type())
	{
		case LT_BINARY:
			predicted = new CBinaryLabels(combined);
			break;

		case LT_MULTICLASS:
			predicted = new CMulticlassLabels(combined);
			break;

		case LT_REGRESSION:
			predicted = new CRegressionLabels(combined);
			break;

		default:
			SG_ERROR("Unsupported label type\n");
	}
	
	m_labels->add_subset(SGVector<index_t>(idx.get_array(), idx.get_num_elements(), false));
	float64_t res = eval->evaluate(predicted, m_labels);
	m_labels->remove_subset();

	return res;
}

CDynamicArray<index_t>* CBaggingMachine::get_oob_indices(const SGVector<index_t>& in_bag)
{
	SGVector<bool> out_of_bag(m_features->get_num_vectors());
	out_of_bag.set_const(true);

	// mark the ones that are in_bag
	index_t oob_count = m_features->get_num_vectors();
	for (index_t i = 0; i < in_bag.vlen; i++)
	{
		if (out_of_bag[in_bag[i]])
		{
			out_of_bag[in_bag[i]] = false;
			oob_count--;
		}
	}

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

