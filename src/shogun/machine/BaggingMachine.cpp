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
	m_machine = NULL;
	SG_UNREF(m_features);
	m_features = NULL;
	SG_UNREF(m_combination_rule);
	m_combination_rule = NULL;
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

	SGMatrix<float64_t> output(data->get_num_vectors(), m_num_bags);
	output.zero();

	//#pragma omp parallel for num_threads(parallel->get_num_threads())
	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags.get_element(i));
		CLabels* l = m->apply(data);
		SGVector<float64_t> lv = l->get_values();
		float64_t* bag_results = output.get_column_vector(i);
		memcpy(bag_results, lv.vector, lv.vlen*sizeof(float64_t));

		SG_UNREF(l);
	}

	SGVector<float64_t> combined = m_combination_rule->combine(output);

	return combined;
}

bool CBaggingMachine::train_machine(CFeatures* data)
{
	REQUIRE(m_machine != NULL, "Machine is not set!");
	REQUIRE(m_bag_size > 0, "Bag size is not set!");
	REQUIRE(m_num_bags > 0, "Number of bag is not set!");

	//#pragma omp parallel for num_threads(parallel->get_num_threads())
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
		m_bags.append_element(c);
	}

	return true;
}

void CBaggingMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_features, "features", "Train features for bagging", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_bags, "num_bags", "Number of bags", MS_AVAILABLE);
	SG_ADD(&m_bag_size, "bag_size", "Number of vectors per bag", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_bags, "bags", "Bags array", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_combination_rule, "combination_rule", "Combination rule to use for aggregating", MS_AVAILABLE);
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
	SG_UNREF(m_machine);
	m_machine = NULL;

	SG_REF(machine);
	m_machine = machine;
}

void CBaggingMachine::init()
{
	m_bags.clear_array();
	m_machine = NULL;
	m_features = NULL;
	m_combination_rule = NULL;
	m_labels = NULL;
	m_num_bags = 0;
	m_bag_size = 0;
}

void CBaggingMachine::set_combination_rule(CCombinationRule* rule)
{
	SG_UNREF(m_combination_rule);
	m_combination_rule = NULL;

	SG_REF(rule);
	m_combination_rule = rule;
}

CCombinationRule* CBaggingMachine::get_combination_rule() const
{
	SG_REF(m_combination_rule);
	return m_combination_rule;
}
