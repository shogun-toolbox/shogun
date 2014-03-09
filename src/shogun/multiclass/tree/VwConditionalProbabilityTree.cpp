/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <vector>
#include <stack>
#include <shogun/multiclass/tree/VwConditionalProbabilityTree.h>

using namespace shogun;
using namespace std;

CMulticlassLabels* CVwConditionalProbabilityTree::apply_multiclass(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STREAMING_VW)
			SG_ERROR("Expected StreamingVwFeatures\n")
		set_features(dynamic_cast<CStreamingVwFeatures*>(data));
	}

	vector<int32_t> predicts;

	m_feats->start_parser();
	while (m_feats->get_next_example())
	{
		predicts.push_back(apply_multiclass_example(m_feats->get_example()));
		m_feats->release_example();
	}
	m_feats->end_parser();

	CMulticlassLabels *labels = new CMulticlassLabels(predicts.size());
	for (size_t i=0; i < predicts.size(); ++i)
		labels->set_int_label(i, predicts[i]);
	return labels;
}

int32_t CVwConditionalProbabilityTree::apply_multiclass_example(VwExample* ex)
{
	ex->ld->label = FLT_MAX; // this will disable VW learning from this example

	compute_conditional_probabilities(ex);
	SGVector<float64_t> probs(m_leaves.size());
	for (map<int32_t,bnode_t*>::iterator it = m_leaves.begin(); it != m_leaves.end(); ++it)
	{
		probs[it->first] = accumulate_conditional_probability(it->second);
	}
	return SGVector<float64_t>::arg_max(probs.vector, 1, probs.vlen);
}

void CVwConditionalProbabilityTree::compute_conditional_probabilities(VwExample *ex)
{
	stack<bnode_t *> nodes;
	nodes.push((bnode_t*) m_root);

	while (!nodes.empty())
	{
		bnode_t *node = nodes.top();
		nodes.pop();
		if (node->left())
		{
			nodes.push(node->left());
			nodes.push(node->right());

			// don't calculate for leaf
			node->data.p_right = train_node(ex, node);
		}
	}
}

float64_t CVwConditionalProbabilityTree::accumulate_conditional_probability(bnode_t *leaf)
{
	float64_t prob = 1;
	bnode_t *par = (bnode_t*) leaf->parent();
	while (par != NULL)
	{
		if (leaf == par->left())
			prob *= (1-par->data.p_right);
		else
			prob *= par->data.p_right;

		leaf = par;
		par = (bnode_t*) leaf->parent();
	}

	return prob;
}

bool CVwConditionalProbabilityTree::train_machine(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STREAMING_VW)
			SG_ERROR("Expected StreamingVwFeatures\n")
		set_features(dynamic_cast<CStreamingVwFeatures*>(data));
	}
	else
	{
		if (!m_feats)
			SG_ERROR("No data features provided\n")
	}

	m_machines->reset_array();
	SG_UNREF(m_root);
	m_root = NULL;

	m_leaves.clear();

	m_feats->start_parser();
	for (int32_t ipass=0; ipass < m_num_passes; ++ipass)
	{
		while (m_feats->get_next_example())
		{
			train_example(m_feats->get_example());
			m_feats->release_example();
		}

		if (ipass < m_num_passes-1)
			m_feats->reset_stream();
	}
	m_feats->end_parser();

	return true;
}

void CVwConditionalProbabilityTree::train_example(VwExample *ex)
{
	int32_t label = static_cast<int32_t>(ex->ld->label);

	if (m_root == NULL)
	{
		m_root = new bnode_t();
		m_root->data.label = label;
		printf("  insert %d %p\n", label, m_root);
		m_leaves.insert(make_pair(label,(bnode_t*) m_root));
		m_root->machine(create_machine(ex));
		return;
	}

	if (m_leaves.find(label) != m_leaves.end())
	{
		train_path(ex, m_leaves[label]);
	}
	else
	{
		bnode_t *node = (bnode_t*) m_root;
		while (node->left() != NULL)
		{
			// not a leaf
			bool is_left = which_subtree(node, ex);
			if (is_left)
				ex->ld->label = 0;
			else
				ex->ld->label = 1;
			train_node(ex, node);

			if (is_left)
				node = node->left();
			else
				node = node->right();
		}

		printf("  remove %d %p\n", node->data.label, m_leaves[node->data.label]);
		m_leaves.erase(node->data.label);

		bnode_t *left_node = new bnode_t();
		left_node->data.label = node->data.label;
		node->data.label = -1;
		CVowpalWabbit *node_vw = dynamic_cast<CVowpalWabbit *>(m_machines->get_element(node->machine()));
		CVowpalWabbit *vw = new CVowpalWabbit(node_vw);
		SG_UNREF(node_vw);
		vw->set_learner();
		m_machines->push_back(vw);
		left_node->machine(m_machines->get_num_elements()-1);
		printf("  insert %d %p\n", left_node->data.label, left_node);
		m_leaves.insert(make_pair(left_node->data.label, left_node));
		node->left(left_node);

		bnode_t *right_node = new bnode_t();
		right_node->data.label = label;
		right_node->machine(create_machine(ex));
		printf("  insert %d %p\n", label, right_node);
		m_leaves.insert(make_pair(label, right_node));
		node->right(right_node);
	}
}

void CVwConditionalProbabilityTree::train_path(VwExample *ex, bnode_t *node)
{
	ex->ld->label = 0;
	train_node(ex, node);

	bnode_t *par = (bnode_t*) node->parent();
	while (par != NULL)
	{
		if (par->left() == node)
			ex->ld->label = 0;
		else
			ex->ld->label = 1;

		train_node(ex, par);
		node = par;
		par = (bnode_t*) node->parent();
	}
}

float64_t CVwConditionalProbabilityTree::train_node(VwExample *ex, bnode_t *node)
{
	CVowpalWabbit *vw = dynamic_cast<CVowpalWabbit*>(m_machines->get_element(node->machine()));
	ASSERT(vw)
	float64_t pred = vw->predict_and_finalize(ex);
	if (ex->ld->label != FLT_MAX)
		vw->get_learner()->train(ex, ex->eta_round);
	SG_UNREF(vw);
	return pred;
}

int32_t CVwConditionalProbabilityTree::create_machine(VwExample *ex)
{
	CVowpalWabbit *vw = new CVowpalWabbit(m_feats);
	vw->set_learner();
	ex->ld->label = 0;
	vw->predict_and_finalize(ex);
	m_machines->push_back(vw);
	return m_machines->get_num_elements()-1;
}
