/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/tree/ConditionalProbabilityTree.h>

using namespace shogun;
using namespace std;

bool CConditionalProbabilityTree::train_machine(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STREAMING_VW)
			SG_ERROR("Expected StreamingVwFeatures\n");
		set_features(dynamic_cast<CStreamingVwFeatures*>(data));
	}
	else
	{
		if (!m_feats)
			SG_ERROR("No data features provided\n");
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

		m_feats->reset_stream();
	}
	m_feats->end_parser();
}

void CConditionalProbabilityTree::train_example(VwExample *ex)
{
	int32_t label = static_cast<int32_t>(ex->ld->label);

	if (m_root == NULL)
	{
		m_root = new CTreeMachineNode();
		m_leaves.insert(make_pair(label, m_root));
		m_root->machine(create_machine(ex));
		return;
	}

	if (m_leaves.find(label) != m_leaves.end())
	{
		train_path(ex, m_leaves[label]);
	}
	else
	{
		CTreeMachineNode *node = m_root;
		while (m_root->left() != NULL)
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

		// TODO: node->left should be a clone of node
		// TODO: remove node from m_leaves, replace it with left
		CTreeMachineNode *new_node = new CTreeMachineNode();
		new_node->machine(create_machine(ex));
		m_leaves.insert(make_pair(label, new_node));
		node->right(new_node);
	}
}

void CConditionalProbabilityTree::train_path(VwExample *ex, CTreeMachineNode *node)
{
	ex->ld->label = 0;
	train_node(ex, node);

	CTreeMachineNode *par = node->parent();
	while (par != NULL)
	{
		if (par->left() == node)
			ex->ld->label = 0;
		else
			ex->ld->label = 1;

		train_node(ex, par);
		node = par;
		par = node->parent();
	}
}

void CConditionalProbabilityTree::train_node(VwExample *ex, CTreeMachineNode *node)
{
	CVowpalWabbit *vw = dynamic_cast<CVowpalWabbit*>(m_machines->get_element(node->machine()));
	ASSERT(vw);
	vw->predict_and_finalize(ex);
	SG_UNREF(vw);
}

int32_t CConditionalProbabilityTree::create_machine(VwExample *ex)
{
	CVowpalWabbit *vw = new CVowpalWabbit(m_feats);
	vw->set_learner();
	ex->ld->label = 0;
	vw->predict_and_finalize(ex);
	m_machines->push_back(vw);
	return m_machines->get_num_elements();
}
