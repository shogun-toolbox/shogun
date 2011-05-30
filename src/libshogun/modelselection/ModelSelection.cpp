/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelection.h"

using namespace shogun;

CModelSelection::CModelSelection()
{

}

CModelSelection::CModelSelection(CMachine* machine, CFeatures* features,
		CLabels* labels) :
	m_machine(machine), m_features(features), m_labels(labels)
{

}

CModelSelection::~CModelSelection()
{

}

void CModelSelection::set_parameters(CModelSelectionParameters* param_tree_root)
{
	m_param_tree_root=param_tree_root;
}

void CModelSelection::run()
{
//	if (!m_param_tree_root)
//		SG_ERROR("no parameters set\n");
//
//	CSGObject* current=m_param_tree_root->get_child_nodes()->get_first_element();
//	while(current)
//	{
//
//		current=m_param_tree_root->get_child_nodes()->get_next_element();
//	}

}
