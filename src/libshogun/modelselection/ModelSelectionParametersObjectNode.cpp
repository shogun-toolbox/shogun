/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelectionParametersObjectNode.h"
#include "modelselection/ModelSelectionParametersTree.h"

using namespace shogun;

CModelSelectionParametersObjectNode::CModelSelectionParametersObjectNode()
{
	m_sgobject=NULL;
}

CModelSelectionParametersObjectNode::CModelSelectionParametersObjectNode(
		CSGObject* sgobject) :
	m_sgobject(sgobject)
{
	SG_REF(m_sgobject);
}

CModelSelectionParametersObjectNode::~CModelSelectionParametersObjectNode()
{
	SG_UNREF(m_sgobject);
}

void CModelSelectionParametersObjectNode::append_child(
		CModelSelectionParametersTree* child)
{
	m_child_nodes.append_element(child);
}

void CModelSelectionParametersObjectNode::print(const char* prefix)
{
	SG_PRINT("%s%s\n", prefix, m_sgobject->get_name());
	CModelSelectionParametersTree::print(prefix);
}
