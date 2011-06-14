/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelectionParametersNameNode.h"

using namespace shogun;

CModelSelectionParametersNameNode::CModelSelectionParametersNameNode()
{
	m_node_name=NULL;

}

CModelSelectionParametersNameNode::CModelSelectionParametersNameNode(
		const char* node_name) :
	m_node_name(node_name)
{

}

CModelSelectionParametersNameNode::~CModelSelectionParametersNameNode()
{

}

void CModelSelectionParametersNameNode::append_child(
		CModelSelectionParametersTree* child)
{
	SG_ERROR("Not possible to append child to %s\n", get_name());
}

void CModelSelectionParametersNameNode::print(const char* prefix)
{
	SG_PRINT("%s%s\n", prefix, m_node_name);
	CModelSelectionParametersTree::print(prefix);
}
