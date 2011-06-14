/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelectionParametersTree.h"

using namespace shogun;

CModelSelectionParametersTree::CModelSelectionParametersTree()
{
}

CModelSelectionParametersTree::~CModelSelectionParametersTree()
{
}

void CModelSelectionParametersTree::destroy()
{
	for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
		m_child_nodes[i]->destroy();

	delete this;
}

bool CModelSelectionParametersTree::has_children()
{
	return m_child_nodes.get_num_elements()>0;
}

void CModelSelectionParametersTree::print(const char* prefix)
{
	/* enlarge prefix by one \t */
	int32_t length=strlen(prefix);
	char* new_prefix=new char[length+1];
	for (index_t i=0; i<length; ++i)
		new_prefix[i]='\t';

	/* end character */
	new_prefix[length]='\0';

	/* print all children with new prefix */
	for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
		m_child_nodes[i]->print(new_prefix);

	/* clean up */
	delete[] new_prefix;
}
