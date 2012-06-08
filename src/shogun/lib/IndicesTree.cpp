/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/IndicesTree.h>

using namespace shogun;

SGVector<float64_t> CIndicesTree::get_ind() const
{
	SG_WARNING("Not implemented, uses one supernode\n");
	SGVector<float64_t> ind(3);
	ind[0] = -1;
	ind[1] = -1;
	ind[2] = 1.0;

	return ind;
}

void CIndicesTree::print_tree() const
{
	print_tree_recursive(m_root_node,0);
}

void CIndicesTree::print_tree_recursive(CIndicesTreeNode* node, int32_t level) const
{
	for (int32_t i=0; i<level; i++)
		SG_PRINT("\t");

	SG_PRINT("[ ");
	for (int32_t i=0; i<node->node_indices.vlen; i++)
		SG_PRINT(" %d ",node->node_indices[i]);
	SG_PRINT("] %f \n", node->weight);

	for (int32_t i=0; i<node->child_nodes.index(); i++)
		print_tree_recursive(node->child_nodes[i],level+1);
}
