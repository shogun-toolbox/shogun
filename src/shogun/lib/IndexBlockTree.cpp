/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/IndexBlockTree.h>
#include <shogun/lib/SGMatrix.h>
#include <utility>
#include <vector>

using namespace std;
using namespace shogun;

struct tree_node_t
{
	tree_node_t** desc;
	int32_t n_desc;
	int32_t sub_nodes_count;
	int32_t idx;
};

struct block_tree_node_t
{
	block_tree_node_t(int32_t min, int32_t max, float64_t w)
	{
		t_min_index = min;
		t_max_index = max;
		weight = w;
	}
	int32_t t_min_index, t_max_index;
	float64_t weight;
};

int count_sub_nodes_recursive(tree_node_t* node, int32_t self)
{
	if (node->n_desc==0)
	{
		return 1;
	}
	else
	{
		int c = 0;
		for (int32_t i=0; i<node->n_desc; i++)
		{
			c += count_sub_nodes_recursive(node->desc[i], self);
		}
		if (self)
			node->sub_nodes_count = c;
		return c + self;
	}
}

void print_tree(tree_node_t* node, int tabs)
{
	for (int32_t t=0; t<tabs; t++)
		io::print("  ");
	io::print("{} {}\n",node->idx, node->sub_nodes_count);
	for (int32_t i=0; i<node->n_desc; i++)
		print_tree(node->desc[i],tabs+1);
}

int32_t fill_G_recursive(tree_node_t* node, vector<int32_t>* G)
{
	int32_t c=1;
	G->push_back(node->idx);
	for (int32_t i=0; i<node->n_desc; i++)
		c+= fill_G_recursive(node->desc[i], G);
	return c;
}

void fill_ind_recursive(tree_node_t* node, vector<block_tree_node_t>* tree_nodes, int32_t lower)
{
	int32_t l = lower;
	for (int32_t i=0; i<node->n_desc; i++)
	{
		int32_t c = node->desc[i]->sub_nodes_count;
		if (c>0)
		{
			tree_nodes->push_back(block_tree_node_t(l,l+c-1,1.0));
			fill_ind_recursive(node->desc[i], tree_nodes, l);
			l+=c;
		}
		else
			l++;
	}
}

void collect_tree_nodes_recursive(const std::shared_ptr<IndexBlock>& subtree_root_block, vector<block_tree_node_t>* tree_nodes)
{
	auto sub_blocks = subtree_root_block->get_sub_blocks();
	if (sub_blocks.size()>0)
	{
		for (const auto& iterator: sub_blocks)
		{
			SG_DEBUG("Block [{} {}] ",iterator->get_min_index(), iterator->get_max_index())
			tree_nodes->push_back(block_tree_node_t(iterator->get_min_index(),iterator->get_max_index(),iterator->get_weight()));
			if (iterator->get_num_sub_blocks()>0)
				collect_tree_nodes_recursive(iterator, tree_nodes);
		}
	}
}

IndexBlockTree::IndexBlockTree() :
	IndexBlockRelation(), m_root_block(NULL),
	m_general(false)
{

}

IndexBlockTree::IndexBlockTree(std::shared_ptr<IndexBlock> root_block) : IndexBlockRelation(),
	m_root_block(NULL), m_general(false)
{
	set_root_block(std::move(root_block));
}

IndexBlockTree::IndexBlockTree(SGMatrix<float64_t> adjacency_matrix, bool include_supernode) :
	IndexBlockRelation(),
	m_root_block(NULL), m_general(true)
{
	ASSERT(adjacency_matrix.num_rows == adjacency_matrix.num_cols)
	int32_t n_features = adjacency_matrix.num_rows;

	// well ordering is assumed

	tree_node_t* nodes = SG_CALLOC(tree_node_t, n_features);

	int32_t* nz_row = SG_CALLOC(int32_t, n_features);
	for (int32_t i=0; i<n_features; i++)
	{
		nodes[i].idx = i;
		nodes[i].sub_nodes_count = 0;
		int32_t c = 0;
		for (int32_t j=i; j<n_features; j++)
		{
			if (adjacency_matrix(j,i)!=0.0)
				nz_row[c++] = j;
		}
		nodes[i].n_desc = c;
		nodes[i].desc = SG_MALLOC(tree_node_t*, c);
		for (int32_t j=0; j<c; j++)
		{
			nodes[i].desc[j] = &nodes[nz_row[j]];
		}
		if (nz_row[c] == n_features)
			break;
	}
	SG_FREE(nz_row);

	vector<int32_t> G;
	vector<int32_t> ind_t;
	int current_l_idx = 1;
	for (int32_t i=1; i<n_features; i++)
	{
		if (nodes[i].n_desc > 0)
		{
			int sub_count = fill_G_recursive(&nodes[i],&G);
			ind_t.push_back(current_l_idx);
			ind_t.push_back(current_l_idx+sub_count-1);
			ind_t.push_back(1.0);
			current_l_idx += sub_count;
		}
	}
	/*
	io::print("[");
	for (int32_t i=0; i<G.size(); i++)
		io::print(" {} ",G[i]);
	io::print("]\n");
	io::print("[");
	for (int32_t i=0; i<ind_t.size(); i++)
		io::print(" {} ",ind_t[i]);
	io::print("]\n");
	*/

	int32_t supernode_offset = include_supernode ? 3 : 0;
	m_precomputed_ind_t = SGVector<float64_t>((int32_t)ind_t.size()+supernode_offset);
	if (include_supernode)
	{
		m_precomputed_ind_t[0] = -1;
		m_precomputed_ind_t[1] = -1;
		m_precomputed_ind_t[2] = 1.0;
	}
	for (int32_t i=0; i<(int32_t)ind_t.size(); i++)
		m_precomputed_ind_t[i+supernode_offset] = ind_t[i];
	m_precomputed_G = SGVector<float64_t>((int32_t)G.size());
	for (int32_t i=0; i<(int32_t)G.size(); i++)
		m_precomputed_G[i] = G[i] + 1;
	m_general = true;
	/*
	count_sub_nodes_recursive(nodes,1);
	print_tree(nodes,0);
	int32_t n_leaves = count_sub_nodes_recursive(nodes,0);
	m_precomputed_ind_t = SGVector<float64_t>((n_features-n_leaves)*3);
	io::print("n_leaves = {}\n",n_leaves);
	vector<block_tree_node_t> blocks;
	fill_ind_recursive(nodes, &blocks, 1);
	m_precomputed_ind_t[0] = -1;
	m_precomputed_ind_t[1] = -1;
	m_precomputed_ind_t[2] = 1.0;

	for (int32_t i=0; i<(int)blocks.size(); i++)
	{
		m_precomputed_ind_t[3+3*i+0] = blocks[i].t_min_index;
		m_precomputed_ind_t[3+3*i+1] = blocks[i].t_max_index;
		m_precomputed_ind_t[3+3*i+2] = blocks[i].weight;
	}
	*/
	for (int32_t i=0; i<n_features; i++)
		SG_FREE(nodes[i].desc);
	SG_FREE(nodes);
}

IndexBlockTree::IndexBlockTree(const SGVector<float64_t>& G, const SGVector<float64_t>& ind_t) :
	IndexBlockRelation(),
	m_root_block(NULL), m_general(true)
{
	m_precomputed_G = G;
	m_precomputed_ind_t = ind_t;
}

IndexBlockTree::IndexBlockTree(SGVector<float64_t> ind_t) :
	IndexBlockRelation(),
	m_root_block(NULL), m_general(false)
{
	m_precomputed_ind_t = ind_t;
}

IndexBlockTree::~IndexBlockTree()
{

}

std::shared_ptr<IndexBlock> IndexBlockTree::get_root_block() const
{

	return m_root_block;
}

void IndexBlockTree::set_root_block(std::shared_ptr<IndexBlock> root_block)
{


	m_root_block = std::move(root_block);
}

SGVector<index_t> IndexBlockTree::get_SLEP_ind()
{
	not_implemented(SOURCE_LOCATION);
	return SGVector<index_t>();
}

SGVector<float64_t> IndexBlockTree::get_SLEP_G()
{
	return m_precomputed_G;
}

bool IndexBlockTree::is_general() const
{
	return m_general;
}

SGVector<float64_t> IndexBlockTree::get_SLEP_ind_t() const
{
	if (m_precomputed_ind_t.vlen)
		return m_precomputed_ind_t;

	else
	{
		ASSERT(m_root_block)

		vector<IndexBlock> blocks;
		vector<block_tree_node_t> tree_nodes;

		collect_tree_nodes_recursive(m_root_block, &tree_nodes);

		SGVector<float64_t> ind_t(3+3*tree_nodes.size());
		// supernode
		ind_t[0] = -1;
		ind_t[1] = -1;
		ind_t[2] = 1.0;

		for (int32_t i=0; i<(int32_t)tree_nodes.size(); i++)
		{
			ind_t[3+i*3] = tree_nodes[i].t_min_index + 1;
			ind_t[3+i*3+1] = tree_nodes[i].t_max_index;
			ind_t[3+i*3+2] = tree_nodes[i].weight;
		}

		return ind_t;
	}
}
