/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef INDEXBLOCKTREE_H_
#define INDEXBLOCKTREE_H_

#include <lib/IndexBlock.h>
#include <lib/IndexBlockRelation.h>

namespace shogun
{

/** @brief class IndexBlockTree used to represent
 * tree guided feature relation.
 *
 * Can be constructed via CIndexBlock instance having
 * sub blocks, adjacency matrix or precomputed indices.
 */
class CIndexBlockTree : public CIndexBlockRelation
{
public:

	/** default constructor */
	CIndexBlockTree();

	/** constructor from index block
	 * @param root_block root block of the tree
	 */
	CIndexBlockTree(CIndexBlock* root_block);

	/** constructor from adjacency matrix
	 * @param adjacency_matrix adjacency matrix
	 * @param include_supernode whether to include supernode
	 */
	CIndexBlockTree(SGMatrix<float64_t> adjacency_matrix, bool include_supernode);

	/** constructor from general precomputed indices
	 * each node is represented with indices G[ind_t.min:ind_t.max]
	 * and weight ind_t.weight
	 * @param G custom G containing mapping indices
	 * @param ind_t custom ind_t containing flatten parameters of each node [min,max,weight]
	 */
	CIndexBlockTree(SGVector<float64_t> G, SGVector<float64_t> ind_t);

	/** constructor from basic precomputed indices
	 * each node is represented with indices ind_t.min:ind_t.max
	 * and weight ind_t.weight
	 * @param ind_t custom ind_t containing flatten parameters of each node [min,max,weight]
	 */
	CIndexBlockTree(SGVector<float64_t> ind_t);

	/** destructor */
	virtual ~CIndexBlockTree();

	/** get root IndexBlock */
	CIndexBlock* get_root_block() const;

	/** set root block */
	void set_root_block(CIndexBlock* root_block);

	/** returns information about blocks in
	 * SLEP "ind" format
	 */
	virtual SGVector<index_t> get_SLEP_ind();

	/** returns information about blocks in
	 * SLEP "G" format
	 */
	virtual SGVector<float64_t> get_SLEP_G();

	/** returns information about blocks relations
	 * in SLEP "ind_t" format
	 */
	virtual SGVector<float64_t> get_SLEP_ind_t() const;

	/** returns relation type */
	virtual EIndexBlockRelationType get_relation_type() const { return TREE; }

	/** whether relation is general, i.e. not well ordered */
	bool is_general() const;

	/** get name */
	const char* get_name() const { return "IndexBlockTree"; };

protected:

	/** root block */
	CIndexBlock* m_root_block;

	/** general */
	bool m_general;

	/** precomputed ind_t */
	SGVector<float64_t> m_precomputed_ind_t;

	/** precomputed G */
	SGVector<float64_t> m_precomputed_G;
};

}
#endif

