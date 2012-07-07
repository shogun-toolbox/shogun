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

#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/IndexBlockRelation.h>

namespace shogun
{

class CIndexBlockTree : public CIndexBlockRelation
{
public:

	/** default constructor */
	CIndexBlockTree();

	/** constructor
	 * @param root_block root block of the tree
	 */
	CIndexBlockTree(CIndexBlock* root_block);

	/** constructor
	 * @param adjacency_matrix adjacency matrix
	 */
	CIndexBlockTree(SGMatrix<float64_t> adjacency_matrix);
	
	/** constructor
	 * @param G custom G
	 * @param ind_t custom ind_t
	 */
	CIndexBlockTree(SGVector<float64_t> G, SGVector<float64_t> ind_t);

	/** destructor */
	virtual ~CIndexBlockTree();

	/** get root IndexBlock */
	CIndexBlock* get_root_block() const;

	/** set root block */
	void set_root_block(CIndexBlock* root_block);

	/** returns information about blocks in 
	 * SLEP "ind" format
	 */
	virtual SGVector<index_t> get_SLEP_ind() const;

	virtual SGVector<float64_t> get_SLEP_G() const;

	/** returns information about blocks relations
	 * in SLEP "ind_t" format
	 */
	virtual SGVector<float64_t> get_SLEP_ind_t() const;

	virtual EIndexBlockRelationType get_relation_type() const { return TREE; }

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

