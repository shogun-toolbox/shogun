/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef INDEXBLOCKGROUP_H_
#define INDEXBLOCKGROUP_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/IndexBlockRelation.h>

namespace shogun
{

class CIndexBlock;
class CList;

/** @brief class IndexBlockGroup used to represent
 * group-based feature relation.
 *
 * Currently can be constructed with a few CIndexBlock
 * instances.
 */
class CIndexBlockGroup : public CIndexBlockRelation
{
public:

	/** default constructor */
	CIndexBlockGroup();

	/** destructor */
	virtual ~CIndexBlockGroup();

	/** add IndexBlock to the group
	 * @param block IndexBlock to add
	 */
	void add_block(CIndexBlock* block);

	/** remove IndexBlock from the group
	 * @param block IndexBlock to remove
	 */
	void remove_block(CIndexBlock* block);

	/** returns information about IndexBlocks in
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	virtual EIndexBlockRelationType get_relation_type() const { return GROUP; }

	/** get name */
	const char* get_name() const { return "IndexBlockGroup"; };

protected:

	/** blocks in group */
	CList* m_blocks;

};

}
#endif

