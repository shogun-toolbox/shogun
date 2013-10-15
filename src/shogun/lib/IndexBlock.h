/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef INDEXBLOCK_H_
#define INDEXBLOCK_H_

#include <shogun/lib/SGVector.h>
#include <shogun/lib/List.h>

namespace shogun
{

/** @brief class IndexBlock used to represent
 * contiguous indices of one group (e.g. block of related features)
 */
class CIndexBlock : public CSGObject
{
public:

	/** default constructor */
	CIndexBlock();

	/** constructor
	 * @param min_index smallest index of the index block
	 * @param max_index largest index of the index block
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	CIndexBlock(index_t min_index, index_t max_index,
	      float64_t weight=1.0, const char* name="task");

	/** destructor */
	~CIndexBlock();

	/** get min index */
	index_t get_min_index() const { return m_min_index; }
	/** set max index */
	void set_min_index(index_t min_index) { m_min_index = min_index; }
	/** get min index */
	index_t get_max_index() const { return m_max_index; }
	/** set max index */
	void set_max_index(index_t max_index) { m_max_index = max_index; }
	/** get weight */
	float64_t get_weight() const { return m_weight; }
	/** set weight */
	void set_weight(float64_t weight) { m_weight = weight; }

	/** get name */
	virtual const char* get_name() const { return "IndexBlock"; };

	/** get subtasks */
	CList* get_sub_blocks();

	/** get num subtasks */
	int32_t get_num_sub_blocks();

	/** adds sub-block
	 * @param sub_block subtask to add
	 */
	void add_sub_block(CIndexBlock* sub_block);

private:
	/** min index */
	index_t m_min_index;

	/** max index */
	index_t m_max_index;

	/** weight */
	float64_t m_weight;

	/** subtasks */
	CList* m_sub_blocks;

};

}
#endif
