/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __SUBSET_H_
#define __SUBSET_H_

#include <shogun/base/SGObject.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Wrapper class for an index subset which is used by SubsetStack.
 * Currently the SGVector with the indices is possibly freed with free_vector()
 * This might change in the future when there is reference counting for
 * SGVectors. For now, please set do_free flag to false and handle memory stuff
 * yourself. Using subset vectors with do_free==true may cause trouble */
class CSubset: public CSGObject
{
	friend class CSubsetStack;

public:
	/** default constructor, do not use */
	CSubset();

	/** constructor
	 *
	 * @param subset_idx vector of subset indices. Please do set th do_free flag
	 * to false and handle vector memory managment yourself.
	 * TODO once there is reference couting for vectors, use it.
	 */
	CSubset(const SGVector<index_t>& subset_idx);

	/** destructor. Calls free_vector of index vector */
	virtual ~CSubset();

	/** @return size of subset index array */
	index_t get_size() const { return m_subset_idx.vlen; }

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "Subset"; }
private:
	void init();

private:
	SGVector<index_t> m_subset_idx;
};

}
#endif /* __SUBSET_H_ */
