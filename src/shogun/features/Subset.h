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
 * SGVectors. For now, please either use an index vector only once per subset
 * and set do_free flag to true, or set it to false and handle memory stuff
 * yourself */
class CSubset: public CSGObject
{
	friend class CSubsetStack;

public:
	/** default constructor, do not use */
	CSubset();

	/** constructor
	 *
	 * @param subset_idx vector of subset indices. free_vector is called in
	 * destructor, so decide whether it should be deleted when passing vector
	 */
	CSubset(const SGVector<index_t>& subset_idx);

	/** destructor. Calls free_vector of index vector*/
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
