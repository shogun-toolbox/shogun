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
 * Currently, all index sets that are given to a CSubset instance are COPIED.
 * This is bad because these should be re-usable, however, until there is no
 * reference counting for SGVectors, this wont be possible. TODO */
class CSubset: public CSGObject
{
	friend class CSubsetStack;

public:
	/** default constructor, do not use */
	CSubset();

	/** constructor
	 *
	 * @param subset_idx vector of subset indices. Is always copied.
	 * TODO, this might be changes once reference counting for SGVectors is
	 * there
	 */
	CSubset(SGVector<index_t> subset_idx);

	/** destructor */
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
