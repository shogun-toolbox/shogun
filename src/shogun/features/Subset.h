/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __SUBSET_H_
#define __SUBSET_H_

#include <shogun/base/SGObject.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief class for adding subset support to a class. Provides an interface for
 * getting/setting subset_matrices and index conversion.
 * Do not inherit from this class, use it as variable. Write wrappers for all
 * get/set functions.
 */
class CSubset: public CSGObject
{
public:
	/** default constructor, do not use */
	CSubset();

	/** constructor
	 *
	 * @param subset_idx vector of subset indices, is deleted in destructor
	 */
	CSubset(const SGVector<index_t>& subset_idx);

	/** destructor */
	virtual ~CSubset();

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "Subset"; }

	/** get size of subset
	 * @return size of subset
	 */
	inline const index_t get_size() const { return m_subset_idx.vlen; }

	/* @ return largest index in subset */
	inline const index_t get_max_index() const
	{
		return CMath::max(m_subset_idx.vector, m_subset_idx.vlen);
	}

	/* @ return smallest index in subset */
	inline const index_t get_min_index() const
	{
		return CMath::min(m_subset_idx.vector, m_subset_idx.vlen);
	}

	/** @return a copy of this instance with a copy of the index vector */
	CSubset* duplicate();

	/** returns the corresponding real index (in array) of a subset index
	 * (if there is a subset)
	 *
	 * @ return array index of the provided subset index
	 */
	inline index_t subset_idx_conversion(index_t idx) const
	{
		return m_subset_idx.vector ? m_subset_idx.vector[idx] : idx;
	}
private:
	void init();

private:
	const SGVector<index_t> m_subset_idx;
};

}
#endif /* __SUBSET_H_ */
