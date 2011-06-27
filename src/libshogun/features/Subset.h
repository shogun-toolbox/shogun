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

#include "base/SGObject.h"

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
	CSubset();
	virtual ~CSubset();

	/** removes (and deletes) the current subset indices matrix */
	void remove_subset();

	/** getter for the subset indices
	 *
	 * @return SGVector with subset indices array (no copy)
	 */
	SGVector<index_t> get_subset() const { return m_subset; }

	bool has_subset() const { return m_subset.vector!=NULL; }

	/** setter for the subset indices. deletes any old subset vector before
	 *
	 * @param subset SGVector with subset indices array (directly stored)
	 */
	void set_subset(SGVector<index_t> subset);

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "Subset"; }

	/** returns the corresponding real index (in array) of a subset index
	 * (if there is a subset)
	 *
	 * @ return array index of the provided subset index
	 */
	inline index_t subset_idx_conversion(index_t idx) const
	{
		return m_subset.vector ? m_subset.vector[idx] : idx;
	}

private:
	SGVector<index_t> m_subset;
};

}
#endif /* __SUBSET_H_ */
