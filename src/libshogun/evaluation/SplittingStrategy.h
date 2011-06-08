/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __SPLITTINGSTRATEGY_H_
#define __SPLITTINGSTRATEGY_H_

#include "base/SGObject.h"
#include "base/DynArray.h"

namespace shogun
{

class CLabels;

/** @brief Abstract base class for all splitting types.
 * Takes a CLabels instance and generates a desired number of subsets which are
 * being accessed by their indices via the method  generate_subset_indices(...).
 *
 * When being extended, the abstract method build_subsets() has to be
 * implemented AND to be called in the constructor of sub-classes.
 * Implementations have to fill the DynArray<index_t> elements in the
 * (inherited) m_subset_indices variable. Note that these elements are already
 * created by the constructor of this class - they just have to be filled. Every
 * element represents one index subset.
 */
class CSplittingStrategy: public CSGObject
{
public:
	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 * @param num_subsets desired number of subsets, the labels are split into
	 */
	CSplittingStrategy(CLabels* labels, index_t num_subsets);

	/** destructor */
	virtual ~CSplittingStrategy();

	/** generates a newly created SGVector<index_t> with indices of the subset
	 * with the desired index
	 *
	 * @param subset_idx subset index of the to be generated vector indices
	 * @return newly created vector of subset indices of the specified subset
	 */
	SGVector<index_t>* generate_subset_indices(index_t subset_idx);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "SplittingStrategy";
	}

protected:
	/** Abstract method.
	 * Has to fill the elements of the m_subset_indices variable with concrete
	 * indices. Note that DynArray<index_t> instances for every subset are
	 * created in the constructor of this class - they just have to be filled.
	 */
	virtual void build_subsets()=0;

	CLabels* m_labels;
	DynArray<DynArray<index_t>*> m_subset_indices;
};
}

#endif /* __SPLITTINGSTRATEGY_H_ */
