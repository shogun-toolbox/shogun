/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __SUBSETSTACK_H_
#define __SUBSETSTACK_H_

#include <base/SGObject.h>
#include <mathematics/Math.h>
#include <lib/DynamicObjectArray.h>
#include <features/Subset.h>


namespace shogun
{

/** @brief class to add subset support to another class. A CSubsetStackStack instance
 * should be added and wrapper methods to all interfaces should be added.
 *
 * The subsets are organized as a stack. One can add arbritary many index sets
 * which always refer to the current subset (identity if no element in stack)
 * This way, one can define "subsets of subsets". Use the index conversion
 * method to get original indices.
 *
 * Internally, a stack of active subsets is saved. Each time an index set is
 * added, a new element will be put on stack, using the old element to get
 * mappig. On  removal, the last element on stack will be removed. This is done
 * for computational convenience.
 */
class CSubsetStack: public CSGObject
{
public:
	/** Constructor. Creates empty subset stack
	 */
	CSubsetStack();

	/** copy constructor
	 */
	CSubsetStack(const CSubsetStack& other);

	/** destructor */
	virtual ~CSubsetStack();

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "SubsetStack"; }

	/** adds an index set to the current subset
	 * @param subset index subset to add
	 * */
	virtual void add_subset(SGVector<index_t> subset);

	/** removes the last added index set */
	virtual void remove_subset();

	/** removes all subsets, leaving this subset being the pure identity */
	virtual void remove_all_subsets();

	/** @return size of active subset */
	inline index_t get_size() const
	{
		if (!has_subsets())
			SG_WARNING("CSubsetStack::get_size(): No subset in stack!\n")

		return has_subsets() ? m_active_subset->get_size() : -1;
	}

	/** @return true iff no subset was added */
	virtual bool has_subsets() const
	{
		return m_active_subsets_stack->get_num_elements();
	}

	/** returns last (active) subset of the stack
	 *
	 * @return active subset
	 */
	CSubset* get_last_subset() const { return m_active_subset; }

	/** returns the corresponding real index of a subset index
	 * Maps through all added subsets in stack.
	 *
	 * @return array index of the provided subset index
	 */
	inline index_t subset_idx_conversion(index_t idx) const
	{
		return m_active_subset ? m_active_subset->m_subset_idx.vector[idx] : idx;
	}

private:
	/** registers and initializes parameters */
	void init();

private:
	/** stack of active subsets. All active subsets are stored to avoid
	 * recomputing them when subsets are removed. There is always the identity
	 * subset as first element in here (only internal visible, has_subsets()
	 * returns false if only this identity is present) */
	CDynamicObjectArray* m_active_subsets_stack;

	/** active index subset. Last element on stack for quick access */
	CSubset* m_active_subset;
};

}
#endif /* __SUBSETSTACK_H_ */
