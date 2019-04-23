/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Soeren Sonnenburg, Soumyajit De,
 *          Chiyuan Zhang, Viktor Gal, Fernando Iglesias, Bjoern Esser,
 *          Yuyu Zhang
 */

#ifndef __SUBSETSTACK_H_
#define __SUBSETSTACK_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/features/Subset.h>


namespace shogun
{

/** @brief class to add subset support to another class. A SubsetStackStack instance
 * should be added and wrapper methods to all interfaces should be added.
 *
 * The subsets are organized as a stack. One can add arbritary many index sets
 * which always refer to the current subset (identity if no element in stack)
 * This way, one can define "subsets of subsets". Use the index conversion
 * method to get original indices.
 *
 * Internally, a stack of active subsets is saved. Each time an index set is
 * added, a new element will be put on stack, using the old element to get
 * mapping. On  removal, the last element on stack will be removed. This is done
 * for computational convenience.
 */
class SubsetStack: public SGObject
{
public:
	/** Constructor. Creates empty subset stack
	 */
	SubsetStack();

	/** copy constructor
	 */
	SubsetStack(const SubsetStack& other);

	/** destructor */
	virtual ~SubsetStack() = default;

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "SubsetStack"; }

	/** Adds a subset of indices on top of the current subsets (possibly
	 * subset of subset). Every call causes a new active index vector
	 * to be stored. Added subsets can be removed one-by-one. If this is not
	 * needed, add_subset_in_place() should be used (does not store
	 * intermediate index vectors)
	 *
	 * @param subset subset of indices to add
	 * */
	virtual void add_subset(const SGVector<index_t>& subset);

	/** Sets/changes latest added subset. This allows to add multiple subsets
	 * with in-place memory requirements. They cannot be removed one-by-one
	 * afterwards, only the latest active can. If this is needed, use
	 * add_subset(). If no subset is active, this just adds.
	 *
	 * @param subset subset of indices to replace the latest one with.
	 * */
	virtual void add_subset_in_place(SGVector<index_t> subset);

	/** removes the last added index set */
	virtual void remove_subset();

	/** removes all subsets, leaving this subset being the pure identity */
	virtual void remove_all_subsets();

	/** @return size of active subset */
	inline index_t get_size() const
	{
		if (!has_subsets())
			SG_WARNING("SubsetStack::get_size(): No subset in stack!\n")

		return has_subsets() ? m_active_subset->get_size() : -1;
	}

	/** @return true iff subset was added */
	virtual bool has_subsets() const
	{
		return (m_active_subsets_stack.size() > 0);
	}

	/** returns last (active) subset of the stack
	 *
	 * @return active subset
	 */
	std::shared_ptr<Subset> get_last_subset() const { return m_active_subset; }

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
	std::vector<std::shared_ptr<Subset>> m_active_subsets_stack;

	/** active index subset. Last element on stack for quick access */
	std::shared_ptr<Subset> m_active_subset;
};

}
#endif /* __SUBSETSTACK_H_ */
