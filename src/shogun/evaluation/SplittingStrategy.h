/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Fernando Iglesias, Yuyu Zhang,
 *          Sergey Lisitsyn
 */

#ifndef __SPLITTINGSTRATEGY_H_
#define __SPLITTINGSTRATEGY_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

class Labels;

/** @brief Abstract base class for all splitting types.
 * Takes a Labels instance and generates a desired number of subsets which are
 * being accessed by their indices via the method  generate_subset_indices(...).
 *
 * When being extended, the abstract method build_subsets() has to be
 * implemented.
 * build_subsets implementations HAVE TO call reset_subsets before, in order
 * to allow calling them from the outside. Also they HAVE to set the m_is_filled
 * flag to true (otherwise there will be an error when accessing the index sets)
 *
 * Implementations have to (re)fill the DynamicArray<index_t> elements in the
 * (inherited) m_subset_indices variable. Note that these elements are already
 * created by the constructor of this class - they just have to be filled. Every
 * element represents one index subset.
 *
 * Calling the method agains means that the indices are rebuilt.
 */
class SplittingStrategy: public SGObject
{
public:
	/** constructor */
	SplittingStrategy();

	/** constructor
	 *
	 * @param labels labels to be (possibly) used for splitting
	 * @param num_subsets desired number of subsets, the labels are split into
	 */
	SplittingStrategy(std::shared_ptr<Labels> labels, index_t num_subsets);

	/** destructor */
	virtual ~SplittingStrategy();

	/** generates a newly created SGVector<index_t> with indices of the subset
	 * with the desired index
	 *
	 * @param subset_idx subset index of the to be generated vector indices
	 * @return newly created vector of subset indices of the specified
	 * subset is written here.
	 *
	 * Error if there are no index sets
	 */
	SGVector<index_t> generate_subset_indices(index_t subset_idx);

	/** generates a newly created SGVector<index_t> with inverse indices of the
	 * subset with the desired index. inverse here means all other indices.
	 *
	 * @param subset_idx subset index of the to be generated inverse indices
	 * @return newly created vector of the subset's inverse indices is
	 * written here.
	 *
	 * Error if there are no index sets
	 */
	SGVector<index_t> generate_subset_inverse(index_t subset_idx);

	/** @return number of subsets. */
	index_t get_num_subsets() const;

	/** Abstract method.
	 * Has to refill the elements of the m_subset_indices variable with concrete
	 * indices. Note that DynamicArray<index_t> instances for every subset are
	 * created in the constructor of this class - they just have to be filled.
	 */
	virtual void build_subsets()=0;

protected:
	/** resets the current subsets, meaning that all the arrays of indices will
	 * be empty again. To be called before build_subsets. */
	void reset_subsets();

private:
	void init();


protected:

	/** labels */
	std::shared_ptr<Labels> m_labels;

	/** subset indices */
	std::shared_ptr<DynamicObjectArray> m_subset_indices;

	/** additional variable to store number of index subsets */
	index_t m_num_subsets;

	/** flag to check whether there is a set of index sets stored. If not,
	 * call build_subsets() */
	bool m_is_filled;
};
}

#endif /* __SPLITTINGSTRATEGY_H_ */
