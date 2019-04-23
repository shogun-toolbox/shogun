/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, 
 *          Evgeniy Andreev, Soumyajit De, Yuyu Zhang
 */

#ifndef __SUBSET_H_
#define __SUBSET_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
/** @brief Wrapper class for an index subset which is used by SubsetStack. */
class Subset: public SGObject
{
	friend class SubsetStack;

public:
	/** default constructor, do not use */
	Subset();

	/** constructor
	 *
	 * @param subset_idx vector of subset indices.
	 */
	Subset(const SGVector<index_t>& subset_idx);

	/** destructor */
	virtual ~Subset();

	/** @return size of subset index array */
	index_t get_size() const { return m_subset_idx.vlen; }

	/** @return name of the SGSerializable */
	inline const char* get_name() const { return "Subset"; }

	/** get subset indices */
	SGVector<index_t> get_subset_idx() const { return m_subset_idx; }

private:
	void init();

private:
	SGVector<index_t> m_subset_idx;
};

}
#endif /* __SUBSET_H_ */
