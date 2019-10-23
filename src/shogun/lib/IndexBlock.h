/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Yuyu Zhang, Thoralf Klein,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef INDEXBLOCK_H_
#define INDEXBLOCK_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief class IndexBlock used to represent
 * contiguous indices of one group (e.g. block of related features)
 */
class IndexBlock : public SGObject
{
public:

	/** default constructor */
	IndexBlock();

	/** constructor
	 * @param min_index smallest index of the index block
	 * @param max_index largest index of the index block
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	IndexBlock(index_t min_index, index_t max_index,
	      float64_t weight=1.0, const char* name="task");

	/** destructor */
	~IndexBlock();

	/** get min index */
	index_t get_min_index() const { return m_min_index; }
	/** set max index */
	void set_min_index(index_t min_index) { m_min_index = min_index; }
	/** get min index */
	index_t get_max_index() const { return m_max_index; }
	/** set max index */
	void set_max_index(index_t max_index) { m_max_index = max_index; }
	/** get weight */
	float64_t get_weight() const { return m_weight; }
	/** set weight */
	void set_weight(float64_t weight) { m_weight = weight; }

	/** get name */
	virtual const char* get_name() const { return "IndexBlock"; };

	/** get subtasks */
	std::vector<std::shared_ptr<IndexBlock>> get_sub_blocks();

	/** get num subtasks */
	int32_t get_num_sub_blocks();

	/** adds sub-block
	 * @param sub_block subtask to add
	 */
	void add_sub_block(const std::shared_ptr<IndexBlock>& sub_block);

private:
	/** min index */
	index_t m_min_index;

	/** max index */
	index_t m_max_index;

	/** weight */
	float64_t m_weight;

	/** subtasks */
	std::vector<std::shared_ptr<IndexBlock>> m_sub_blocks;

};

}
#endif
