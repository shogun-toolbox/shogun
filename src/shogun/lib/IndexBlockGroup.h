/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Yuyu Zhang, Björn Esser
 */

#ifndef INDEXBLOCKGROUP_H_
#define INDEXBLOCKGROUP_H_

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/IndexBlockRelation.h>

namespace shogun
{

class CIndexBlock;
class CList;

/** @brief class IndexBlockGroup used to represent
 * group-based feature relation.
 *
 * Currently can be constructed with a few CIndexBlock
 * instances.
 */
class CIndexBlockGroup : public CIndexBlockRelation
{
public:

	/** default constructor */
	CIndexBlockGroup();

	/** destructor */
	virtual ~CIndexBlockGroup();

	/** add IndexBlock to the group
	 * @param block IndexBlock to add
	 */
	void add_block(CIndexBlock* block);

	/** remove IndexBlock from the group
	 * @param block IndexBlock to remove
	 */
	void remove_block(CIndexBlock* block);

	/** returns information about IndexBlocks in
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	virtual EIndexBlockRelationType get_relation_type() const { return GROUP; }

	/** get name */
	const char* get_name() const { return "IndexBlockGroup"; };

protected:

	/** blocks in group */
	CList* m_blocks;

};

}
#endif

