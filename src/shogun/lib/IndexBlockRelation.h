/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Yuyu Zhang
 */

#ifndef INDEXBLOCKRELATION_H_
#define INDEXBLOCKRELATION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{

class List;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EIndexBlockRelationType
{
	GROUP,
	TREE
};
#endif

/** @brief class IndexBlockRelation
 *
 */
class IndexBlockRelation : public SGObject
{
public:

	/** default constructor */
	IndexBlockRelation()
	{
	}

	/** destructor */
	virtual ~IndexBlockRelation()
	{
	}

	/** get name */
	const char* get_name() const { return "IndexBlockRelation"; };

	/** get relation type */
	virtual EIndexBlockRelationType get_relation_type() const = 0;

protected:

	/** check list of blocks */
	bool check_blocks_list(std::shared_ptr<List> blocks);

};

}
#endif
