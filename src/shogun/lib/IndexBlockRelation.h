/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef INDEXBLOCKRELATION_H_
#define INDEXBLOCKRELATION_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CList;

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
class CIndexBlockRelation : public CSGObject
{
public:

	/** default constructor */
	CIndexBlockRelation()
	{
	}

	/** destructor */
	virtual ~CIndexBlockRelation()
	{
	}

	/** get name */
	const char* get_name() const { return "IndexBlockRelation"; };

	/** get relation type */
	virtual EIndexBlockRelationType get_relation_type() const = 0;

protected:

	/** check list of blocks */
	bool check_blocks_list(CList* blocks);

};

}
#endif
