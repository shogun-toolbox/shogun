/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTIONPARAMETERSTREE_H_
#define __MODELSELECTIONPARAMETERSTREE_H_

#include "base/SGObject.h"
#include "base/DynArray.h"

namespace shogun
{

class CModelSelectionParametersTree : public CSGObject
{
public:
	CModelSelectionParametersTree();
	virtual ~CModelSelectionParametersTree();

	inline virtual const char* get_name() const=0;

	/** SG_PRINT's the tree of which this node is the base
	 *
	 * @param prefix_num a number of '\t' tabs that is put before each output
	 * to have a more readable print layout
	 */
	virtual void print(const char* prefix);

	/** method to recursively delete the complete tree of which this node is
	 * the root */
	virtual void destroy();

protected:
	 /** @return true if it has children */
	bool has_children();

protected:
	DynArray<CModelSelectionParametersTree*> m_child_nodes;
};

}
#endif /* __MODELSELECTIONPARAMETERSTREE_H_ */
