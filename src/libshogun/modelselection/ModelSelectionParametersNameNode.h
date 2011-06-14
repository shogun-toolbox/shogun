/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTIONPARAMETERSNAMENODE_H_
#define __MODELSELECTIONPARAMETERSNAMENODE_H_

#include "modelselection/ModelSelectionParametersTree.h"

namespace shogun
{

class CModelSelectionParametersNameNode: public CModelSelectionParametersTree
{
public:
	CModelSelectionParametersNameNode();
	CModelSelectionParametersNameNode(const char* node_name);
	virtual ~CModelSelectionParametersNameNode();

	/** appends a child to this tree.
	 *
	 * @param child child to append
	 */
	void append_child(CModelSelectionParametersTree* child);

	virtual void print(const char* prefix);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "ModelSelectionParametersNameNode";
	}

private:
	const char* m_node_name;
};

}
#endif /* __MODELSELECTIONPARAMETERSNAMENODE_H_ */
