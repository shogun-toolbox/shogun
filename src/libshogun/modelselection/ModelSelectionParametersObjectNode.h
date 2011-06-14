/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTIONPARAMETERSOBJECTNODE_H_
#define __MODELSELECTIONPARAMETERSOBJECTNODE_H_

#include "modelselection/ModelSelectionParametersTree.h"

namespace shogun
{

class CModelSelectionParametersObjectNode : public CModelSelectionParametersTree
{
public:
	CModelSelectionParametersObjectNode();
	CModelSelectionParametersObjectNode(CSGObject* sgobject);

	virtual ~CModelSelectionParametersObjectNode();

	/** appends a child to this tree.
	 *
	 * @param child child to append
	 */
	void append_child(CModelSelectionParametersTree* child);

	virtual void print(const char* prefix);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "CModelSelectionParametersObjectNode";
	}

private:
	CSGObject* m_sgobject;
};

}

#endif /* __MODELSELECTIONPARAMETERSOBJECTNODE_H_ */
