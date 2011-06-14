/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTIONPARAMETERSDISCRETENODE_H_
#define __MODELSELECTIONPARAMETERSDISCRETENODE_H_

#include "modelselection/ModelSelectionParametersTree.h"

namespace shogun
{

template<class T> class CModelSelectionParametersDiscreteNode: public CModelSelectionParametersTree
{
public:
	CModelSelectionParametersDiscreteNode()	{}
	CModelSelectionParametersDiscreteNode(SGVector<T> values) : m_values(values) {}

	virtual ~CModelSelectionParametersDiscreteNode() { delete m_values.vector; }

	void print(const char* prefix)
	{
		SG_PRINT("%sdiscrete values:", prefix);

		/* TODO handly type chaos */
//		CMath::display_vector(m_values.vector, m_values.length);

		CModelSelectionParametersTree::print(prefix);
	}

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "ModelSelectionParametersDiscreteNode";
	}

private:
	SGVector<T> m_values;
};

}
#endif /* __MODELSELECTIONPARAMETERSDISCRETENODE_H_ */
