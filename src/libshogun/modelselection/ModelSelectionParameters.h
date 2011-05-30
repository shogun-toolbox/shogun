/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __CMODELSELECTIONPARAMETERS_H_
#define __CMODELSELECTIONPARAMETERS_H_

#include "base/SGObject.h"
#include "base/DynArray.h"
#include "modelselection/ParameterCombination.h"

namespace shogun
{

enum ERangeType
{
	R_LINEAR, R_EXP, R_LOG
};

/**
 * @brief Class to select parameters and their ranges for model selection. The
 * structure is organized as a tree with different kinds of nodes, depending on
 * the values of its member variables of name and CSGObject.
 * -root node: no name and no CSGObject, may have childs
 * -placeholder node: only has a name and children, used to bundle parameters
 * that belong to the learning machine directly, like "kernel" or "C"
 * -CSGObject node: no name, but a CSGObject, has children which are the
 * parameters of the CSGObject
 * -value node: a node with a (parameter) name and an array of values for that
 * parameter. These ranges may be set using set_range(). This nod is always a
 * leaf
 *
 * After a (legal!) tree is constructed with the append_child method, all
 * possible combinations that are implied by this tree may be extracted with the
 * get_combinatiosn method. It generates a set of trees (different kind than
 * this one) that contain the instanciated parameter combinations.
 *
 */
class CModelSelectionParameters: public CSGObject
{
public:
	/** constructor for a root node */
	CModelSelectionParameters();

	/** constructor for a value node
	 *
	 * @param name name of the parameter the values will belong to
	 */
	CModelSelectionParameters(char* name);

	/** constructor for a CSGObject node
	 *
	 * @param sgobject the CSGObject for this node. Is SG_REF'ed
	 */
	CModelSelectionParameters(CSGObject* sgobject);

	/** destructor. If set, deletes data array and SG_UNREF's the CSGObject */
	~CModelSelectionParameters();

	/** method to recursively delete the complete tree of which this node is
	 * the root */
	void destroy();

	/** appends a child to this tree. only possible if this is no value node
	 *
	 * @param child child to append
	 */
	void append_child(CModelSelectionParameters* child);

	/** setter for the range of this node. Only possible if this is a value
	 * node. A minimum and a maximum is specified, step interval, and an
	 * ERangeType (s. above) of the range, which is used to fill an array with
	 * concrete values. For some range types, a base is required
	 *
	 * Calling this function transforms a placeholder node (without childs) into
	 * a value node.
	 *
	 * @param min minimum of desired range. Requires min<max
	 * @param max maximum of desired range. Requires min<max
	 * @param type the way the values are created, see ERangeType
	 * @param step increment instaval for the values
	 * @param type_base base for EXP or LOG ranges
	 */
	void set_range(float64_t min, float64_t max, ERangeType type,
			float64_t step=1, float64_t type_base=2);

	/** SG_PRINT's the tree of which this node is the base
	 *
	 * @param prefix_num a number of '\t' tabs that is put before each output
	 * to have a more readable print layout
	 */
	void print(int prefix_num=0);

	/** most important method. If the tree was regarding node types and
	 * structure, a set of trees which contain all combinations of parameters
	 * that are implied by this tree is generated.
	 *
	 * @param result all trees of parameter combinations are put into here
	 *
	 */
	void get_combinations(DynArray<CParameterCombination*>& result);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "ModelSelectionParameters";
	}

protected:
	/** checks if this node has children
	 *
	 * @return true if it has children
	 */
	bool has_childs()
	{
		return m_child_nodes.get_num_elements()>0;
	}

private:
	char* m_node_name;
	SGVector<float64_t>* m_values;
	CSGObject* m_sgobject;
	DynArray<CModelSelectionParameters*> m_child_nodes;
};

}
#endif /* __CMODELSELECTIONPARAMETERS_H_ */
