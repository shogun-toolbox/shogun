/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTIONPARAMETERS_H_
#define __MODELSELECTIONPARAMETERS_H_

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
 *
 * -root node: no name and no CSGObject, may have children. Note that root nodes
 * call destroy() upon destructor call and destroy the complete tree
 *
 * -placeholder node: only has a name and children, used to bundle parameters
 * that belong to the learning machine directly, like "kernel" or "C"
 *
 * -CSGObject node: has name and a CSGObject, has children which are the
 * parameters of the CSGObject. CSGObjects are SG_REF'ed/SG_UNREF'ed
 *
 * -value node: a node with a (parameter) name and an array of values for that
 * parameter. These ranges may be set using set_range(). This node is always a
 * leaf
 *
 * After a (legal!) tree is constructed with the append_child method, all
 * possible combinations that are implied by this tree may be extracted with the
 * get_combinations method. It generates a set of trees (different kind than
 * this one) that contain the instanciated parameter combinations.
 *
 * Note again that CSGObjects are SG_REF'ed/SG_UNREF'ed. The method
 * get_combinations() does not do any more. So the produced trees of parameter
 * combinations have to be processes BEFORE this tree is deleted, or there will
 * be an error if the CSGObjects are not referenced elsewhere
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
	CModelSelectionParameters(const char* node_name);

	/** constructor for a CSGObject node
	 *
	 * @param sgobject the CSGObject for this node. Is SG_REF'ed
	 * @name name of the parameter of the CSGObject
	 */
	CModelSelectionParameters(const char* node_name, CSGObject* sgobject);

	/** destructor. If set, deletes data array and SG_UNREF's the CSGObject */
	~CModelSelectionParameters();

	/** method to recursively unref all nods of this tree */
	void unref_childs();

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
	 * Calling this function transforms a placeholder node (without children)
	 * into a value node.
	 *
	 * @param min minimum of desired range. Requires min<max
	 * @param max maximum of desired range. Requires min<max
	 * @param type the way the values are created, see ERangeType
	 * @param step increment instaval for the values
	 * @param type_base base for EXP or LOG ranges
	 */
	void set_range(float64_t min, float64_t max, ERangeType type,
			float64_t step=1, float64_t type_base=2);

	/** setter for values of this node.
	 * If the latter are not possible to be produced by set_range, a vector may
	 * be specified directly.
	 *
	 * @param values value vector
	 */
	void set_values(SGVector<float64_t> values);

	/** SG_PRINT's the tree of which this node is the base
	 *
	 * @param prefix_num a number of '\t' tabs that is put before each output
	 * to have a more readable print layout
	 */
	void print_tree(int prefix_num=0);

	/** most important method. If the tree was regarding node types and
	 * structure, a set of trees which contain all combinations of parameters
	 * that are implied by this tree is generated.
	 *
	 * @result result all trees of parameter combinations are put into here
	 *
	 */
	DynArray<CParameterCombination*>* get_combinations();

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "ModelSelectionParameters";
	}

private:
	void init();

protected:
	/** checks if this node has children
	 *
	 * @return true if it has children
	 */
	bool has_children()
	{
		return m_child_nodes.get_num_elements()>0;
	}

private:
	CSGObject* m_sgobject;
	const char* m_node_name;
	SGVector<float64_t> m_values;
	DynArray<CModelSelectionParameters*> m_child_nodes;
};

}
#endif /* __MODELSELECTIONPARAMETERS_H_ */
