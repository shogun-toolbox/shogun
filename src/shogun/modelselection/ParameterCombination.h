/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __PARAMETERCOMBINATION_H__
#define __PARAMETERCOMBINATION_H__

#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

class CModelSelectionParameters;
class CMachine;
class Parameter;

/**
 * @brief class that holds ONE combination of parameters for a learning machine.
 * The structure is organized as a tree. Every node may hold a name or an
 * instance of a Parameter class. Nodes may have children. The nodes are
 * organized in such way, that every parameter of a model for model selection
 * has one node and sub-parameters are stored in sub-nodes. Using a tree of this
 * class, parameters of models may easily be set.
 * There are these types of nodes:
 *
 * -root node: no name and no Parameter instance, every tree has such a node as
 * root. Has children.
 *
 * -Parameter node: a node with no name and an instance of Parameter, filled
 * with one or more values. There may be different elements in these Parameter
 * instances. Parameter nodes may have children with sub-parameters.
 *
 * Again: Leafs of the tree may only be Parameter nodes.
 *
 */
class CParameterCombination : public CSGObject
{

friend class CModelSelectionParameters;

public:
	/** constructor for a root node */
	CParameterCombination();

	/** Prints a representation of the current node
	 *
	 * @param prefix number of '\t' signs that will be prefixed for every output.
	 * At each recursion level, one is added.
	 */
	void print_tree(int prefix_num=0) const;

	/** constructor for a Parameter node */
	CParameterCombination(Parameter* param);

	/** destructor
	 * also recursively destroys complete tree (SG_UNREF of child nodes) */
	virtual ~CParameterCombination();

	/** applies this parameter tree to a parameter instance
	 *
	 * @param parameter Parameter instance to apply parameter tree to
	 */
	void apply_to_parameter(Parameter* parameter) const;

	/**applies this parameter tree to a learning machine
	 * (wrapper for apply_to_parameter() method)
	 *
	 * @param machine learning machine to apply parameter tree to
	 */
	void apply_to_machine(CMachine* machine) const;

	/** appends a child to this node
	 *
	 * @param child child to append
	 */
	void append_child(CParameterCombination* child);

	/** Copies the complete tree of this node. Note that nodes are actually
	 * copied. If this is a parameter node, a NEW Parameter instance to the same
	 * data is created in the copy
	 *
	 * @return copy of the tree with this node as root as described above
	 */
	CParameterCombination* copy_tree() const;

	/** Takes a set of sets of leafs nodes (!) and produces a set of instances
	 * of this class that contain every combination of the parameters in the leaf
	 * nodes in their Parameter variables. All combinations are put into a newly
	 * created tree. The root of this tree will be a copy of a specified node
	 *
	 * created Parameter instances are added to the result set.
	 *
	 * @param sets Set of sets of leafs to combine
	 * @param new_root root node that is copied and put as root into all result
	 * trees
	 * @param result result set of tree combinations
	 */
	static CDynamicObjectArray<CParameterCombination>* leaf_sets_multiplication(
			const CDynamicObjectArray<CDynamicObjectArray<CParameterCombination> >& sets,
			const CParameterCombination* new_root);

	/** checks whether this node has children
	 *
	 * @return true if node has children
	 */
	bool has_children() const
	{
		return m_child_nodes->get_num_elements()>0;
	}

	/** Returns a newly created array with pointers to newly created Parameter
	 * instances, which contain all combinations of the provided Parameters.
	 *
	 * @param set_1 array of Parameter instances
	 * @param set_2 array of Parameter instances
	 * @return result array with all combinations
	 */
	static DynArray<Parameter*>* parameter_set_multiplication(
			const DynArray<Parameter*>& set_1,
			const DynArray<Parameter*>& set_2);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "ParameterCombination";
	}

private:
	void init();

private:
	Parameter* m_param;
	CDynamicObjectArray<CParameterCombination>* m_child_nodes;
};
}

#endif /* __PARAMETERCOMBINATION_H__ */
