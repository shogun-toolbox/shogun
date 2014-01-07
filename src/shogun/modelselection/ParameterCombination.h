/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __PARAMETERCOMBINATION_H__
#define __PARAMETERCOMBINATION_H__

#include <lib/DynamicObjectArray.h>
#include <lib/Map.h>

namespace shogun
{
class CModelSelectionParameters;
class CMachine;
class Parameter;

/** @brief Class that holds ONE combination of parameters for a learning
 * machine. The structure is organized as a tree. Every node may hold a name or
 * an instance of a Parameter class. Nodes may have children. The nodes are
 * organized in such way, that every parameter of a model for model selection
 * has one node and sub-parameters are stored in sub-nodes. Using a tree of this
 * class, parameters of models may easily be set. There are these types of
 * nodes:
 *
 * -root node: no name and no Parameter instance, every tree has such a node as
 * root. Has children.
 *
 * -Parameter node: a node with no name and an instance of Parameter, filled
 * with one or more values. There may be different elements in these Parameter
 * instances. Parameter nodes may have children with sub-parameters.
 *
 * Again: Leafs of the tree may only be Parameter nodes.
 */
class CParameterCombination : public CSGObject
{
friend class CModelSelectionParameters;

public:
	/** constructor for a root node */
	CParameterCombination();

	/** constructor for a parameter node
	 *
	 * @param param parameter node
	 */
	CParameterCombination(Parameter* param);

	/** constructor for an object. Builds parameter combination of the gradient
	 * parameters.
	 *
	 * It adds parameters recursively starting from given object (given object
	 * becomes the root node).
	 *
	 * @param obj object to build parameter combination
	 */
	CParameterCombination(CSGObject* obj);

	/** destructor also recursively destroys complete tree (SG_UNREF of child
	 * nodes)
	 */
	virtual ~CParameterCombination();

	/** Prints a representation of the current node
	 *
	 * @param prefix_num number of tabs that will be prefixed for every output.
	 * At each recursion level, one is added.
	 */
	void print_tree(int prefix_num=0) const;

	/** applies this parameter tree to a parameter instance
	 *
	 * Recursively iterates over all children of the tree and sets model
	 * selection parameters of children to sub-parameters
	 *
	 * @param parameter Parameter instance to apply parameter tree to
	 */
	void apply_to_modsel_parameter(Parameter* parameter) const;

	/** applies this parameter tree to a learning machine (wrapper for
	 * apply_to_modesel_parameter() method)
	 *
	 * @param machine learning machine to apply parameter tree to
	 */
	void apply_to_machine(CMachine* machine) const;

	/** appends a child to this node
	 *
	 * @param child child to append
	 */
	void append_child(CParameterCombination* child);

	/** Adds (copies of) all children of given node
	 *
	 * @param node (copies of) children of given node are added to this one
	 */
	void merge_with(CParameterCombination* node);

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
	 * @result result set of tree combinations
	 */
	static CDynamicObjectArray* leaf_sets_multiplication(
			const CDynamicObjectArray& sets,
			const CParameterCombination* new_root);

	/** Sets specific parameter to specified value.
	 *
	 * @param name Name of parameter
	 * @param value value to be set
	 * @param parent The CSObject that directly holds this parameter
	 * @param index index if the parameter is a vector
	 *
	 * @return bool true if value successfully set.
	 */
	template <typename T>
	bool set_parameter(const char* name,
			T value, CSGObject* parent, index_t index = -1)
	{
		bool match = false;

		if (m_param)
		{
			for (index_t i = 0; i < m_param->get_num_parameters(); ++i)
			{
					void* param = m_param->get_parameter(i)->m_parameter;

					if (m_param->get_parameter(i)->m_datatype.m_ptype
							==PT_SGOBJECT)
					{
						if (parent == (*((CSGObject**)param)))
							match = true;
					}

			}

		}

		bool result = false;

		for (index_t i = 0; i < m_child_nodes->get_num_elements(); ++i)
		{
			CParameterCombination* child = (CParameterCombination*)
					m_child_nodes->get_element(i);

			if (!match)
				 result |= child->set_parameter(name, value, parent, index);

			else
				 result |= child->set_parameter_helper(name, value, index);

			SG_UNREF(child);

		}

		return result;
	}

	/** Gets specific parameter by name.
	 *
	 * @param name Name of parameter
	 * @param parent The CSObject that directly holds this parameter
	 *
	 * return specified parameter. NULL if not found.
	 */
	TParameter* get_parameter(const char* name, CSGObject* parent);

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
	virtual const char* get_name() const
	{
		return "ParameterCombination";
	}

	/** returns total length of the parameters in combination
	 *
	 * @return total length of the parameters in combination
	 */
	virtual uint32_t get_parameters_length() { return m_parameters_length; }

	/** builds map, which contains parameters and its values.
	 *
	 * This method adds to map only parameters of floating point type.
	 *
	 * @param values_map map, which contains parameters and its values
	 */
	virtual void build_parameter_values_map(
			CMap<TParameter*, SGVector<float64_t> >* values_map);

	/** builds map, which contains parameters and its parents
	 *
	 * @param parent_map map, which contains parameters and its parents
	 */
	virtual void build_parameter_parent_map(
			CMap<TParameter*, CSGObject*>* parent_map);

protected:
	/** Takes a set of sets of (non-value) trees and returns a set with all
	 * combinations of the elements, where only combinations of trees with
	 * different names are built.
	 *
	 * @param sets set of sets of CParameterCombination instances which
	 * represent the trees to be multiplied
	 * @param new_root this new root is put in front of all products
	 * @return set of trees with the given root as root and all combinations
	 * of the trees in the sets as children
	 */
	static CDynamicObjectArray* non_value_tree_multiplication(
				const CDynamicObjectArray* sets,
				const CParameterCombination* new_root);

	/** Takes a set of sets of trees and extracts all trees with a given name.
	 * Assumes that in a (inner) set, all trees have the same name on their
	 * single parameter. Used by get_combinations
	 *
	 * @param sets set of sets of CParameterCombination instances to search in
	 * @param desired_name tree with this name is searched
	 * @return set of trees with the desired name
	 */
	static CDynamicObjectArray* extract_trees_with_name(
			const CDynamicObjectArray* sets, const char* desired_name);

	/** Gets parameter by name in current node.
	 *
	 * @param name name of parameter
	 * @return parameter. Null if not found.
	 */
	TParameter* get_parameter_helper(const char* name);

	/** Sets parameter by name in current node.
	 *
	 * @param name name of parameter
	 * @param value of parameter
	 * @param index index if parameter is a vector
	 *
	 * @return true if found.
	 */
	bool set_parameter_helper(const char* name, bool value, index_t index);

	/** Sets parameter by name in current node.
	 *
	 * @param name name of parameter
	 * @param value of parameter
	 * @param index index if parameter is a vector
	 *
	 * @return true if found.
	 */
	bool set_parameter_helper(const char* name, int32_t value, index_t index);

	/** Sets parameter by name in current node.
	 *
	 * @param name name of parameter
	 * @param value of parameter
	 * @param index index if parameter is a vector
	 *
	 * @return true if found.
	 */
	bool set_parameter_helper(const char* name, float64_t value, index_t index);

private:
	void init();

protected:
	/** parameter of combination */
	Parameter* m_param;

	/** child parameters */
	CDynamicObjectArray* m_child_nodes;

	/** total length of the parameters in combination */
	uint32_t m_parameters_length;
};
}
#endif /* __PARAMETERCOMBINATION_H__ */
