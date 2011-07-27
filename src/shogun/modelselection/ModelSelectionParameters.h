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

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

class CParameterCombination;

enum ERangeType
{
	R_LINEAR, R_EXP, R_LOG
};

/** value type of a model selection parameter node */
enum EMSParamType
{
	/** no type */
	MSPT_NONE=0,

	/* float64_t */
	MSPT_FLOAT64,

	/* int32_t */
	MSPT_INT32,
};

/**
 * @brief Class to select parameters and their ranges for model selection. The
 * structure is organized as a tree with different kinds of nodes, depending on
 * the values of its member variables of name and CSGObject.
 *
 * -root node: no name and no CSGObject, may have children
 *
 * -CSGObject node: has name and a CSGObject, may have children which are the
 * parameters of the CSGObject. CSGObjects are SG_REF'ed/SG_UNREF'ed
 *
 * -value node: a node with a (parameter) name and an array of values for that
 * parameter. These ranges may be set using build_values().
 * This node is always a leaf.
 *
 * After a (legal!) tree is constructed with the append_child method, all
 * possible combinations that are implied by this tree may be extracted with the
 * get_combinations method. It generates a set of trees (different kind than
 * this one) that contain the instantiated parameter combinations.
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

	/** appends a child to this tree. only possible if this is no value node
	 *
	 * @param child child to append
	 */
	void append_child(CModelSelectionParameters* child);

	/** setter for values of this node.
	 * If the latter are not possible to be produced by set_range, a vector may
	 * be specified directly.
	 *
	 * @param values value vector
	 */
	void set_values(SGVector<void> values);

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
	CDynamicObjectArray<CParameterCombination>* get_combinations();

	/** float64_t wrapper for build_values() */
	void build_values(float64_t min, float64_t max, ERangeType type,
			float64_t step=1.0, float64_t type_base=2.0);

	/** int32_t wrapper for build_values() */
	void build_values(int32_t min, int32_t max, ERangeType type, int32_t step=1,
			int32_t type_base=2);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "ModelSelectionParameters";
	}

private:
	void init();

	/** deletes the values vector with respect to its type */
	void delete_values();

	/** generic wrapper for create_range_array */
	void build_values(EMSParamType param_type, void* min, void* max,
			ERangeType type, void* step, void* type_base);

protected:
	/** checks if this node has children
	 *
	 * @return true if it has children
	 */
	bool has_children() const
	{
		return m_child_nodes->get_num_elements()>0;
	}

private:
	CSGObject* m_sgobject;
	const char* m_node_name;
	SGVector<void> m_values;
	CDynamicObjectArray<CModelSelectionParameters>* m_child_nodes;
	EMSParamType m_value_type;
};

/** Creates an array of values specified by the parameters.
 * A minimum and a maximum is specified, step interval, and an
 * ERangeType (s. above) of the range, which is used to fill an array with
 * concrete values. For some range types, a base is required.
 * All values are given by void pointers to them (type conversion is done
 * via m_value_type variable).
 *
 * @param min minimum of desired range. Requires min<max
 * @param max maximum of desired range. Requires min<max
 * @param type the way the values are created, see ERangeType
 * @param step increment instaval for the values
 * @param type_base base for EXP or LOG ranges
 */
template <class T> SGVector<T> create_range_array(T min, T max,
		ERangeType type, T step, T type_base)
{
	if (max<min)
		SG_SERROR("unable build values: max=%f < min=%f\n", max, min);

	/* create value vector */
	index_t num_values=CMath::round(max-min)/step+1;
	SGVector<T> result(num_values);

	/* fill array */
	for (index_t i=0; i<num_values; ++i)
	{
		T current=min+i*step;

		switch (type)
		{
		case R_LINEAR:
			result.vector[i]=current;
			break;
		case R_EXP:
			result.vector[i]=CMath::pow((float64_t)type_base, current);
			break;
		case R_LOG:
			if (current<=0)
				SG_SERROR("log(x) with x=%f\n", current);

			/* custom base b: log_b(i*step)=log_2(i*step)/log_2(b) */
			result.vector[i]=CMath::log2(current)/CMath::log2(type_base);
			break;
		default:
			SG_SERROR("unknown range type!\n");
			break;
		}
	}

	return result;
}

}
#endif /* __MODELSELECTIONPARAMETERS_H_ */
