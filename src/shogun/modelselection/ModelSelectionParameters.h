/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn,
 *          Yuyu Zhang, Evgeniy Andreev, Thoralf Klein, Leon Kuchenbecker
 */

#ifndef __MODELSELECTIONPARAMETERS_H_
#define __MODELSELECTIONPARAMETERS_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class ParameterCombination;

/** type of range */
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

	MSPT_FLOAT64_VECTOR,

	MSPT_INT32_VECTOR,

	MSPT_FLOAT64_SGVECTOR,

	MSPT_INT32_SGVECTOR,
};

/**
 * @brief Class to select parameters and their ranges for model selection. The
 * structure is organized as a tree with different kinds of nodes, depending on
 * the values of its member variables of name and SGObject.
 *
 * -root node: no name and no SGObject, may have children
 *
 * -SGObject node: has name and a SGObject, may have children which are the
 * parameters of the SGObject. SGObjects are SG_REF'ed/SG_UNREF'ed
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
class ModelSelectionParameters: public RandomMixin<SGObject>
{
public:
	/** constructor for a root node */
	ModelSelectionParameters();

	/** constructor for a value node
	 *
	 * @param node_name name of the parameter the values will belong to
	 */
	ModelSelectionParameters(const char* node_name);

	/** constructor for a SGObject node
	 *
	 * @param sgobject the SGObject for this node. Is SG_REF'ed
	 * @name name of the parameter of the SGObject
	 */
	ModelSelectionParameters(const char* node_name, std::shared_ptr<SGObject> sgobject);

	/** destructor. If set, deletes data array and SG_UNREF's the SGObject */
	~ModelSelectionParameters();

	/** appends a child to this tree. only possible if this is no value node
	 *
	 * @param child child to append
	 */
	void append_child(std::shared_ptr<ModelSelectionParameters> child);

	/** setter for values of this node.
	 * If the latter are not possible to be produced by set_range, a vector may
	 * be specified directly.
	 *
	 * @param values value vector. no ref counting takes place here
	 * @param value_type type of the provided vector
	 */
	template <class T>
	void set_values(const SGVector<T>& values, EMSParamType value_type)
	{
	    /* possibly delete old range values */
	    delete_values();
	    m_values=values.vector;
	    m_values_length=values.vlen;
	    m_value_type=value_type;
	}

	/** prints the tree of which this node is the base
	 *
	 * @param prefix_num a number of tabs that is put before each output
	 * to have a more readable print layout
	 */
	void print_tree(int prefix_num=0);

	/** most important method. If the tree was regarding node types and
	 * structure, a set of trees which contain all combinations of parameters
	 * that are implied by this tree is generated.
	 *
	 * @param prefix_num a number of tabs that is put before each output
	 * to have a more readable print layout
	 * @return result all trees of parameter combinations are put into here
	 */
	std::shared_ptr<DynamicObjectArray> get_combinations(index_t prefix_num=1);

	/** Instead of generating an array of combinations, get_single_combination
	 * generates a single  combination of parameters. The choice of
	 * values is constrained by the value arrays built by build_values.
	 * The choice of values may be random. If not, the lowest possible
	 * values are chosen.
	 *
	 * @param rand Is our choice random?
	 *
	 * @return parameter tree of random parameter values.
	 */
	std::shared_ptr<ParameterCombination> get_single_combination(bool rand = true);

	/** float64_t wrapper for build_values() */
	void build_values(float64_t min, float64_t max, ERangeType type,
			float64_t step=1.0, float64_t type_base=2.0);

	void build_values_vector(float64_t min, float64_t max, ERangeType type,
			void* vector, index_t* size, float64_t step=1.0,
			float64_t type_base=2.0);

	void build_values_sgvector(float64_t min, float64_t max, ERangeType type,
			void* vector, float64_t step=1.0, float64_t type_base=2.0);

	/** int32_t wrapper for build_values() */
	void build_values(int32_t min, int32_t max, ERangeType type, int32_t step=1,
			int32_t type_base=2);

	void build_values_vector(int32_t min, int32_t max, ERangeType type,
			void* vector, index_t* size, int32_t step=1,
			int32_t type_base=2);

	void build_values_sgvector(int32_t min, int32_t max, ERangeType type, void* vector,
			int32_t step=1, int32_t type_base=2);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
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
	std::shared_ptr<SGObject> m_sgobject;
	const char* m_node_name;
	void* m_values;
	index_t m_values_length;
	index_t* m_vector_length;
	std::shared_ptr<DynamicObjectArray> m_child_nodes;
	EMSParamType m_value_type;
	void*	m_vector;
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
		error("unable build values: max={} < min={}", max, min);

	/* create value vector, no ref-counting */
	index_t num_values=Math::round((max-min)/step)+1;
	SGVector<T> result(num_values, false);

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
			result.vector[i]=Math::pow((float64_t)type_base, current);
			break;
		case R_LOG:
			if (current<=0)
				error("log(x) with x={}", current);

			/* custom base b: log_b(i*step)=log_2(i*step)/log_2(b) */
			result.vector[i]=Math::log2(current)/Math::log2(type_base);
			break;
		default:
			error("unknown range type!");
			break;
		}
	}

	return result;
}

}
#endif /* __MODELSELECTIONPARAMETERS_H_ */
