/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Written (W) 2012 Jacob Walker
 *
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>
#include <lib/DataType.h>
#include <base/Parameter.h>
#include <base/DynArray.h>
#include <lib/Set.h>

using namespace shogun;

CModelSelectionParameters::CModelSelectionParameters()
{
	init();
}

CModelSelectionParameters::CModelSelectionParameters(const char* node_name)
{
	init();

	m_node_name=node_name;
}

CModelSelectionParameters::CModelSelectionParameters(const char* node_name,
		CSGObject* sgobject)
{
	init();

	m_node_name=node_name;
	m_sgobject=sgobject;
	SG_REF(sgobject);
}

void CModelSelectionParameters::init()
{
	m_node_name=NULL;
	m_sgobject=NULL;
	m_child_nodes=new CDynamicObjectArray();
	SG_REF(m_child_nodes);
	m_value_type=MSPT_NONE;
	m_values=NULL;
	m_values_length=0;

	/* no parameter registering. These parameter nodes will not be serialized */
}

CModelSelectionParameters::~CModelSelectionParameters()
{
	SG_UNREF(m_child_nodes);
	SG_UNREF(m_sgobject);

	delete_values();
}

void CModelSelectionParameters::append_child(CModelSelectionParameters* child)
{
	/* only possible if there are no values set */
	if (m_values)
		SG_ERROR("not possible to append child: there already is a range\n")

	/* do a basic check if the add is possible */
	if (m_sgobject)
	{
		/* (does this node's CSGObject contain a parameter with the name of the
		 * child?) to prevent problems when trying to set parameters that do not
		 * exist */
		if (child->m_node_name)
		{
			if (!m_sgobject->m_parameters->contains_parameter(child->m_node_name))
			{
				SG_ERROR("Not possible to add child, node with CSGObject \"%s\""
						" does not contain a parameter called \"%s\"\n",
						m_sgobject->get_name(), child->m_node_name);
			}
		}
		else
		{
			SG_ERROR("Not possible to add child which has no name.\n")
		}
	}

	m_child_nodes->append_element(child);
}

template <class T>
void CModelSelectionParameters::set_values(const SGVector<T>& values,
		EMSParamType value_type)
{
	/* possibly delete old range values */
	delete_values();
	m_values=values.vector;
	m_values_length=values.vlen;
	m_value_type=value_type;
}

void CModelSelectionParameters::build_values(float64_t min, float64_t max,
		ERangeType type, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void CModelSelectionParameters::build_values_vector(float64_t min, float64_t max,
		ERangeType type, void* vector, index_t* size, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64_VECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector_length = size;
	m_vector = vector;
}

void CModelSelectionParameters::build_values_sgvector(float64_t min, float64_t max,
		ERangeType type, void* vector, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64_SGVECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector = vector;
}

void CModelSelectionParameters::build_values(int32_t min, int32_t max,
		ERangeType type, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void CModelSelectionParameters::build_values_vector(int32_t min, int32_t max,
		ERangeType type, void* vector, index_t* size, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32_VECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector_length = size;
	m_vector = vector;
}

void CModelSelectionParameters::build_values_sgvector(int32_t min, int32_t max,
		ERangeType type, void* vector, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32_SGVECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector = vector;
}

void CModelSelectionParameters::build_values(EMSParamType value_type, void* min,
		void* max, ERangeType type, void* step, void* type_base)
{
	if (m_sgobject || has_children())
	{
		SG_ERROR("unable to set range for an CSGObject model selection "
				"parameter\n");
	}

	/* possibly delete old range values */
	delete_values();

	/* save new type */
	m_value_type=value_type;

	if (value_type==MSPT_FLOAT64 ||
			value_type==MSPT_FLOAT64_VECTOR
			|| value_type==MSPT_FLOAT64_SGVECTOR)
	{
		SGVector<float64_t> values=create_range_array<float64_t>(
				*((float64_t*)min),
				*((float64_t*)max),
				type,
				*((float64_t*)step),
				*((float64_t*)type_base));

		m_values=values.vector;
		m_values_length=values.vlen;
	}
	else if (value_type==MSPT_INT32 ||
			value_type==MSPT_INT32_VECTOR
			|| value_type==MSPT_INT32_SGVECTOR)
	{
		SGVector<int32_t> values=create_range_array<int32_t>(
				*((int32_t*)min),
				*((int32_t*)max),
				type,
				*((int32_t*)step),
				*((int32_t*)type_base));

		m_values=values.vector;
		m_values_length=values.vlen;
	}
	else if (value_type==MSPT_NONE)
	{
		SG_ERROR("Value node has no type!\n")
	}
	else
	{
		SG_ERROR("Unknown type for model selection parameter!\n")
	}
}

CParameterCombination* CModelSelectionParameters::get_single_combination(
		bool is_rand)
{
	/* If this is a value node, then randomly pick a value from the built
	 * range */
	if (m_values)
	{

		index_t i = 0;

		if (is_rand)
			i = CMath::random(0, m_values_length-1);

		Parameter* p=new Parameter();

		switch (m_value_type)
		{
		case MSPT_FLOAT64_SGVECTOR:
		{
			SGVector<float64_t>* param_vect = (SGVector<float64_t>*)m_vector;

			for (index_t j = 0; j < param_vect->vlen; j++)
			{
				if (is_rand)
					i = CMath::random(0, m_values_length-1);
				(*param_vect)[j] = ((float64_t*)m_values)[i];
			}
			p->add(param_vect, m_node_name);
			break;
		}
		case MSPT_FLOAT64_VECTOR:
		{
			float64_t* param_vect = (float64_t*)m_vector;

			for (index_t j = 0; j < *m_vector_length; j++)
			{
				if (is_rand)
					i = CMath::random(0, m_values_length-1);
				(param_vect)[j] = ((float64_t*)m_values)[i];
			}
			p->add_vector(&param_vect, m_vector_length, m_node_name);
			break;
		}
		case MSPT_INT32_SGVECTOR:
		{
			SGVector<int32_t>* param_vect = (SGVector<int32_t>*)m_vector;

			for (index_t j = 0; j < param_vect->vlen; j++)
			{
				if (is_rand)
					i = CMath::random(0, m_values_length-1);
				(*param_vect)[j] = ((int32_t*)m_values)[i];
			}
			p->add(param_vect, m_node_name);
			break;
		}
		case MSPT_INT32_VECTOR:
		{
			int32_t* param_vect = (int32_t*)m_vector;

			for (index_t j = 0; j < *m_vector_length; j++)
			{
				if (is_rand)
					i = CMath::random(0, m_values_length-1);
				(param_vect)[j] = ((int32_t*)m_values)[i];
			}
			p->add_vector(&param_vect, m_vector_length, m_node_name);
			break;
		}
		case MSPT_FLOAT64:
			p->add(&((float64_t*)m_values)[i], m_node_name);
			break;
		case MSPT_INT32:
			p->add(&((int32_t*)m_values)[i], m_node_name);;
			break;
		case MSPT_NONE:
			SG_ERROR("Value node has no type!\n")
			break;
		default:
			SG_ERROR("Unknown type for model selection parameter!\n")
			break;
		}

		return new CParameterCombination(p);
	}

	CParameterCombination* new_root=NULL;

	/*Complain if we have a bad node*/
	if (!((m_sgobject && m_node_name) || (!m_node_name && !m_sgobject)))
		SG_ERROR("Illegal CModelSelectionParameters node type.\n")

	/* Incorporate SGObject and root nodes with children*/
	if (m_child_nodes->get_num_elements())
	{

		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			p->add(&m_sgobject, m_node_name);
		new_root = new CParameterCombination(p);
		}

		else
			new_root = new CParameterCombination();

		for (index_t i = 0; i < m_child_nodes->get_num_elements(); ++i)
		{
			CModelSelectionParameters* current =
					(CModelSelectionParameters*)m_child_nodes->get_element(i);

			CParameterCombination* c = current->get_single_combination(is_rand);

			new_root->append_child(c);

			SG_UNREF(current);
		}

		return new_root;
	}

	/*Incorporate childless nodes*/
	else
	{

		if (m_sgobject)
		{
			Parameter* p = new Parameter();
			p->add(&m_sgobject, m_node_name);
			return new CParameterCombination(p);
		}

		else
		{
			new_root = new CParameterCombination();
			return new_root;
		}
	}

}



CDynamicObjectArray* CModelSelectionParameters::get_combinations(
		index_t num_prefix)
{
	char* prefix=SG_MALLOC(char, num_prefix+1);
	prefix[num_prefix]='\0';
	for (index_t i=0; i<num_prefix; ++i)
		prefix[i]='\t';

	SG_DEBUG("%s------>entering CModelSelectionParameters::get_combinations() "
			"for \"%s\"\n", prefix, m_node_name ? m_node_name : "root");
	CDynamicObjectArray* result=new CDynamicObjectArray();

	/* value case: node with values and no children.
	 * build trees of Parameter instances which each contain one value
	 */

	if (m_values)
	{
		for (index_t i=0; i<m_values_length; ++i)
		{
			// create tree with only one parameter element //
			Parameter* p=new Parameter();

			switch (m_value_type)
			{
			case MSPT_FLOAT64:
				p->add(&((float64_t*)m_values)[i], m_node_name);
				break;
			case MSPT_INT32:
				p->add(&((int32_t*)m_values)[i], m_node_name);;
				break;
			case MSPT_NONE:
				SG_ERROR("%sValue node has no type!\n", prefix)
				break;
			default:
				SG_ERROR("%sUnknown type for model selection parameter!\n",
						prefix);
				break;
			}

			result->append_element(new CParameterCombination(p));
		}

		SG_DEBUG("%s------>leaving CModelSelectionParameters::get_combinations()"
				"for \"%s\"\n", prefix, m_node_name ? m_node_name : "root");

		SG_FREE(prefix);
		return result;
	}


	/* two cases here, similar
	 * -case CSGObject:
	 * -case root node (no name, no values, but children
	 * build all permutations of the result trees of children with values and
	 * combine them iteratively children which are something different
	 */
	if (!((m_sgobject && m_node_name) || (!m_node_name && !m_sgobject)))
		SG_ERROR("%sIllegal CModelSelectionParameters node type.\n", prefix)

	/* only consider combinations if this node has children */
	if (m_child_nodes->get_num_elements())
	{
		/* split value and non-value child combinations */
		CDynamicObjectArray value_children;
		CDynamicObjectArray non_value_children;

		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			CModelSelectionParameters* current=
					(CModelSelectionParameters*)m_child_nodes->get_element(i);

			/* split children with values and children with other */
			if (current->m_values)
				value_children.append_element(current);
			else
				non_value_children.append_element(current);

			SG_UNREF(current);
		}

		/* extract all tree sets of all value children */
		CDynamicObjectArray value_node_sets;
		for (index_t i=0; i<value_children.get_num_elements(); ++i)
		{
			/* recursively get all combinations in a new array */
			CModelSelectionParameters* value_child=
					(CModelSelectionParameters*)value_children.get_element(i);
			value_node_sets.append_element(value_child->get_combinations(
					num_prefix+1));
			SG_UNREF(value_child);
		}

		/* build product of all these tree sets */

		/* new root node is needed for new trees, depends on current case */
		CParameterCombination* new_root=NULL;
		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			p->add(&m_sgobject, m_node_name);
			new_root=new CParameterCombination(p);
		}
		else
			new_root=new CParameterCombination();

		SG_REF(new_root);

		CDynamicObjectArray* value_combinations=
				CParameterCombination::leaf_sets_multiplication(value_node_sets,
						new_root);

		SG_UNREF(new_root);

		if (!non_value_children.get_num_elements())
			*result=*value_combinations;
		/* in the other case, the non-values have also to be treated, but
		 * combined iteratively */
		else
		{
			/* extract all tree sets of non-value nodes */
//			SG_PRINT("%sextracting combinations of non-value nodes\n", prefix)
			CDynamicObjectArray* non_value_combinations=
					new CDynamicObjectArray();
			for (index_t i=0; i<non_value_children.get_num_elements(); ++i)
			{
				/* recursively get all combinations in a new array */
				CModelSelectionParameters* non_value_child=
						(CModelSelectionParameters*)
						non_value_children.get_element(i);

//				SG_PRINT("%s\tcurrent non-value child\n", prefix)
//				non_value_child->print_tree(num_prefix+1);

				CDynamicObjectArray* current_combination=
						non_value_child->get_combinations(num_prefix+2);
				non_value_combinations->append_element(current_combination);
				SG_UNREF(non_value_child);

//				SG_PRINT("%s\tcombinations of non-value nodes:\n", prefix)
//				for (index_t j=0; j<current_combination->get_num_elements(); ++j)
//				{
//					CParameterCombination* c=(CParameterCombination*)
//							current_combination->get_element(j);
//					c->print_tree(num_prefix+2);
//					SG_UNREF(c);
//				}
			}
//			SG_PRINT("%sdone extracting combinations of non-value nodes\n",
//					prefix);

			/* Now, combine combinations of value and non-value nodes */

			/* if there are only non-value children, nothing is combined */
			if (!value_combinations->get_num_elements())
			{
				/* non-value children have to be multipied first, then, all
				 * these products are just appended */

				/* temporary new root is needed to put fron all product trees */
				if (m_sgobject)
				{
					Parameter* p=new Parameter();
					p->add(&m_sgobject, m_node_name);
					new_root=new CParameterCombination(p);
				}
				else
					new_root=new CParameterCombination();

				CDynamicObjectArray* non_value_products=
						CParameterCombination::non_value_tree_multiplication(
								non_value_combinations, new_root);

				SG_UNREF(new_root);

				SG_UNREF(non_value_combinations);
				non_value_combinations=non_value_products;

				/* append all non-value combinations to result */
				for (index_t i=0; i<non_value_combinations->get_num_elements(); ++i)
				{
					CParameterCombination* current=(CParameterCombination*)
							non_value_combinations->get_element(i);
					result->append_element(current);
					SG_UNREF(current);
				}
			}
			else
			{
				/* before combinations are built, produce products of non-value
				 * combinations. new root is temporarily needed to put front
				 * all new trees */
				if (m_sgobject)
				{
					Parameter* p=new Parameter();
					p->add(&m_sgobject, m_node_name);
					new_root=new CParameterCombination(p);
				}
				else
					new_root=new CParameterCombination();

				CDynamicObjectArray* non_value_products=
						CParameterCombination::non_value_tree_multiplication(
								non_value_combinations, new_root);

				SG_UNREF(new_root);

				SG_UNREF(non_value_combinations);
				non_value_combinations=non_value_products;

				for (index_t i=0; i<value_combinations->get_num_elements(); ++i)
				{
					CParameterCombination* current_value_tree=
							(CParameterCombination*)
							value_combinations->get_element(i);

					for (index_t j=0; j
							<non_value_combinations->get_num_elements(); ++j)
				{
						CParameterCombination* current_non_value_tree=
								(CParameterCombination*)
								non_value_combinations->get_element(j);

						/* copy current value tree and add all childs of non-
						 * value combination. Then add new node to result */
						CParameterCombination* value_copy=
								current_value_tree->copy_tree();

						value_copy->merge_with(current_non_value_tree);
						result->append_element(value_copy);

						SG_UNREF(current_non_value_tree);
					}

					SG_UNREF(current_value_tree);
				}
			}

			/* clean up*/
			SG_UNREF(non_value_combinations);
		}

		SG_UNREF(value_combinations);
	}
	else
	{
		/* if there are no children of a sgobject or root node, result is
		 * only one element (sgobject node) or empty (root node)
		 */
		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			p->add(&m_sgobject, m_node_name);
			result->append_element(new CParameterCombination(p));
		}
	}

//	SG_PRINT("%sresult is a set of %d elements:\n", prefix,
//			result->get_num_elements());
//	for (index_t i=0; i<result->get_num_elements(); ++i)
//	{
//		CParameterCombination* current=(CParameterCombination*)
//				result->get_element(i);
//		current->print_tree(num_prefix+1);
//		SG_UNREF(current);
//	}

	SG_DEBUG("%s------>leaving CModelSelectionParameters::get_combinations()"
			"for \"%s\"\n", prefix, m_node_name ? m_node_name : "root");
	SG_FREE(prefix);
	return result;
}

void CModelSelectionParameters::print_tree(int prefix_num)
{
	/* prefix is enlarged */
	char* prefix=SG_MALLOC(char, prefix_num+1);
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	if (has_children())
	{
		if (m_sgobject)
			SG_PRINT("%s%s:\"%s\"\n", prefix, m_node_name, m_sgobject->get_name())
		else
			SG_PRINT("%s%s with\n", prefix, m_node_name ? m_node_name : "root")

		/* now recursively print successors */

		/* cast safe because only CModelSelectionParameters are added to list */
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			CModelSelectionParameters* child=
					(CModelSelectionParameters*)m_child_nodes->get_element(i);
			child->print_tree(prefix_num+1);
			SG_UNREF(child);
		}
	}
	else
	{
		/* has to be a node with name and a numeric range or a single sg_object
		 * without children*/
		if (m_sgobject)
		{
			SG_PRINT("%s%s:\"%s\"\n", prefix, m_node_name, m_sgobject->get_name())
		}
		else
		{
			if (m_values)
			{
				// value node
				SG_PRINT("%s%s with values: ", prefix, m_node_name)

				switch (m_value_type)
				{
				case MSPT_FLOAT64: case MSPT_FLOAT64_VECTOR:
					case MSPT_FLOAT64_SGVECTOR:

					SGVector<float64_t>::display_vector((float64_t*)m_values,
							m_values_length);
					break;
				case MSPT_INT32: case MSPT_INT32_VECTOR:
					case MSPT_INT32_SGVECTOR:

					SGVector<int32_t>::display_vector((int32_t*)m_values,
							m_values_length);;
					break;
				case MSPT_NONE:
					SG_ERROR("Value node has no type!\n")
					break;
				default:
					SG_ERROR("Unknown type for model selection parameter!\n")
					break;
				}
			}
			else
				SG_PRINT("root\n")
		}
	}

	SG_FREE(prefix);
}

void CModelSelectionParameters::delete_values()
{
	if (m_values)
	{
		switch (m_value_type)
		{
		case MSPT_FLOAT64: case MSPT_FLOAT64_VECTOR:
			case MSPT_FLOAT64_SGVECTOR:

			SG_FREE((float64_t*)m_values);
			break;
		case MSPT_INT32: case MSPT_INT32_VECTOR:
			case MSPT_INT32_SGVECTOR:

			SG_FREE((int32_t*)m_values);
			break;
		case MSPT_NONE:
			SG_ERROR("Value node has no type!\n")
			break;
		default:
			SG_ERROR("Unknown type for model selection parameter!\n")
			break;
		}
	}
}
